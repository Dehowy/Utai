"""MIDI-guided hard mute — the 关键出装 post-hoc rest cleanup.

For every frame where MIDI note == 0 (SP/rest), force audio amplitude to 0
with a smooth 5-10 ms fade at boundaries to avoid clicks.

The model was trained with SP=0 force but L1 never reaches exact zero;
RVC's HiFi-GAN then decodes residual small HuBERT into electric/noise artifacts.
Since the score tells us EXACTLY when a frame should be silent, we just
force-mute those frames at inference time.

Usage:
    python scripts/apply_midi_mute.py <input.wav> <output.wav> <utt_id>
    # auto-finds pitch/duration tensors from data/processed/<dataset>/
"""
from __future__ import annotations
import sys, argparse, os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

V2 = Path(__file__).resolve().parents[1]
PROCESSED = V2 / "data" / "processed"

# Audio frame rate mapping: RVC outputs 48kHz, HuBERT runs 50fps.
# Audio samples per HuBERT frame = 48000 / 50 = 960.
AUDIO_SR = 48000
HUBERT_FPS = 50
SAMPLES_PER_FRAME = AUDIO_SR // HUBERT_FPS  # 960
FADE_MS = 5.0  # fade at each silence boundary — 5-10ms avoids audible clicks
MIN_REST_MS = 120.0  # only mute rests longer than this. Shorter MIDI=0 gaps
                     # are often labeling artifacts between connected notes; muting
                     # them creates unwanted "蹦字" (choppy) effect.

# Consonants exempt from voicing-gate mute.
# Unvoiced: no F0 in real audio, model voicing_logit always low.
# Voiced: F0 present but often noisy/transient, causes false-positive gate triggers
#   at consonant-vowel boundaries.
UNVOICED_PHONEMES = {
    # Stops
    "p", "pʰ", "t", "tʰ", "k", "kʰ", "ʔ",
    # Fricatives
    "f", "s", "ɕ", "ʂ", "ɸ", "h", "x", "ç", "θ",
    # Affricates
    "ts", "tsʰ", "tɕ", "tɕʰ", "ʈʂ", "ʈʂʰ", "tʃ",
}
VOICED_CONSONANTS = {
    # Voiced stops
    "b", "d", "g",
    # Voiced fricatives
    "v", "z", "ʑ", "ɣ", "ð",
    # Voiced affricates
    "dz", "dʑ", "dʒ", "ɖʐ",
    # Nasals
    "m", "n", "ŋ", "ɲ", "ɴ",
    # Liquids/glides
    "l", "r", "ɾ", "w", "j", "ɹ", "ɥ",
}
EXEMPT_PHONEMES = UNVOICED_PHONEMES | VOICED_CONSONANTS


def build_frame_midi(utt_id: str, dataset: str) -> np.ndarray:
    """Load pitch + duration, expand to per-frame MIDI. Force 0 where phoneme==SP.

    Score tools often carry previous-note MIDI into SP/rest phonemes instead of 0.
    We treat phoneme==SP as 'rest' regardless of MIDI value — this catches mute cases
    where MIDI inherited non-zero from adjacent notes.
    """
    import yaml
    proc_dir = PROCESSED / dataset
    pitch = torch.load(proc_dir / "pitch" / f"{utt_id}.pt", weights_only=False).numpy()
    duration = torch.load(proc_dir / "duration" / f"{utt_id}.pt", weights_only=False).numpy()
    phonemes = torch.load(proc_dir / "phonemes" / f"{utt_id}.pt", weights_only=False).numpy()
    with open(proc_dir / "vocab.yaml", encoding="utf-8") as f:
        vocab = yaml.safe_load(f)
    sp_id = vocab["phone2id"].get("SP", 3)
    is_sp = (phonemes == sp_id)
    # Force pitch = 0 at SP phoneme positions
    pitch_fixed = pitch.copy()
    pitch_fixed[is_sp] = 0
    return np.repeat(pitch_fixed, duration).astype(np.int32)


def build_frame_is_unvoiced_consonant(utt_id: str, dataset: str) -> np.ndarray:
    """Per-frame boolean: True iff phoneme is a consonant (unvoiced OR voiced) —
    exempt from voicing gate. Name kept for backward compat."""
    import yaml
    proc_dir = PROCESSED / dataset
    with open(proc_dir / "vocab.yaml", encoding="utf-8") as f:
        vocab = yaml.safe_load(f)
    phone2id = vocab["phone2id"]
    exempt_ids = {phone2id[p] for p in EXEMPT_PHONEMES if p in phone2id}
    phonemes = torch.load(proc_dir / "phonemes" / f"{utt_id}.pt", weights_only=False).numpy()
    duration = torch.load(proc_dir / "duration" / f"{utt_id}.pt", weights_only=False).numpy()
    frame_ph = np.repeat(phonemes, duration)
    return np.isin(frame_ph, list(exempt_ids))


def apply_voicing_gate(frame_midi: np.ndarray, voicing_sig: np.ndarray,
                        is_unvoiced_consonant: np.ndarray | None = None,
                        threshold: float = 0.3,
                        exempt_dilate_frames: int = 3) -> np.ndarray:
    """Force frame_midi to 0 where voicing_sig < threshold even if MIDI > 0.

    SKIPS frames that are unvoiced consonants OR within ±exempt_dilate_frames of one.
    Bias toward preserving audio (minimize Type 1 = wrongly muting real phonemes).
    """
    T = len(frame_midi)
    if len(voicing_sig) != T:
        idx = np.round(np.arange(T) * (len(voicing_sig) - 1) / max(T - 1, 1)).astype(np.int64)
        voicing_sig = voicing_sig[idx]
    patched = frame_midi.copy()
    gate_mask = voicing_sig < threshold
    if is_unvoiced_consonant is not None:
        # Dilate exempt region: ±N frames around each unvoiced consonant frame
        exempt = is_unvoiced_consonant.copy()
        if exempt_dilate_frames > 0:
            for shift in range(1, exempt_dilate_frames + 1):
                exempt[shift:] |= is_unvoiced_consonant[:-shift]
                exempt[:-shift] |= is_unvoiced_consonant[shift:]
        gate_mask = gate_mask & ~exempt
    patched[gate_mask] = 0
    return patched


NOTE_ONSET_FADE_MS = 0.0  # Run 8 listening verdict: don't mute 音头 (notes attacks).
                           # Fade-in at rest→voice + dip at note→note cause soft/delayed咬字
                           # and rhythm. Only voice→rest 音尾 fade-out kept.


def build_envelope(frame_midi: np.ndarray, audio_len: int, sr: int = AUDIO_SR,
                   fade_ms: float = FADE_MS, min_rest_ms: float = MIN_REST_MS,
                   note_onset_fade_ms: float = NOTE_ONSET_FADE_MS) -> np.ndarray:
    """Build amplitude envelope: 1.0 where frame_midi > 0, 0.0 where == 0.

    Only mute sustained rests longer than min_rest_ms. Short MIDI=0 gaps
    (labeling artifacts between connected notes) are left voiced.

    Smooth with linear ramp over fade_ms at every real transition.
    """
    T = len(frame_midi)
    voiced_frame = (frame_midi > 0).astype(np.int32)

    # Find runs of MIDI==0; only those longer than min_rest_frames are "real rests"
    min_rest_frames = int(round(min_rest_ms / 1000.0 * HUBERT_FPS))
    if min_rest_frames < 1:
        min_rest_frames = 1
    # Patch short MIDI=0 runs back to voiced=1
    i = 0
    while i < T:
        if voiced_frame[i] == 0:
            j = i
            while j < T and voiced_frame[j] == 0:
                j += 1
            run_len = j - i
            if run_len < min_rest_frames:
                voiced_frame[i:j] = 1  # treat as voiced (labeling gap)
            i = j
        else:
            i += 1
    voiced_samples = np.repeat(voiced_frame.astype(np.float32), SAMPLES_PER_FRAME)
    # May need to pad/truncate to audio_len
    if voiced_samples.size > audio_len:
        voiced_samples = voiced_samples[:audio_len]
    elif voiced_samples.size < audio_len:
        voiced_samples = np.concatenate([
            voiced_samples,
            np.full(audio_len - voiced_samples.size, voiced_samples[-1] if T else 0.0),
        ])

    fade_samples = int(fade_ms * sr / 1000.0)

    # Run 8 listening verdict: DON'T mute 音头 (rest→voice rising fade disabled).
    # Only 音尾 fade-out kept — rising edge left as sharp transition from 0 to 1.
    # Sharp rest→voice edge preserves咬字attack; soft rising fade makes vocals sound
    # tentative/delayed.
    if fade_samples > 1:
        diff = np.diff(np.concatenate([[voiced_samples[0]], voiced_samples]))
        falling = np.where(diff < -0.5)[0]
        for f in falling:
            start = max(f - fade_samples, 0)
            ramp = np.linspace(1.0, 0.0, f - start, dtype=np.float32)
            voiced_samples[start:f] = ramp

    # Note-onset fade: dip + ramp at every pitch change WITHIN voiced region
    # (kill attack artifacts at note→note boundaries where existing rise/fall logic doesn't apply).
    note_fade_samples = int(note_onset_fade_ms * sr / 1000.0)
    if note_fade_samples > 1:
        # Detect pitch transitions within voiced region
        # Use original frame_midi (not the short-rest-patched voiced_frame)
        for i in range(1, T):
            if frame_midi[i] > 0 and frame_midi[i - 1] > 0 and frame_midi[i] != frame_midi[i - 1]:
                # Pitch change at frame i — taper both sides
                sample_idx = i * SAMPLES_PER_FRAME
                # Fade-out: previous note's last note_fade_samples
                fout_start = max(sample_idx - note_fade_samples, 0)
                fout_end = min(sample_idx, audio_len)
                if fout_end > fout_start:
                    ramp = np.linspace(
                        voiced_samples[fout_start], 0.0, fout_end - fout_start, dtype=np.float32
                    )
                    voiced_samples[fout_start:fout_end] = np.minimum(voiced_samples[fout_start:fout_end], ramp)
                # Fade-in: new note's first note_fade_samples
                fin_start = sample_idx
                fin_end = min(sample_idx + note_fade_samples, audio_len)
                if fin_end > fin_start:
                    ramp = np.linspace(0.0, 1.0, fin_end - fin_start, dtype=np.float32)
                    voiced_samples[fin_start:fin_end] = np.minimum(voiced_samples[fin_start:fin_end], ramp)
    return voiced_samples


def apply_mute(in_wav: Path, out_wav: Path, utt_id: str, dataset: str,
               fade_ms: float = FADE_MS, min_rest_ms: float = MIN_REST_MS,
               note_onset_fade_ms: float = NOTE_ONSET_FADE_MS,
               voicing_dir: Path | None = None,
               voicing_threshold: float = 0.3,
               exempt_dilate_frames: int = 3) -> None:
    audio, sr = sf.read(str(in_wav))
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    frame_midi = build_frame_midi(utt_id, dataset)
    if voicing_dir is not None:
        voicing_pt = voicing_dir / f"{utt_id}.pt"
        if voicing_pt.exists():
            data = torch.load(voicing_pt, weights_only=False)
            unvoiced_cons = build_frame_is_unvoiced_consonant(utt_id, dataset)
            frame_midi = apply_voicing_gate(
                frame_midi, data["voicing_sig"].numpy(),
                is_unvoiced_consonant=unvoiced_cons,
                threshold=voicing_threshold,
                exempt_dilate_frames=exempt_dilate_frames,
            )
    envelope = build_envelope(frame_midi, len(audio), sr=sr,
                              fade_ms=fade_ms, min_rest_ms=min_rest_ms,
                              note_onset_fade_ms=note_onset_fade_ms)
    audio_muted = audio * envelope
    sf.write(str(out_wav), audio_muted, sr)


def apply_mute_to_dir(in_dir: Path, out_dir: Path, dataset: str,
                       utt_id_map: dict, fade_ms: float = FADE_MS,
                       min_rest_ms: float = MIN_REST_MS,
                       note_onset_fade_ms: float = NOTE_ONSET_FADE_MS,
                       voicing_dir: Path | None = None,
                       voicing_threshold: float = 0.3,
                       exempt_dilate_frames: int = 3) -> None:
    """Apply mute to all sampleN_*.wav in in_dir, write to out_dir.

    utt_id_map: {sample_index (1-based): utt_id} — must be known from eval_audio logs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for wav in sorted(in_dir.glob("sample*.wav")):
        # Parse sample N from filename
        name = wav.stem  # sample1_pred_full
        sample_n = int(name.split("_")[0].replace("sample", ""))
        if sample_n not in utt_id_map:
            continue
        utt_id = utt_id_map[sample_n]
        out_wav = out_dir / wav.name
        try:
            apply_mute(wav, out_wav, utt_id, dataset, fade_ms, min_rest_ms,
                       note_onset_fade_ms, voicing_dir, voicing_threshold,
                       exempt_dilate_frames)
            print(f"  muted: {wav.name}  (utt={utt_id})")
        except Exception as e:
            print(f"  SKIP {wav.name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory of sampleN_*.wav")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", required=True, help="Dataset name in data/processed/")
    ap.add_argument("--utt-ids", nargs="+", required=True,
                    help="Utt IDs in order sample1, sample2, sample3...")
    ap.add_argument("--min-rest-ms", type=float, default=MIN_REST_MS,
                    help="Min continuous MIDI=0 length to mute (ms). Shorter = labeling artifact.")
    ap.add_argument("--fade-ms", type=float, default=FADE_MS)
    ap.add_argument("--note-onset-fade-ms", type=float, default=NOTE_ONSET_FADE_MS,
                    help="Brief dip+ramp at note→note pitch transitions (attack artifact damping).")
    ap.add_argument("--voicing-dir", default=None,
                    help="Dir with {utt_id}.pt containing voicing_sig; enables voicing-gated mute.")
    ap.add_argument("--voicing-threshold", type=float, default=0.3)
    ap.add_argument("--exempt-dilate-frames", type=int, default=3)
    args = ap.parse_args()
    utt_id_map = {i + 1: utt for i, utt in enumerate(args.utt_ids)}
    voicing_dir = Path(args.voicing_dir) if args.voicing_dir else None
    apply_mute_to_dir(Path(args.in_dir), Path(args.out_dir), args.dataset,
                      utt_id_map, fade_ms=args.fade_ms, min_rest_ms=args.min_rest_ms,
                      note_onset_fade_ms=args.note_onset_fade_ms,
                      voicing_dir=voicing_dir, voicing_threshold=args.voicing_threshold,
                      exempt_dilate_frames=args.exempt_dilate_frames)


if __name__ == "__main__":
    main()
