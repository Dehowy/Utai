"""WORLD F0 snap to MIDI — kill 跑调 by replacing audio F0 with score-derived F0.

Pipeline:
  1. pyworld decompose audio → (f0, sp, ap)
  2. Derive target F0 from per-frame MIDI (midi_to_hz)
  3. Blend: corrected_f0 = f0 - alpha × (lowpass(f0) - target_f0)
     - alpha=1.0 hard snap (long-term drift → 0, vibrato preserved if vibrato is high-freq)
     - alpha=0.7 soft anchor (natural)
     - alpha=0.0 no correction
  4. pyworld synthesize (sp + ap unchanged) → new audio

Relies on pyworld's frame period (5ms by default). Audio stays at whatever SR input has.

Usage:
    python scripts/apply_world_snap.py --in-dir X --out-dir Y --dataset ace_opencpop \\
      --utt-ids 'ace_acesinger_27#2001000033' ... --alpha 1.0
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pyworld as pw
import soundfile as sf
import torch

V2 = Path(__file__).resolve().parents[1]
PROCESSED = V2 / "data" / "processed"
HUBERT_FPS = 50
WORLD_FRAME_PERIOD = 5.0  # ms


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """MIDI (int) → Hz, 0 → 0."""
    hz = np.zeros_like(midi, dtype=np.float64)
    mask = midi > 0
    hz[mask] = 440.0 * (2.0 ** ((midi[mask].astype(np.float64) - 69.0) / 12.0))
    return hz


def build_frame_midi(utt_id: str, dataset: str) -> np.ndarray:
    """Load pitch + duration, expand to per-HuBERT-frame MIDI (50fps)."""
    proc_dir = PROCESSED / dataset
    pitch = torch.load(proc_dir / "pitch" / f"{utt_id}.pt", weights_only=False).numpy()
    duration = torch.load(proc_dir / "duration" / f"{utt_id}.pt", weights_only=False).numpy()
    return np.repeat(pitch, duration).astype(np.int32)


def resample_midi_to_world(frame_midi_50fps: np.ndarray, n_world_frames: int) -> np.ndarray:
    """Resample per-HuBERT-frame MIDI (50fps = 20ms) to WORLD frames (200fps = 5ms)."""
    T50 = len(frame_midi_50fps)
    if T50 == 0 or n_world_frames == 0:
        return np.zeros(n_world_frames, dtype=np.int32)
    # For each WORLD frame idx, map to nearest HuBERT frame
    idx = np.round(np.arange(n_world_frames) * (T50 - 1) / max(n_world_frames - 1, 1)).astype(np.int64)
    idx = np.clip(idx, 0, T50 - 1)
    return frame_midi_50fps[idx]


def lowpass(x: np.ndarray, cutoff_hz: float = 2.0, fs: float = 1000.0 / WORLD_FRAME_PERIOD) -> np.ndarray:
    """Simple Butterworth-ish via FFT-based lowpass, preserving the slow trend of F0."""
    if len(x) < 8:
        return x.copy()
    from scipy.signal import butter, filtfilt
    b, a = butter(N=2, Wn=cutoff_hz / (fs / 2), btype="lowpass")
    return filtfilt(b, a, x)


def correct_f0(f0: np.ndarray, target_f0: np.ndarray, alpha: float = 1.0,
               transition_cutoff_hz: float = 3.0, debug: bool = False) -> np.ndarray:
    """Blend F0 toward target with smooth (not discrete) note transitions.

    Strategy: decompose each signal into slow trend + fast residual (slides/vibrato).
      slow = lowpass at transition_cutoff_hz (typical 3Hz = ~300ms smoothing)
      residual = signal - slow
    Snap the SLOW component toward target (also smoothed), keep the fast residual
    from audio. Result: long-term pitch matches score but has natural
    fast movement (vibrato, portamento).

    alpha=1.0: full snap of slow component to target_smooth
    alpha=0.7: soft — pulls 70% toward target
    alpha=0.0: no correction

    Setting transition_cutoff_hz high (8-10Hz) → more score-driven, closer to hard snap.
    Setting lower (1-2Hz) → more natural but less drift-correction.
    """
    out = f0.copy()
    voiced_f0 = (f0 > 30.0)
    voiced_target = (target_f0 > 30.0)
    voiced = voiced_f0 & voiced_target

    if not voiced.any():
        if debug:
            print(f"  [correct_f0] no overlapping voiced frames")
        return out

    # For smoothing we need continuous data. Fill unvoiced with nearest voiced.
    def fill_nearest(sig: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill unvoiced positions with nearest voiced value."""
        if not mask.any():
            return sig
        idx_voiced = np.where(mask)[0]
        filled = sig.copy()
        for i in range(len(sig)):
            if not mask[i]:
                nearest = idx_voiced[np.argmin(np.abs(idx_voiced - i))]
                filled[i] = sig[nearest]
        return filled

    f0_filled = fill_nearest(f0, voiced_f0)
    target_filled = fill_nearest(target_f0, voiced_target)

    # Smooth both to same cutoff, giving natural transitions
    f0_slow = lowpass(f0_filled, cutoff_hz=transition_cutoff_hz)
    target_slow = lowpass(target_filled, cutoff_hz=transition_cutoff_hz)
    f0_fast = f0_filled - f0_slow  # slides + vibrato + noise

    # Corrected slow trend: blend f0_slow toward target_slow
    if alpha >= 0.999:
        corrected_slow = target_slow
    else:
        corrected_slow = f0_slow - alpha * (f0_slow - target_slow)

    corrected = corrected_slow + f0_fast

    if debug:
        v_idx = np.where(voiced)[0][:10]
        print(f"  [correct_f0] {len(v_idx)} shown. cutoff={transition_cutoff_hz}Hz, alpha={alpha}")
        for i in v_idx:
            print(f"    frame {i}: f0={f0[i]:.1f} slow={f0_slow[i]:.1f} target={target_f0[i]:.1f} target_slow={target_slow[i]:.1f} → corr={corrected[i]:.1f}")

    # Only apply where both voiced (per-frame decision)
    out[voiced] = corrected[voiced]
    return out


def process_one(in_wav: Path, out_wav: Path, utt_id: str, dataset: str,
                alpha: float = 1.0, transition_cutoff_hz: float = 3.0,
                debug: bool = False) -> None:
    audio, sr = sf.read(str(in_wav))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio64 = audio.astype(np.float64)

    # WORLD decompose
    f0, t = pw.dio(audio64, sr, frame_period=WORLD_FRAME_PERIOD)
    f0 = pw.stonemask(audio64, f0, t, sr)
    sp = pw.cheaptrick(audio64, f0, t, sr)
    ap = pw.d4c(audio64, f0, t, sr)

    # Target F0 from MIDI
    frame_midi_50 = build_frame_midi(utt_id, dataset)
    midi_world = resample_midi_to_world(frame_midi_50, len(f0))
    target_f0 = midi_to_hz(midi_world)

    if debug:
        n_world = len(f0)
        n_hubert = len(frame_midi_50)
        print(f"  [debug] audio len {len(audio)} samples @ {sr}Hz = {len(audio)/sr:.2f}s")
        print(f"  [debug] world n_frames={n_world}  hubert n_frames={n_hubert}")
        print(f"  [debug] world voiced frames: {(f0 > 30).sum()}")
        print(f"  [debug] target voiced frames: {(target_f0 > 30).sum()}")
        # Show first MIDI transitions
        changes = np.where(np.diff(frame_midi_50) != 0)[0]
        print(f"  [debug] first 3 MIDI changes at hubert frames: {changes[:3].tolist()}")
        if len(changes):
            i = changes[0] + 1
            midi_val = frame_midi_50[i]
            expected_hz = 440.0 * (2.0 ** ((midi_val - 69.0) / 12.0))
            print(f"  [debug] hubert frame {i}: midi={midi_val}  expected_hz={expected_hz:.1f}")

    # Snap F0
    f0_corrected = correct_f0(f0, target_f0, alpha=alpha,
                               transition_cutoff_hz=transition_cutoff_hz, debug=debug)

    # Synthesize
    audio_new = pw.synthesize(f0_corrected, sp, ap, sr, frame_period=WORLD_FRAME_PERIOD)
    # Match original length
    if len(audio_new) > len(audio):
        audio_new = audio_new[:len(audio)]
    elif len(audio_new) < len(audio):
        audio_new = np.pad(audio_new, (0, len(audio) - len(audio_new)))
    sf.write(str(out_wav), audio_new.astype(np.float32), sr)


def process_dir(in_dir: Path, out_dir: Path, dataset: str,
                utt_id_map: dict, alpha: float = 1.0,
                transition_cutoff_hz: float = 3.0, debug: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for wav in sorted(in_dir.glob("sample*.wav")):
        name = wav.stem
        sample_n = int(name.split("_")[0].replace("sample", ""))
        if sample_n not in utt_id_map:
            continue
        utt_id = utt_id_map[sample_n]
        out_wav = out_dir / wav.name
        try:
            if debug:
                print(f"=== {wav.name} ===")
            process_one(wav, out_wav, utt_id, dataset, alpha=alpha,
                        transition_cutoff_hz=transition_cutoff_hz, debug=debug)
            print(f"  snapped: {wav.name}  (utt={utt_id}, alpha={alpha}, cutoff={transition_cutoff_hz}Hz)")
        except Exception as e:
            print(f"  SKIP {wav.name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--utt-ids", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Correction strength on slow trend. 1.0=full snap, 0.7=soft.")
    ap.add_argument("--cutoff-hz", type=float, default=3.0,
                    help="Transition smoothness. 3Hz=300ms transitions (natural), "
                         "8Hz=125ms (closer to hard snap), 1Hz=1s (very natural but weak correction).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    utt_id_map = {i + 1: utt for i, utt in enumerate(args.utt_ids)}
    process_dir(Path(args.in_dir), Path(args.out_dir), args.dataset, utt_id_map,
                alpha=args.alpha, transition_cutoff_hz=args.cutoff_hz, debug=args.debug)


if __name__ == "__main__":
    main()
