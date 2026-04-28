"""
Score2HuBERT v2 — Kiritan + Itako singing database preprocessing.

Inputs:
    - Kiritan: kiritan/kiritan_singing/kiritan_singing/{wav,mono_label}/NN.{wav,lab}
              lab format: floating-point seconds
                  "0.0000000 18.6263777 pau"
    - Itako:   kiritan/itako_singing/{wav,mono_label}/itakoNN.{wav,lab}
              lab format: integer 100ns units
                  "923400 12000000 sil"

Each song is a long continuous recording (64s-272s). We segment into training-sized
chunks (~1.5-12s) at pauses/silences/breaths longer than 0.3s.

Output:
    hubert/{id}.pt       torch.float32 [T_frame, 768]
    f0/{id}.pt           torch.float32 [T_frame]
    phonemes/{id}.pt     torch.long    [T_phone]
    pitch/{id}.pt        torch.long    [T_phone]   (MIDI estimated from F0)
    duration/{id}.pt     torch.long    [T_phone]   (frames, sum == T_frame)

IDs:  kiritan_{song:02d}_{seg:03d}  or  itako_{song:02d}_{seg:03d}
Split: 90/10 train/test (hash of id)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "RVC" / "RVC20240604Nvidia"))

from src.preprocessing.extract_hubert import HuBERTExtractor  # noqa: E402
from src.data.phoneme_v1 import PhonemeVocab, OPENJTALK_TO_IPA  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess_kiritan_itako")


SAMPLE_RATE = 16000
FRAME_RATE = 50
HOP = SAMPLE_RATE // FRAME_RATE  # 320

PAUSE_TOKENS = {"pau", "sil", "sp", "br"}
LONG_PAUSE_SEC = 0.3           # pause longer than this splits segments
SEG_MIN_SEC = 1.5
SEG_MAX_SEC = 12.0


# ─── Lab file parsing ──────────────────────────────────────────────────────


def parse_lab_file(lab_path: Path, fmt: str) -> List[Tuple[float, float, str]]:
    """Return list of (start_sec, end_sec, phoneme).

    fmt: 'sec' (float seconds) or 'htk' (integer 100ns units).
    """
    intervals: List[Tuple[float, float, str]] = []
    with open(lab_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                if fmt == "sec":
                    s = float(parts[0])
                    e = float(parts[1])
                else:  # htk 100ns units
                    s = int(parts[0]) * 1e-7
                    e = int(parts[1]) * 1e-7
            except ValueError:
                continue
            ph = parts[2]
            intervals.append((s, e, ph))
    return intervals


# ─── Segmentation ──────────────────────────────────────────────────────────


def segment_by_pauses(
    intervals: List[Tuple[float, float, str]],
    long_pause: float = LONG_PAUSE_SEC,
    max_sec: float = SEG_MAX_SEC,
) -> List[List[Tuple[float, float, str]]]:
    """Split an interval list into chunks at long pauses.

    Each chunk starts with the first non-pause phoneme and ends just before
    the next "long pause" (>= long_pause). Short pauses/breaths inside a chunk
    are kept (they provide useful SP/br supervision).

    Also enforces max_sec: if a chunk grows beyond that, force-split at the
    next short pause (>0.05s) or, failing that, at a phoneme boundary.
    """
    segments: List[List[Tuple[float, float, str]]] = []
    cur: List[Tuple[float, float, str]] = []

    def chunk_dur(chunk):
        if not chunk:
            return 0.0
        return chunk[-1][1] - chunk[0][0]

    for (s, e, ph) in intervals:
        is_pause = ph in PAUSE_TOKENS
        pau_dur = (e - s) if is_pause else 0.0

        if is_pause and pau_dur >= long_pause:
            # Split here
            if cur:
                # Trim any trailing pauses in chunk
                while cur and cur[-1][2] in PAUSE_TOKENS:
                    cur.pop()
                if cur:
                    segments.append(cur)
            cur = []
            continue

        # Add to current chunk
        # Skip leading pauses
        if not cur and is_pause:
            continue
        cur.append((s, e, ph))

        # Enforce max_sec
        if chunk_dur(cur) > max_sec:
            # Try to split at the last short-pause boundary within cur
            cut = None
            for i in range(len(cur) - 1, 0, -1):
                if cur[i][2] in PAUSE_TOKENS:
                    cut = i
                    break
            if cut is None:
                cut = len(cur) - 1  # split at last phone boundary
            head, tail = cur[:cut], cur[cut:]
            while head and head[-1][2] in PAUSE_TOKENS:
                head.pop()
            if head:
                segments.append(head)
            # Retain tail as the new current chunk; strip leading pauses
            while tail and tail[0][2] in PAUSE_TOKENS:
                tail.pop(0)
            cur = tail

    if cur:
        while cur and cur[-1][2] in PAUSE_TOKENS:
            cur.pop()
        if cur:
            segments.append(cur)

    return segments


# ─── Phoneme conversion ────────────────────────────────────────────────────


def jp_phoneme_to_ipa_list(ph: str) -> List[str]:
    """Map one OpenJTalk-ish phoneme to IPA segments (may be multi-segment)."""
    # Normalize pauses
    if ph in ("pau", "sil", "sp"):
        return ["SP"]
    if ph == "br":
        return ["br"]

    # Handle palatalized ones: "ky", "gy", ...
    if ph in OPENJTALK_TO_IPA:
        return list(OPENJTALK_TO_IPA[ph])
    # Unknown -> SP fallback
    return ["SP"]


# ─── Audio + features ──────────────────────────────────────────────────────


@torch.no_grad()
def hubert_from_array(model, audio16k: np.ndarray, device: str, use_fp16: bool) -> torch.Tensor:
    audio = torch.from_numpy(audio16k).float().unsqueeze(0).to(device)
    if use_fp16:
        audio = audio.half()
    audio = F.pad(audio, (40, 40), mode="reflect")
    feats = model.extract_features(source=audio, padding_mask=None, output_layer=12)
    feat = feats[0].squeeze(0).float().cpu()
    return feat


def rmvpe_to_frames(rmvpe, audio16k: np.ndarray, n_target: int) -> torch.Tensor:
    """Run RMVPE (100fps) then downsample to n_target @ 50fps."""
    f0_100 = rmvpe.infer_from_audio(audio16k.astype(np.float32), thred=0.03)
    f0_100 = np.asarray(f0_100, dtype=np.float32)
    if f0_100.size == 0 or n_target <= 0:
        return torch.zeros(n_target, dtype=torch.float32)
    src_len = f0_100.shape[0]
    idxs = np.clip((np.arange(n_target) * 2), 0, src_len - 1)
    return torch.from_numpy(f0_100[idxs]).float()


def f0_hz_to_midi(f0_hz: float) -> float:
    if f0_hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(f0_hz / 440.0)


# ─── Segment processing ────────────────────────────────────────────────────


def split_jp_to_ipa_with_duration(
    jp_phonemes: List[str],
    jp_dur_frames: List[int],
) -> Tuple[List[str], List[int]]:
    """Convert a sequence of OpenJTalk phones (with per-phone frame durations)
    to IPA phones, splitting multi-segment mappings proportionally.

    A multi-segment mapping (e.g. `ky` -> [k, j]) divides the original duration
    equally across segments, then the remainder is pushed to the last segment.
    """
    out_ph: List[str] = []
    out_dur: List[int] = []
    for ph, d in zip(jp_phonemes, jp_dur_frames):
        segs = jp_phoneme_to_ipa_list(ph)
        n = len(segs)
        if n == 1:
            out_ph.append(segs[0])
            out_dur.append(d)
            continue
        per = d // n
        rem = d - per * n
        for i, s in enumerate(segs):
            out_ph.append(s)
            out_dur.append(per + (rem if i == n - 1 else 0))
    return out_ph, out_dur


def split_for(utt_id: str) -> str:
    h = int(hashlib.md5(utt_id.encode("utf-8")).hexdigest(), 16)
    return "train" if (h % 10) < 9 else "test"


def build_segment_entry(
    audio16k: np.ndarray,
    seg: List[Tuple[float, float, str]],
    hubert_model,
    rmvpe,
    vocab: PhonemeVocab,
    device: str,
    use_fp16: bool,
    utt_id: str,
    out_dirs: Dict[str, Path],
    dataset: str,
    singer: str,
) -> Optional[Dict]:
    """Process one segment -> save .pt files, return metadata dict (or None)."""
    seg_start = seg[0][0]
    seg_end = seg[-1][1]
    dur_sec = seg_end - seg_start
    if dur_sec < SEG_MIN_SEC or dur_sec > SEG_MAX_SEC:
        return None

    s_samp = int(round(seg_start * SAMPLE_RATE))
    e_samp = int(round(seg_end * SAMPLE_RATE))
    s_samp = max(0, min(s_samp, len(audio16k)))
    e_samp = max(0, min(e_samp, len(audio16k)))
    if e_samp - s_samp < int(SEG_MIN_SEC * SAMPLE_RATE):
        return None
    audio_seg = audio16k[s_samp:e_samp].astype(np.float32)

    # HuBERT first (authoritative T_frame)
    hubert = hubert_from_array(hubert_model, audio_seg, device=device, use_fp16=use_fp16)
    T = hubert.shape[0]
    if T < 5:
        return None
    f0 = rmvpe_to_frames(rmvpe, audio_seg, T)

    # Per-phoneme frame durations in the segment's local time
    jp_phonemes: List[str] = []
    jp_dur_frames: List[int] = []
    for (s, e, ph) in seg:
        local_s = max(0.0, s - seg_start)
        local_e = max(local_s, e - seg_start)
        start_f = int(round(local_s * FRAME_RATE))
        end_f = int(round(local_e * FRAME_RATE))
        d = max(1, end_f - start_f)
        jp_phonemes.append(ph)
        jp_dur_frames.append(d)

    # Rescale so sum(jp_dur) == T
    cur_sum = sum(jp_dur_frames)
    if cur_sum != T:
        scale = T / cur_sum
        rescaled = [max(1, int(round(d * scale))) for d in jp_dur_frames]
        diff = T - sum(rescaled)
        if diff != 0:
            j = int(np.argmax(rescaled))
            rescaled[j] += diff
            if rescaled[j] < 1:
                rescaled[j] = 1
                diff2 = T - sum(rescaled)
                if diff2 != 0:
                    rescaled[int(np.argmax(rescaled))] += diff2
        jp_dur_frames = rescaled

    # Convert JP -> IPA with duration split
    ipa_phs, ipa_durs = split_jp_to_ipa_with_duration(jp_phonemes, jp_dur_frames)

    # Fix any remaining duration drift
    diff = T - sum(ipa_durs)
    if diff != 0:
        j = int(np.argmax(ipa_durs))
        ipa_durs[j] += diff
        if ipa_durs[j] < 1:
            ipa_durs[j] = 1
            j2 = int(np.argmax(ipa_durs))
            ipa_durs[j2] += T - sum(ipa_durs)

    assert sum(ipa_durs) == T, f"duration sum {sum(ipa_durs)} != T {T}"
    assert len(ipa_phs) == len(ipa_durs), "ipa_phs / ipa_durs length mismatch"
    assert all(d >= 1 for d in ipa_durs), f"zero-length phoneme: {ipa_durs}"

    # Encode phonemes -> IDs
    ph_ids = vocab.encode(ipa_phs)

    # MIDI per IPA phoneme (from F0 median of that phoneme's frames)
    f0_np = f0.numpy()
    midi_list: List[int] = []
    last_midi = 0
    pos = 0
    for d in ipa_durs:
        slice_ = f0_np[pos : pos + d]
        voiced = slice_[slice_ > 0]
        if len(voiced) > 0:
            med_hz = float(np.median(voiced))
            midi = int(round(f0_hz_to_midi(med_hz)))
            if midi < 24 or midi > 108:
                midi = last_midi
            else:
                last_midi = midi
        else:
            midi = last_midi
        midi_list.append(midi)
        pos += d

    # Save tensors
    torch.save(hubert, out_dirs["hubert"] / f"{utt_id}.pt")
    torch.save(f0, out_dirs["f0"] / f"{utt_id}.pt")
    torch.save(torch.tensor(ph_ids, dtype=torch.long), out_dirs["phonemes"] / f"{utt_id}.pt")
    torch.save(torch.tensor(midi_list, dtype=torch.long), out_dirs["pitch"] / f"{utt_id}.pt")
    torch.save(torch.tensor(ipa_durs, dtype=torch.long), out_dirs["duration"] / f"{utt_id}.pt")

    return {
        "id": utt_id,
        "n_phones": len(ipa_phs),
        "n_frames": T,
        "language": "ja",
        "lang_id": vocab.get_language_id("ja"),
        "singing_ratio": 1.0,
        "dataset": dataset,
        "singer": singer,
        "split": split_for(utt_id),
    }


def process_song(
    wav_path: Path,
    lab_path: Path,
    lab_fmt: str,
    song_id: str,
    hubert_model,
    rmvpe,
    vocab: PhonemeVocab,
    device: str,
    use_fp16: bool,
    out_dirs: Dict[str, Path],
    dataset: str,
    singer: str,
    id_prefix: str,
) -> List[Dict]:
    # Load + resample
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    else:
        audio = audio.astype(np.float32)

    intervals = parse_lab_file(lab_path, fmt=lab_fmt)
    if not intervals:
        return []

    segments = segment_by_pauses(intervals)

    meta_out: List[Dict] = []
    for seg_idx, seg in enumerate(segments):
        utt_id = f"{id_prefix}_{song_id}_{seg_idx:03d}"
        try:
            entry = build_segment_entry(
                audio16k=audio,
                seg=seg,
                hubert_model=hubert_model,
                rmvpe=rmvpe,
                vocab=vocab,
                device=device,
                use_fp16=use_fp16,
                utt_id=utt_id,
                out_dirs=out_dirs,
                dataset=dataset,
                singer=singer,
            )
        except Exception as e:
            logger.warning(f"seg fail {utt_id}: {e}")
            entry = None
        if entry is not None:
            meta_out.append(entry)
    return meta_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kiritan-root", default=str(REPO_ROOT / "kiritan" / "kiritan_singing" / "kiritan_singing"))
    parser.add_argument("--itako-root", default=str(REPO_ROOT / "kiritan" / "itako_singing"))
    parser.add_argument("--out-root", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--hubert-model", default=str(REPO_ROOT / "RVC/RVC20240604Nvidia/assets/hubert/hubert_base.pt"))
    parser.add_argument("--rmvpe-model", default=str(REPO_ROOT / "RVC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt"))
    parser.add_argument("--vocab", default=str(REPO_ROOT / "data/processed/ace_opencpop/vocab.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--skip", choices=["none", "kiritan", "itako"], default="none")
    args = parser.parse_args()

    vocab = PhonemeVocab(args.vocab)

    # HuBERT / RMVPE
    logger.info("Loading HuBERT...")
    hubert_ext = HuBERTExtractor(args.hubert_model, layer=12, device=args.device, use_fp16=args.fp16)
    hubert_model = hubert_ext.model

    logger.info("Loading RMVPE...")
    sys.path.insert(0, str(REPO_ROOT / "RVC" / "RVC20240604Nvidia"))
    sys.path.insert(0, str(REPO_ROOT / "RVC" / "RVC20240604Nvidia" / "infer"))
    from infer.lib.rmvpe import RMVPE
    rmvpe = RMVPE(args.rmvpe_model, is_half=args.fp16, device=args.device)

    def run_dataset(
        wav_dir: Path,
        lab_dir: Path,
        lab_fmt: str,
        out_dir: Path,
        dataset: str,
        singer: str,
        id_prefix: str,
        song_id_from_name: callable,
    ) -> List[Dict]:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs = {n: (out_dir / n) for n in ("hubert", "f0", "phonemes", "pitch", "duration")}
        for d in out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.vocab, out_dir / "vocab.yaml")

        wav_files = sorted(wav_dir.glob("*.wav"))
        logger.info(f"[{dataset}] {len(wav_files)} songs in {wav_dir}")

        all_meta: List[Dict] = []
        pbar = tqdm(wav_files, desc=dataset, unit="song")
        for wav_path in pbar:
            song_id = song_id_from_name(wav_path.stem)
            lab_path = lab_dir / f"{wav_path.stem}.lab"
            if not lab_path.exists():
                logger.warning(f"missing lab: {lab_path}")
                continue
            meta = process_song(
                wav_path=wav_path,
                lab_path=lab_path,
                lab_fmt=lab_fmt,
                song_id=song_id,
                hubert_model=hubert_model,
                rmvpe=rmvpe,
                vocab=vocab,
                device=args.device,
                use_fp16=args.fp16,
                out_dirs=out_dirs,
                dataset=dataset,
                singer=singer,
                id_prefix=id_prefix,
            )
            all_meta.extend(meta)
            pbar.set_postfix(segs=len(all_meta))

        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(all_meta, f, ensure_ascii=False, indent=0)

        total_frames = sum(e["n_frames"] for e in all_meta)
        logger.info(
            f"[{dataset}] {len(all_meta)} segs, {total_frames} frames, "
            f"{total_frames/FRAME_RATE/3600:.2f}h"
        )
        return all_meta

    # Kiritan
    if args.skip != "kiritan":
        kiritan_root = Path(args.kiritan_root)
        run_dataset(
            wav_dir=kiritan_root / "wav",
            lab_dir=kiritan_root / "mono_label",
            lab_fmt="sec",
            out_dir=Path(args.out_root) / "kiritan",
            dataset="kiritan",
            singer="kiritan",
            id_prefix="kiritan",
            song_id_from_name=lambda s: s,  # "01" -> "01"
        )

    # Itako
    if args.skip != "itako":
        itako_root = Path(args.itako_root)
        run_dataset(
            wav_dir=itako_root / "wav",
            lab_dir=itako_root / "mono_label",
            lab_fmt="htk",
            out_dir=Path(args.out_root) / "itako",
            dataset="itako",
            singer="itako",
            id_prefix="itako",
            song_id_from_name=lambda s: s.replace("itako", ""),  # "itako01" -> "01"
        )


if __name__ == "__main__":
    main()
