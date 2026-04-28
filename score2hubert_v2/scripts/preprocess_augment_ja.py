"""Augment JA datasets via pyworld: pitch shift ±2 semitones + time stretch ×0.95/1.05.

For each existing JA sample, generates 4 augmented variants with new utt_ids:
    <orig_id>_aug_p+2   — pitch +2 semitones
    <orig_id>_aug_p-2   — pitch -2 semitones
    <orig_id>_aug_t1.05 — time stretch ×1.05 (slower)
    <orig_id>_aug_t0.95 — time stretch ×0.95 (faster)

Each variant saves fresh hubert/f0/phonemes/pitch/duration tensors.
Appends new entries to metadata.json (keep_split consistent with orig entry).

After UTAU drop (Run 8), JA training set dropped from 22K → 3.2K samples.
This script lifts it back to ~16K via WORLD-decomposition augment.

Rationale for WORLD:
  - Phase vocoder (librosa) introduces chipmunk at pitch shift
  - WORLD decomposes f0/ap/sp independently → modify one without affecting others
  - HuBERT extracted from clean augmented audio stays near natural distribution

Usage:
    py -3.10 scripts/preprocess_augment_ja.py --dataset kiritan --limit 3 --dry-run
    py -3.10 scripts/preprocess_augment_ja.py --dataset all
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "RVC" / "RVC20240604Nvidia"))

from src.data.phoneme_v1 import PhonemeVocab  # noqa: E402
from src.preprocessing.extract_hubert import HuBERTExtractor  # noqa: E402

FRAME_RATE = 50
SAMPLE_RATE = 16000
HOP = SAMPLE_RATE // FRAME_RATE  # 320
PAUSE_ORIG = {"pau", "sil", "br", "cl", "sp"}
LONG_PAUSE = 0.3
SEG_MIN = 1.5
SEG_MAX = 12.0
WORLD_FRAME_PERIOD = 5.0  # ms

# Dataset configs (song_id inverse: utt_id → song_id + seg_idx)
CONFIGS = {
    "kiritan": {
        "fmt": "sec",
        "wav_dir": REPO_ROOT / "kiritan/kiritan_singing/kiritan_singing/wav",
        "lab_dir": REPO_ROOT / "kiritan/kiritan_singing/kiritan_singing/mono_label",
        "id_prefix": "kiritan",
        "wav_stem_from_song_id": lambda sid: sid,
    },
    "itako": {
        "fmt": "htk",
        "wav_dir": REPO_ROOT / "kiritan/itako_singing/wav",
        "lab_dir": REPO_ROOT / "kiritan/itako_singing/mono_label",
        "id_prefix": "itako",
        "wav_stem_from_song_id": lambda sid: f"itako{sid}",
    },
    "oniku": {
        "fmt": "htk",
        "wav_dir": REPO_ROOT / "ONIKU_KURUMI_UTAGOE_DB",  # nested: <song>/<song>.wav
        "lab_dir": REPO_ROOT / "ONIKU_KURUMI_UTAGOE_DB",
        "id_prefix": "oniku",
        "wav_stem_from_song_id": lambda sid: sid,
        "nested": True,  # <song>/<song>.{wav,lab}
    },
    "ofuton": {
        "fmt": "htk",
        "wav_dir": REPO_ROOT / "OFUTON_P_UTAGOE_DB/OFUTON_P_UTAGOE_DB",
        "lab_dir": REPO_ROOT / "OFUTON_P_UTAGOE_DB/OFUTON_P_UTAGOE_DB",
        "id_prefix": "ofuton",
        "wav_stem_from_song_id": lambda sid: sid,
        "nested": True,
    },
    "natsume": {
        "fmt": "sec",
        "wav_dir": REPO_ROOT / "Natsume_Singing_DB_0713/wav",
        "lab_dir": REPO_ROOT / "Natsume_Singing_DB_0713/mono_label",
        "id_prefix": "natsume",
        "wav_stem_from_song_id": lambda sid: sid,
    },
}


def parse_lab(path: Path, fmt: str):
    out = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 3:
                continue
            try:
                if fmt == "htk":
                    s, e = int(p[0]) * 1e-7, int(p[1]) * 1e-7
                else:
                    s, e = float(p[0]), float(p[1])
                out.append((s, e, p[2]))
            except ValueError:
                continue
    return out


def segment_by_pauses(intervals, long_pause=LONG_PAUSE, max_sec=SEG_MAX):
    segments = []
    cur = []

    def chunk_dur(c):
        return 0.0 if not c else c[-1][1] - c[0][0]

    for (s, e, ph) in intervals:
        is_pause = ph in PAUSE_ORIG
        pau_dur = (e - s) if is_pause else 0.0
        if is_pause and pau_dur >= long_pause:
            if cur:
                while cur and cur[-1][2] in PAUSE_ORIG:
                    cur.pop()
                if cur:
                    segments.append(cur)
            cur = []
            continue
        if not cur and is_pause:
            continue
        cur.append((s, e, ph))
        if chunk_dur(cur) > max_sec:
            cut = None
            for i in range(len(cur) - 1, 0, -1):
                if cur[i][2] in PAUSE_ORIG:
                    cut = i
                    break
            if cut is None:
                cut = len(cur) - 1
            head, tail = cur[:cut], cur[cut:]
            while head and head[-1][2] in PAUSE_ORIG:
                head.pop()
            if head:
                segments.append(head)
            while tail and tail[0][2] in PAUSE_ORIG:
                tail.pop(0)
            cur = tail
    if cur:
        while cur and cur[-1][2] in PAUSE_ORIG:
            cur.pop()
        if cur:
            segments.append(cur)
    return segments


def f0_hz_to_midi(hz: float) -> int:
    if hz <= 0:
        return 0
    return int(round(69.0 + 12.0 * np.log2(hz / 440.0)))


@torch.no_grad()
def hubert_extract(model, audio: np.ndarray, device: str, use_fp16: bool) -> torch.Tensor:
    a = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    if use_fp16:
        a = a.half()
    a = F.pad(a, (40, 40), mode="reflect")
    feats = model.extract_features(source=a, padding_mask=None, output_layer=12)
    return feats[0].squeeze(0).float().cpu()


def rmvpe_extract(rmvpe, audio: np.ndarray, n_target: int) -> torch.Tensor:
    f0_100 = rmvpe.infer_from_audio(audio.astype(np.float32), thred=0.03)
    f0_100 = np.asarray(f0_100, dtype=np.float32)
    if f0_100.size == 0 or n_target <= 0:
        return torch.zeros(n_target, dtype=torch.float32)
    src_len = f0_100.shape[0]
    idxs = np.clip((np.arange(n_target) * 2), 0, src_len - 1)
    return torch.from_numpy(f0_100[idxs]).float()


def world_decompose(audio: np.ndarray, sr: int):
    """Cache WORLD decomposition for a segment."""
    audio64 = audio.astype(np.float64)
    f0, t = pw.dio(audio64, sr, frame_period=WORLD_FRAME_PERIOD)
    f0 = pw.stonemask(audio64, f0, t, sr)
    sp = pw.cheaptrick(audio64, f0, t, sr)
    ap = pw.d4c(audio64, f0, t, sr)
    return f0, sp, ap


def world_pitch_shift(f0, sp, ap, sr, semitones):
    f0_new = f0 * (2 ** (semitones / 12.0))
    y = pw.synthesize(f0_new, sp, ap, sr, frame_period=WORLD_FRAME_PERIOD)
    return y.astype(np.float32)


def world_time_stretch(f0, sp, ap, sr, ratio):
    """Stretch by ratio (1.05 = 5% slower). Output length = orig × ratio."""
    n_orig = len(f0)
    n_new = int(round(n_orig * ratio))
    if n_new < 2:
        n_new = 2
    idx = np.linspace(0, n_orig - 1, n_new)
    idx_lo = np.floor(idx).astype(np.int64).clip(0, n_orig - 1)
    idx_hi = np.minimum(idx_lo + 1, n_orig - 1)
    w = (idx - idx_lo).astype(np.float64)

    f0_new = (1 - w) * f0[idx_lo] + w * f0[idx_hi]
    sp_new = (1 - w)[:, None] * sp[idx_lo] + w[:, None] * sp[idx_hi]
    ap_new = (1 - w)[:, None] * ap[idx_lo] + w[:, None] * ap[idx_hi]
    y = pw.synthesize(f0_new, sp_new, ap_new, sr, frame_period=WORLD_FRAME_PERIOD)
    return y.astype(np.float32)


def rescale_duration(orig_dur: torch.Tensor, new_T: int) -> torch.Tensor:
    """Rescale duration tensor to sum to new_T, preserving relative proportions."""
    orig_dur = orig_dur.clone().long()
    cur = int(orig_dur.sum().item())
    if cur == new_T:
        return orig_dur
    if cur == 0:
        return orig_dur
    scale = new_T / cur
    rescaled = torch.clamp(torch.round(orig_dur.float() * scale), min=1).long()
    diff = new_T - int(rescaled.sum().item())
    if diff != 0:
        j = int(rescaled.argmax().item())
        rescaled[j] = max(1, int(rescaled[j].item()) + diff)
    # Brute fix
    for _ in range(20):
        d_now = int(rescaled.sum().item())
        if d_now == new_T:
            break
        if d_now < new_T:
            rescaled[int(rescaled.argmax().item())] += new_T - d_now
        else:
            idx = int(rescaled.argmax().item())
            if rescaled[idx] > 1:
                rescaled[idx] -= d_now - new_T
            else:
                break
    return rescaled


def recompute_pitch_from_f0(f0_np: np.ndarray, duration: torch.Tensor) -> torch.Tensor:
    """Pitch (MIDI) per phoneme from median F0 in the phoneme's frame range."""
    out = []
    last_midi = 0
    pos = 0
    for d in duration.tolist():
        seg = f0_np[pos: pos + d]
        voiced = seg[seg > 30]
        if len(voiced) > 0:
            m = f0_hz_to_midi(float(np.median(voiced)))
            if 24 <= m <= 108:
                last_midi = m
                out.append(m)
                pos += d
                continue
        out.append(last_midi)
        pos += d
    return torch.tensor(out, dtype=torch.long)


def process_dataset(dataset: str, hubert_model, rmvpe, device: str, use_fp16: bool,
                    limit: int = None, dry_run: bool = False):
    cfg = CONFIGS[dataset]
    processed_dir = REPO_ROOT / f"data/processed/{dataset}"
    meta_path = processed_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Only process singing entries (should be all)
    entries_by_song = {}
    for e in meta:
        # utt_id format "<prefix>_<song_id>_<seg_idx>"
        parts = e["id"].split("_")
        seg_idx = int(parts[-1])
        song_id = "_".join(parts[1:-1])  # supports oniku/ofuton song_ids with underscores
        entries_by_song.setdefault(song_id, []).append((seg_idx, e))

    if limit is not None:
        # Keep only first `limit` songs for testing
        keep_songs = list(entries_by_song.keys())[:limit]
        entries_by_song = {k: entries_by_song[k] for k in keep_songs}

    out_dirs = {n: processed_dir / n for n in ("hubert", "f0", "phonemes", "pitch", "duration")}
    new_meta_entries = []

    for song_id, entries in tqdm(entries_by_song.items(), desc=f"[{dataset}]"):
        entries.sort(key=lambda x: x[0])
        wav_stem = cfg["wav_stem_from_song_id"](song_id)
        # Load wav path
        if cfg.get("nested"):
            wav_path = cfg["wav_dir"] / wav_stem / f"{wav_stem}.wav"
            lab_path = cfg["lab_dir"] / wav_stem / f"{wav_stem}.lab"
        else:
            wav_path = cfg["wav_dir"] / f"{wav_stem}.wav"
            lab_path = cfg["lab_dir"] / f"{wav_stem}.lab"
        if not wav_path.exists() or not lab_path.exists():
            tqdm.write(f"missing wav/lab: {wav_stem}")
            continue
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        else:
            audio = audio.astype(np.float32)

        intervals = parse_lab(lab_path, cfg["fmt"])
        segments = segment_by_pauses(intervals)

        entry_by_idx = {idx: e for idx, e in entries}

        for seg_idx, seg in enumerate(segments):
            if seg_idx not in entry_by_idx:
                continue  # orig processing dropped this segment
            orig_entry = entry_by_idx[seg_idx]
            utt_id = orig_entry["id"]
            split = orig_entry["split"]
            # Only augment TRAIN entries — keep test split pure orig for fair eval vs Run 5
            if split != "train":
                continue

            seg_start, seg_end = seg[0][0], seg[-1][1]
            s_samp = int(round(seg_start * SAMPLE_RATE))
            e_samp = int(round(seg_end * SAMPLE_RATE))
            s_samp = max(0, min(s_samp, len(audio)))
            e_samp = max(0, min(e_samp, len(audio)))
            if e_samp - s_samp < int(SEG_MIN * SAMPLE_RATE):
                continue
            audio_seg = audio[s_samp:e_samp].astype(np.float32)

            # Load original tensors
            orig_phonemes = torch.load(out_dirs["phonemes"] / f"{utt_id}.pt", weights_only=False)
            orig_duration = torch.load(out_dirs["duration"] / f"{utt_id}.pt", weights_only=False)
            orig_pitch = torch.load(out_dirs["pitch"] / f"{utt_id}.pt", weights_only=False)

            # WORLD decompose once
            f0, sp, ap = world_decompose(audio_seg, SAMPLE_RATE)

            augments = [
                ("aug_p+2", lambda: world_pitch_shift(f0, sp, ap, SAMPLE_RATE, +2.0)),
                ("aug_p-2", lambda: world_pitch_shift(f0, sp, ap, SAMPLE_RATE, -2.0)),
                ("aug_t1.05", lambda: world_time_stretch(f0, sp, ap, SAMPLE_RATE, 1.05)),
                ("aug_t0.95", lambda: world_time_stretch(f0, sp, ap, SAMPLE_RATE, 0.95)),
            ]

            for tag, gen_fn in augments:
                try:
                    audio_aug = gen_fn()
                except Exception as e:
                    tqdm.write(f"{utt_id} {tag}: gen failed: {e}")
                    continue
                if audio_aug.size < int(SEG_MIN * SAMPLE_RATE * 0.5):
                    continue

                # Re-extract HuBERT + F0
                new_hubert = hubert_extract(hubert_model, audio_aug, device, use_fp16)
                T_new = int(new_hubert.shape[0])
                if T_new < 5:
                    continue
                new_f0 = rmvpe_extract(rmvpe, audio_aug, T_new)

                # Duration tensor: unchanged for pitch_shift, rescaled for time_stretch
                if tag.startswith("aug_p"):
                    new_dur = rescale_duration(orig_duration.clone(), T_new)  # Re-sync T
                else:
                    new_dur = rescale_duration(orig_duration.clone(), T_new)
                # Pitch tensor: ±2 semitones for pitch shift, recompute for time stretch
                if tag == "aug_p+2":
                    new_pitch = orig_pitch.clone()
                    new_pitch[new_pitch > 0] = new_pitch[new_pitch > 0] + 2
                elif tag == "aug_p-2":
                    new_pitch = orig_pitch.clone()
                    new_pitch[new_pitch > 0] = torch.clamp(new_pitch[new_pitch > 0] - 2, min=0)
                else:
                    new_pitch = recompute_pitch_from_f0(new_f0.numpy(), new_dur)

                new_id = f"{utt_id}_{tag}"
                if not dry_run:
                    torch.save(new_hubert, out_dirs["hubert"] / f"{new_id}.pt")
                    torch.save(new_f0, out_dirs["f0"] / f"{new_id}.pt")
                    torch.save(orig_phonemes, out_dirs["phonemes"] / f"{new_id}.pt")
                    torch.save(new_pitch, out_dirs["pitch"] / f"{new_id}.pt")
                    torch.save(new_dur, out_dirs["duration"] / f"{new_id}.pt")

                new_meta_entries.append({
                    "id": new_id,
                    "n_phones": orig_entry["n_phones"],
                    "n_frames": T_new,
                    "language": "ja",
                    "lang_id": orig_entry.get("lang_id", 0),
                    "singing_ratio": 1.0,
                    "dataset": dataset,
                    "singer": orig_entry.get("singer", dataset),
                    "split": split,
                    "augment": tag,
                })

    if not dry_run and new_meta_entries:
        # Backup + merge
        backup_path = meta_path.with_suffix(".json.pre_aug_backup")
        if not backup_path.exists():
            backup_path.write_text(json.dumps(meta, indent=0), encoding="utf-8")
        merged = meta + new_meta_entries
        meta_path.write_text(json.dumps(merged, indent=0), encoding="utf-8")
        print(f"[{dataset}] wrote {len(new_meta_entries)} aug entries + backed up metadata.json")
    else:
        print(f"[{dataset}] (dry) would write {len(new_meta_entries)} aug entries")
    return len(new_meta_entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CONFIGS.keys()) + ["all"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="process first N songs per dataset (test)")
    ap.add_argument("--hubert-model", default=str(REPO_ROOT / "RVC/RVC20240604Nvidia/assets/hubert/hubert_base.pt"))
    ap.add_argument("--rmvpe-model", default=str(REPO_ROOT / "RVC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    use_fp16 = True

    print("Loading HuBERT...")
    hx = HuBERTExtractor(args.hubert_model, layer=12, device=args.device, use_fp16=use_fp16)
    hubert_model = hx.model
    print("Loading RMVPE...")
    from infer.lib.rmvpe import RMVPE
    rmvpe = RMVPE(args.rmvpe_model, is_half=use_fp16, device=args.device)

    targets = list(CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    total = 0
    for ds in targets:
        total += process_dataset(ds, hubert_model, rmvpe, args.device, use_fp16,
                                  limit=args.limit, dry_run=args.dry_run)
    print(f"\nTotal new aug entries: {total}")


if __name__ == "__main__":
    main()
