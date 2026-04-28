"""Re-align Score2HuBERT v2 JA datasets using SOFA-JA outputs.

Workflow per (dataset, song):
    1. Parse ORIGINAL lab → segment_by_pauses → segments (preserves sample id structure)
    2. Parse SOFA lab (from D:/MyDev/MSSTRVC/SOFA/segments/sofa_run/<dataset>/htk/phones/<song>.lab)
    3. For each segment: filter SOFA intervals to [seg_start, seg_end]
    4. Verify voiced phoneme count matches; convert to IPA atomic; build duration tensor
    5. Use EXISTING HuBERT/F0 tensors (don't re-extract); recompute pitch (MIDI) from F0
    6. Backup old phonemes/duration/pitch → data/processed_pre_sofa/<dataset>/
    7. Save new phonemes/duration/pitch tensors

Usage:
    py -3.10 scripts/preprocess_sofa_realign.py --dataset kiritan --dry-run --limit 1
    py -3.10 scripts/preprocess_sofa_realign.py --dataset all
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.phoneme_v1 import OPENJTALK_TO_IPA, PhonemeVocab  # noqa: E402

FRAME_RATE = 50
PAUSE_ORIG = {"pau", "sil", "br", "cl", "sp"}
PAUSE_SOFA = {"SP", "AP", "EP"}
DEVOICED_MAP = {"A": "a", "I": "i", "U": "u", "E": "e", "O": "o"}

LONG_PAUSE = 0.3
SEG_MIN = 1.5
SEG_MAX = 12.0

SOFA_LAB_BASE = Path("D:/MyDev/MSSTRVC/SOFA/segments/sofa_run")
PROCESSED_BASE = REPO_ROOT / "data" / "processed"
BACKUP_BASE = REPO_ROOT / "data" / "processed_pre_sofa"

CONFIGS = {
    "kiritan": {
        "fmt": "sec",
        "orig_lab_dir": REPO_ROOT / "kiritan/kiritan_singing/kiritan_singing/mono_label",
        "song_pattern": "*.lab",
        "id_prefix": "kiritan",
        "song_id_from_stem": lambda stem: stem,
    },
    "itako": {
        "fmt": "htk",
        "orig_lab_dir": REPO_ROOT / "kiritan/itako_singing/mono_label",
        "song_pattern": "*.lab",
        "id_prefix": "itako",
        "song_id_from_stem": lambda stem: stem.replace("itako", ""),
    },
    "oniku": {
        "fmt": "htk",
        "orig_lab_dir": REPO_ROOT / "ONIKU_KURUMI_UTAGOE_DB",
        "song_pattern": "*/*.lab",
        "id_prefix": "oniku",
        "song_id_from_stem": lambda stem: stem,
    },
    "ofuton": {
        "fmt": "htk",
        "orig_lab_dir": REPO_ROOT / "OFUTON_P_UTAGOE_DB/OFUTON_P_UTAGOE_DB",
        "song_pattern": "*/*.lab",
        "id_prefix": "ofuton",
        "song_id_from_stem": lambda stem: stem,
    },
    "natsume": {
        "fmt": "sec",
        "orig_lab_dir": REPO_ROOT / "Natsume_Singing_DB_0713/mono_label",
        "song_pattern": "*.lab",
        "id_prefix": "natsume",
        "song_id_from_stem": lambda stem: stem,
    },
}


def parse_lab(path: Path, fmt: str):
    intervals = []
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
                intervals.append((s, e, p[2]))
            except ValueError:
                continue
    return intervals


def segment_by_pauses(intervals, long_pause=LONG_PAUSE, max_sec=SEG_MAX):
    """Replicates preprocess_kiritan_itako.segment_by_pauses (keeping behavior identical)."""
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


def jp_phoneme_to_ipa_segs(ph: str):
    """Map SOFA-JA phoneme to IPA atomic segments. Reuses OPENJTALK_TO_IPA."""
    if ph in PAUSE_SOFA:
        return ["SP"]
    if ph in DEVOICED_MAP:  # safety, should already be lowered
        ph = DEVOICED_MAP[ph]
    if ph in OPENJTALK_TO_IPA:
        return list(OPENJTALK_TO_IPA[ph])
    if ph == "gw":
        return ["g", "w"]
    if ph == "kw":
        return ["k", "w"]
    if ph.startswith("0") and ph[1:] in OPENJTALK_TO_IPA:
        # GS-prefix glottal-stop vowel (e.g. 0a → ʔ + a)
        return ["ʔ"] + list(OPENJTALK_TO_IPA[ph[1:]])
    return ["SP"]


def split_jp_to_ipa(phs_jp, dur_frames):
    """Convert per-phoneme JP+duration to per-IPA-segment, splitting durations."""
    out_ph, out_dur = [], []
    for ph, d in zip(phs_jp, dur_frames):
        segs = jp_phoneme_to_ipa_segs(ph)
        n = len(segs)
        if n == 1:
            out_ph.append(segs[0])
            out_dur.append(d)
        else:
            per = d // n
            rem = d - per * n
            for i, s in enumerate(segs):
                out_ph.append(s)
                out_dur.append(per + (rem if i == n - 1 else 0))
    return out_ph, out_dur


def f0_hz_to_midi(hz: float) -> int:
    if hz <= 0:
        return 0
    return int(round(69.0 + 12.0 * np.log2(hz / 440.0)))


def realign_segment(orig_seg, sofa_intervals, T_frame):
    """Build new (ipa_phs, ipa_durs) for one segment using SOFA timings.

    Returns (ipa_phs, ipa_durs) on success, or None if voiced count mismatch / fix-up failed.
    """
    seg_start = orig_seg[0][0]
    seg_end = orig_seg[-1][1]

    # Filter SOFA intervals overlapping [seg_start, seg_end]
    sofa_in_seg = []
    for s, e, p in sofa_intervals:
        if e <= seg_start + 1e-6:
            continue
        if s >= seg_end - 1e-6:
            continue
        cs = max(s, seg_start)
        ce = min(e, seg_end)
        if ce - cs < 1e-3:
            continue
        sofa_in_seg.append((cs, ce, p))

    if not sofa_in_seg:
        return None

    # Verify voiced count match
    orig_voiced = [(s, e, p) for s, e, p in orig_seg if p not in PAUSE_ORIG]
    sofa_voiced = [(s, e, p) for s, e, p in sofa_in_seg if p not in PAUSE_SOFA]
    if len(orig_voiced) != len(sofa_voiced):
        return None  # boundary issue

    # Frame durations (segment-local time)
    jp_phs, jp_durs = [], []
    for s, e, p in sofa_in_seg:
        local_s = max(0.0, s - seg_start)
        local_e = max(local_s, e - seg_start)
        start_f = int(round(local_s * FRAME_RATE))
        end_f = int(round(local_e * FRAME_RATE))
        d = max(1, end_f - start_f)
        jp_phs.append(p)
        jp_durs.append(d)

    # Rescale to sum=T_frame
    cur_sum = sum(jp_durs)
    if cur_sum != T_frame and cur_sum > 0:
        scale = T_frame / cur_sum
        rescaled = [max(1, int(round(d * scale))) for d in jp_durs]
        diff = T_frame - sum(rescaled)
        if diff != 0:
            j = int(np.argmax(rescaled))
            rescaled[j] = max(1, rescaled[j] + diff)
        # Final brute fix
        for _ in range(20):
            d_now = sum(rescaled)
            if d_now == T_frame:
                break
            if d_now < T_frame:
                rescaled[int(np.argmax(rescaled))] += T_frame - d_now
            else:
                idx = int(np.argmax(rescaled))
                if rescaled[idx] > 1:
                    rescaled[idx] -= d_now - T_frame
                else:
                    return None
        if sum(rescaled) != T_frame:
            return None
        jp_durs = rescaled

    # JP → IPA atomic
    ipa_phs, ipa_durs = split_jp_to_ipa(jp_phs, jp_durs)

    # Final drift fix
    diff = T_frame - sum(ipa_durs)
    if diff != 0:
        j = int(np.argmax(ipa_durs))
        ipa_durs[j] = max(1, ipa_durs[j] + diff)
    if sum(ipa_durs) != T_frame:
        return None
    if any(d < 1 for d in ipa_durs):
        return None

    return ipa_phs, ipa_durs


def realign_song(song_id, orig_lab, sofa_lab, fmt, processed_dir, backup_dir, vocab,
                 id_prefix, dry_run=False):
    if not sofa_lab.exists():
        return [(f"{id_prefix}_{song_id}", "no_sofa_lab")]

    orig_intervals = parse_lab(orig_lab, fmt)
    if not orig_intervals:
        return [(f"{id_prefix}_{song_id}", "no_orig")]
    sofa_intervals = parse_lab(sofa_lab, "htk")
    if not sofa_intervals:
        return [(f"{id_prefix}_{song_id}", "no_sofa")]

    segments = segment_by_pauses(orig_intervals)

    results = []
    for seg_idx, seg in enumerate(segments):
        utt_id = f"{id_prefix}_{song_id}_{seg_idx:03d}"

        seg_dur = seg[-1][1] - seg[0][0]
        if seg_dur < SEG_MIN or seg_dur > SEG_MAX:
            results.append((utt_id, "skipped_min_max"))
            continue

        hub_path = processed_dir / "hubert" / f"{utt_id}.pt"
        if not hub_path.exists():
            results.append((utt_id, "no_existing_hubert"))
            continue
        hubert = torch.load(hub_path, weights_only=False)
        T_frame = int(hubert.shape[0])
        if T_frame < 5:
            results.append((utt_id, "too_short"))
            continue

        out = realign_segment(seg, sofa_intervals, T_frame)
        if out is None:
            results.append((utt_id, "realign_failed"))
            continue
        ipa_phs, ipa_durs = out

        ph_ids = vocab.encode(ipa_phs)

        # Recompute pitch from F0
        f0_path = processed_dir / "f0" / f"{utt_id}.pt"
        if not f0_path.exists():
            results.append((utt_id, "no_existing_f0"))
            continue
        f0 = torch.load(f0_path, weights_only=False).numpy()
        midi_list = []
        last_midi = 0
        pos = 0
        for d in ipa_durs:
            sl = f0[pos: pos + d]
            voiced = sl[sl > 0]
            if len(voiced) > 0:
                med_hz = float(np.median(voiced))
                m = f0_hz_to_midi(med_hz)
                if 24 <= m <= 108:
                    last_midi = m
                    midi_list.append(m)
                    pos += d
                    continue
            midi_list.append(last_midi)
            pos += d

        if not dry_run:
            for name, tensor in [
                ("phonemes", torch.tensor(ph_ids, dtype=torch.long)),
                ("duration", torch.tensor(ipa_durs, dtype=torch.long)),
                ("pitch", torch.tensor(midi_list, dtype=torch.long)),
            ]:
                src = processed_dir / name / f"{utt_id}.pt"
                bk = backup_dir / name / f"{utt_id}.pt"
                bk.parent.mkdir(parents=True, exist_ok=True)
                if src.exists() and not bk.exists():
                    shutil.copy(src, bk)
                torch.save(tensor, src)

        results.append((utt_id, "OK"))
    return results


def realign_dataset(dataset, dry_run=False, limit=None):
    cfg = CONFIGS[dataset]
    processed_dir = PROCESSED_BASE / dataset
    backup_dir = BACKUP_BASE / dataset
    sofa_lab_dir = SOFA_LAB_BASE / dataset / "htk" / "phones"

    vocab = PhonemeVocab(str(processed_dir / "vocab.yaml"))

    orig_labs = sorted(cfg["orig_lab_dir"].glob(cfg["song_pattern"]))
    if limit:
        orig_labs = orig_labs[:limit]
    print(f"[{dataset}] {len(orig_labs)} songs (dry_run={dry_run})")

    all_results = []
    for orig_lab in orig_labs:
        wav_stem = orig_lab.stem
        sofa_lab = sofa_lab_dir / f"{wav_stem}.lab"
        song_id = cfg["song_id_from_stem"](wav_stem)
        results = realign_song(
            song_id, orig_lab, sofa_lab, cfg["fmt"],
            processed_dir, backup_dir, vocab, cfg["id_prefix"], dry_run=dry_run,
        )
        all_results.extend(results)

    statuses = {}
    for _, st in all_results:
        statuses[st] = statuses.get(st, 0) + 1
    total = sum(statuses.values())
    ok = statuses.get("OK", 0)
    print(f"[{dataset}] {total} entries: OK={ok} ({100 * ok / max(1, total):.1f}%)")
    for st, n in sorted(statuses.items()):
        if st != "OK":
            print(f"  {st}: {n}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(CONFIGS.keys()) + ["all"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="limit N songs (for testing)")
    args = parser.parse_args()

    targets = list(CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        realign_dataset(ds, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
