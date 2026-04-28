"""Re-align ACE-Opencpop tensors using SOFA-Mandarin outputs.

ACE-specific adapter (complement to preprocess_sofa_realign.py for JA).
ACE's vocab stores IPA (not opencpop-ext); original ACE preprocessing converted opencpop
initials+finals → IPA segments via src/data/phoneme_v1.PINYIN_{INITIAL,FINAL}_TO_IPA.
This script does the same conversion on SOFA output so new phoneme tensors match vocab.

Workflow per ACE arrow row:
  1. Read orig (phn list) from arrow shard
  2. Parse SOFA lab from SOFA output dir (HTK 100ns format)
  3. Verify voiced phoneme count match at opencpop level
  4. Compute per-SOFA-phoneme frame durations from SOFA timings
  5. Split each opencpop phoneme into IPA segments with integer-partitioned durations
  6. Encode via existing ace vocab (IPA tokens)
  7. Recompute per-segment pitch from existing F0 tensor
  8. Backup {phonemes,duration,pitch} to data/processed_pre_sofa/ace_opencpop/ and save new

Blocked on: SOFA-Mandarin inference (step 2) completing first.

Usage:
    py -3.10 scripts/preprocess_sofa_realign_ace.py --dry-run --limit 100
    py -3.10 scripts/preprocess_sofa_realign_ace.py --workers 8
"""

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.phoneme import PAD_ID, SP_ID, PhonemeVocab  # noqa: E402
from src.data.phoneme_v1 import (  # noqa: E402
    PINYIN_INITIAL_TO_IPA,
    PINYIN_FINAL_TO_IPA,
)

FRAME_RATE = 50

# Opencpop-ext / ACE pauses (original arrow uses SP + AP)
PAUSE_ORIG = {"SP", "AP"}
# SOFA-Mandarin pause tokens
PAUSE_SOFA = {"SP", "AP", "EP", "<EOS>"}

ARROW_DIR = Path(
    "D:/MyDev/MSSTRVC/score2hubert/data/raw/ace_opencpop/"
    "espnet___ace-opencpop-segments/default/0.0.0/"
    "a66b1ab0b3bdebfece8d78fd2980dfeebbcfd67c"
)
SOFA_RUN_DIR = Path("D:/MyDev/MSSTRVC/SOFA/segments/sofa_run/ace_opencpop")
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "ace_opencpop"
BACKUP_DIR = REPO_ROOT / "data" / "processed_pre_sofa" / "ace_opencpop"
UTT_PREFIX = "ace_"  # existing processed ids: "ace_<segment_id>"


def opencpop_to_ipa_segs(ph: str) -> list:
    """Map one opencpop-ext phoneme to list of IPA atomic segments (like _convert_pinyin_to_ipa)."""
    if ph in PAUSE_SOFA:
        return ["SP"]
    ph_clean = ph.rstrip("0123456789")
    if ph_clean in ("SP", "AP", "br", "sil", "pau", "sp", ""):
        return ["SP"]
    if ph_clean in PINYIN_INITIAL_TO_IPA:
        return PINYIN_INITIAL_TO_IPA[ph_clean]
    if ph_clean in PINYIN_FINAL_TO_IPA:
        return PINYIN_FINAL_TO_IPA[ph_clean]
    return ["SP"]  # unknown → silence (matches existing behavior)


def split_opencpop_to_ipa(phs, dur_frames):
    """Per-phoneme opencpop+duration -> per-IPA-segment with integer duration partition.

    Guarantees each IPA seg >= 1 frame (short opencpop phones with multi-seg IPA split
    would otherwise end up with 0-frame segs, failing drift-fail).
    When d < n, each seg gets 1 frame (sum overflows by n-d; final drift-fix absorbs).
    When d >= n, standard per/rem allocator preserves sum exactly.
    """
    out_ph, out_dur = [], []
    for ph, d in zip(phs, dur_frames):
        segs = opencpop_to_ipa_segs(ph)
        n = len(segs)
        if n == 1:
            out_ph.append(segs[0])
            out_dur.append(max(1, d))
        elif d >= n:
            per = d // n
            rem = d - per * n
            for i, s in enumerate(segs):
                out_ph.append(s)
                out_dur.append(per + (rem if i == n - 1 else 0))
        else:
            # d < n: give each seg 1 frame (overflow handled by caller's drift fix)
            for s in segs:
                out_ph.append(s)
                out_dur.append(1)
    return out_ph, out_dur


def parse_sofa_lab_htk(path: Path):
    intervals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 3:
                continue
            try:
                s = int(p[0]) * 1e-7
                e = int(p[1]) * 1e-7
                intervals.append((s, e, p[2]))
            except ValueError:
                continue
    return intervals


def find_sofa_lab(singer: str, utt: str) -> Path:
    """SOFA htk output is at <wav_parent>/htk/phones/<utt>.lab (per export_tool.Exporter.save_htk).
    Since our input wavs live per-singer, output is at <singer>/htk/phones/<utt>.lab.
    """
    path = SOFA_RUN_DIR / singer / "htk" / "phones" / f"{utt}.lab"
    return path if path.exists() else None


def f0_hz_to_midi(hz: float) -> int:
    if hz <= 0:
        return 0
    return int(round(69.0 + 12.0 * np.log2(hz / 440.0)))


_worker_vocab = None


def _init_worker(vocab_path_str):
    """Initialize per-worker: load vocab once (avoid 100K re-parses)."""
    global _worker_vocab
    _worker_vocab = PhonemeVocab(vocab_path_str)


def realign_one(args_tuple):
    """Process one ACE segment. Runs in worker process."""
    segment_id, orig_phns, dry_run = args_tuple
    global _worker_vocab
    vocab = _worker_vocab  # loaded in _init_worker

    singer, utt = segment_id.split("#")
    utt_id = f"{UTT_PREFIX}{segment_id}"

    sofa_lab = find_sofa_lab(singer, utt)
    if sofa_lab is None:
        return utt_id, "no_sofa_lab"

    sofa_intervals = parse_sofa_lab_htk(sofa_lab)
    if not sofa_intervals:
        return utt_id, "empty_sofa_lab"

    # Input-side check: lab is an SOFA text file where each line has (start, end, phoneme).
    # If it was the INPUT (space-separated phoneme sequence), parse_sofa_lab_htk returns
    # [] because lines don't have 3 columns. So we already caught that.

    hub_path = PROCESSED_DIR / "hubert" / f"{utt_id}.pt"
    if not hub_path.exists():
        return utt_id, "no_hubert"
    f0_path = PROCESSED_DIR / "f0" / f"{utt_id}.pt"
    if not f0_path.exists():
        return utt_id, "no_f0"

    hubert = torch.load(hub_path, weights_only=False)
    T_frame = int(hubert.shape[0])
    if T_frame < 5:
        return utt_id, "too_short"

    # Voiced count match at opencpop level (before IPA split)
    orig_voiced = [p for p in orig_phns if p not in PAUSE_ORIG]
    sofa_voiced = [p for _, _, p in sofa_intervals if p not in PAUSE_SOFA]
    if len(orig_voiced) != len(sofa_voiced):
        return utt_id, f"count_mismatch_{len(orig_voiced)}v{len(sofa_voiced)}"

    # Frame durations per SOFA phoneme
    seg_start = sofa_intervals[0][0]
    zh_phs, zh_durs = [], []
    for s, e, p in sofa_intervals:
        local_s = max(0.0, s - seg_start)
        local_e = max(local_s, e - seg_start)
        start_f = int(round(local_s * FRAME_RATE))
        end_f = int(round(local_e * FRAME_RATE))
        d = max(1, end_f - start_f)
        zh_phs.append(p)
        zh_durs.append(d)

    # Rescale zh_durs to sum=T_frame
    cur_sum = sum(zh_durs)
    if cur_sum != T_frame and cur_sum > 0:
        scale = T_frame / cur_sum
        rescaled = [max(1, int(round(d * scale))) for d in zh_durs]
        diff = T_frame - sum(rescaled)
        if diff != 0:
            j = int(np.argmax(rescaled))
            rescaled[j] = max(1, rescaled[j] + diff)
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
                    return utt_id, "drift_fail"
        if sum(rescaled) != T_frame:
            return utt_id, "drift_fail"
        zh_durs = rescaled

    # opencpop -> IPA atomic
    ipa_phs, ipa_durs = split_opencpop_to_ipa(zh_phs, zh_durs)

    # Final drift fix — iterate to absorb any overflow/deficit from short-phone IPA splits
    for _ in range(40):
        diff = T_frame - sum(ipa_durs)
        if diff == 0:
            break
        if diff > 0:
            # Grow longest
            ipa_durs[int(np.argmax(ipa_durs))] += diff
        else:
            # Shrink longest (only if > 1)
            idx = int(np.argmax(ipa_durs))
            if ipa_durs[idx] + diff >= 1:
                ipa_durs[idx] += diff
            else:
                take = ipa_durs[idx] - 1
                if take <= 0:
                    break  # all segs are 1; can't compress further
                ipa_durs[idx] = 1
                # Re-loop — remaining overflow handled next iter on new argmax
    if sum(ipa_durs) != T_frame:
        return utt_id, "drift_fail"
    if any(d < 1 for d in ipa_durs):
        return utt_id, "drift_fail"

    # Encode via vocab (IPA tokens). Unknown → PAD_ID=0; replace with SP_ID.
    encoded = []
    for p in ipa_phs:
        tid = vocab.phone2id.get(p, PAD_ID)
        if tid == PAD_ID:
            tid = SP_ID
        encoded.append(tid)

    ph_ids = torch.tensor(encoded, dtype=torch.long)
    dur_tensor = torch.tensor(ipa_durs, dtype=torch.long)

    # Recompute pitch from F0 per IPA segment
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
    pitch_tensor = torch.tensor(midi_list, dtype=torch.long)

    if not dry_run:
        for name, tensor in [
            ("phonemes", ph_ids),
            ("duration", dur_tensor),
            ("pitch", pitch_tensor),
        ]:
            src = PROCESSED_DIR / name / f"{utt_id}.pt"
            bk = BACKUP_DIR / name / f"{utt_id}.pt"
            if src.exists() and not bk.exists():
                bk.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, bk)
            torch.save(tensor, src)

    return utt_id, "OK"


def iter_arrow_rows(shard_path: Path):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(shard_path), "r") as source:
        table = ipc.RecordBatchStreamReader(source).read_all()
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pydict()
        sid = row["segment_id"][0]
        phns = list(row["phn"][0])
        yield sid, phns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    shards = sorted(ARROW_DIR.glob("*.arrow"))
    print(f"Shards: {len(shards)}")

    vocab_path = str(PROCESSED_DIR / "vocab.yaml")

    # Pre-filter: only process utts that have existing HuBERT tensors (original
    # ACE preprocessing excluded test split + zero-audio rows → they have no tensors
    # and shouldn't be realigned)
    hub_dir = PROCESSED_DIR / "hubert"
    existing = set()
    for f in hub_dir.glob("*.pt"):
        existing.add(f.stem)
    print(f"Existing processed utts: {len(existing)}")

    tasks = []
    n_skipped_missing = 0
    for shard in shards:
        for sid, phns in iter_arrow_rows(shard):
            utt_id = f"{UTT_PREFIX}{sid}"
            if utt_id not in existing:
                n_skipped_missing += 1
                continue
            tasks.append((sid, phns, args.dry_run))
            if args.limit and len(tasks) >= args.limit:
                break
        if args.limit and len(tasks) >= args.limit:
            break
    print(f"Rows to realign: {len(tasks)}  (skipped {n_skipped_missing} missing-hubert rows)")

    from collections import Counter
    status_counter = Counter()
    t0 = time.time()
    sample_failures = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(vocab_path,),
    ) as ex:
        futures = [ex.submit(realign_one, t) for t in tasks]
        done = 0
        for fut in as_completed(futures):
            utt_id, status = fut.result()
            bucket = "count_mismatch" if status.startswith("count_mismatch") else status
            status_counter[bucket] += 1
            done += 1
            if status != "OK" and len(sample_failures) < 30:
                sample_failures.append((utt_id, status))
            if done % 5000 == 0:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-3)
                ok = status_counter.get("OK", 0)
                print(f"[{done}/{len(tasks)}] ok_rate={100*ok/done:.1f}% "
                      f"({rate:.0f}/s, {elapsed:.0f}s)")

    elapsed = time.time() - t0
    print("-" * 60)
    total = sum(status_counter.values())
    ok = status_counter.get("OK", 0)
    print(f"TOTAL: {total} entries, OK={ok} ({100*ok/max(1,total):.1f}%), wall={elapsed:.0f}s")
    for st, n in sorted(status_counter.items(), key=lambda x: -x[1]):
        print(f"  {st}: {n}")
    if sample_failures:
        print("\nSample failures (first 30):")
        for sid, st in sample_failures[:30]:
            print(f"  {sid}: {st}")


if __name__ == "__main__":
    main()
