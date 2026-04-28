"""ACE-Opencpop arrow -> 16kHz wav + SOFA-format lab, for SOFA-Mandarin forced alignment.

Source arrow:
  D:/MyDev/MSSTRVC/score2hubert/data/raw/ace_opencpop/.../arrow-train-*.arrow (82 shards)
  + arrow-test-*.arrow (5 shards)
  Each row: audio (48kHz PCM_16 WAV bytes) + segment_id + phn list + timings.

Output:
  D:/MyDev/MSSTRVC/SOFA/segments/sofa_run/ace_opencpop/<singer>/<utt>.{wav,lab}
  where segment_id = "<singer>#<utt>" e.g. "acesinger_1#2001000001".

  .wav: 16kHz mono PCM_16 (downsampled from 48kHz via soxr VHQ)
  .lab: space-separated phonemes, AP -> SP, opencpop-extension vocab (matches SOFA-Mandarin)

Resumable: existing wav+lab pairs are skipped.
Parallel: one task per shard, 8 workers via ProcessPoolExecutor.
"""

import argparse
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ARROW_DIR = Path(
    "D:/MyDev/MSSTRVC/score2hubert/data/raw/ace_opencpop/"
    "espnet___ace-opencpop-segments/default/0.0.0/"
    "a66b1ab0b3bdebfece8d78fd2980dfeebbcfd67c"
)
OUT_ROOT = Path("D:/MyDev/MSSTRVC/SOFA/segments/sofa_run/ace_opencpop")
DST_SR = 16000
SRC_SR = 48000

# ACE -> SOFA-Mandarin vocab mapping
# AP (breath/aspiration) collapses to SP (silence); all other opencpop-ext phonemes pass through.
PHN_MAP = {"AP": "SP"}


def process_shard(shard_path_str: str) -> tuple:
    """Process one arrow shard. Runs in a worker process.

    Returns (shard_name, ok, skipped, failed, bad_ids).
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import soundfile as sf
    import librosa

    shard_path = Path(shard_path_str)
    ok = skipped = failed = 0
    bad_ids = []

    with pa.memory_map(str(shard_path), "r") as source:
        table = ipc.RecordBatchStreamReader(source).read_all()

    n = table.num_rows
    for i in range(n):
        row = table.slice(i, 1).to_pydict()
        sid = row["segment_id"][0]
        try:
            singer, utt = sid.split("#")
        except ValueError:
            failed += 1
            bad_ids.append((sid, "bad_segment_id"))
            continue

        out_dir = OUT_ROOT / singer
        out_wav = out_dir / f"{utt}.wav"
        out_lab = out_dir / f"{utt}.lab"
        if out_wav.exists() and out_lab.exists():
            skipped += 1
            continue

        try:
            aud_bytes = row["audio"][0]["bytes"]
            phns = row["phn"][0]

            # ACE source arrow has ~430 zero-audio entries (44B header-only WAV) in the
            # "2026001XXX" utt range — intentionally invalid segments filtered by original
            # ACE preprocessing. Match that filter here so SOFA doesn't crash on them.
            if len(aud_bytes) < 100:
                failed += 1
                bad_ids.append((sid, f"empty_audio_{len(aud_bytes)}B"))
                continue

            data, sr = sf.read(io.BytesIO(aud_bytes), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1).astype(np.float32)
            if sr != SRC_SR:
                # Unexpected but tolerate — use whatever we got
                pass

            resampled = librosa.resample(
                y=data, orig_sr=sr, target_sr=DST_SR, res_type="soxr_vhq"
            ).astype(np.float32)

            lab_phns = [PHN_MAP.get(p, p) for p in phns]
            lab_line = " ".join(lab_phns)

            out_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_wav), resampled, DST_SR, subtype="PCM_16")
            out_lab.write_text(lab_line, encoding="utf-8")
            ok += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            bad_ids.append((sid, str(e)[:200]))

    return shard_path.name, ok, skipped, failed, bad_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-shards", type=int, default=None,
                   help="only process first N shards (testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="enumerate + report only; don't process")
    args = p.parse_args()

    shards = sorted(ARROW_DIR.glob("*.arrow"))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    print(f"Shards: {len(shards)} (train+test)")
    print(f"Output root: {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for s in shards:
            print(f"  would process: {s.name}")
        return

    t0 = time.time()
    total_ok = total_skipped = total_failed = 0
    all_bad = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_shard, str(s)): s for s in shards}
        done = 0
        for fut in as_completed(futures):
            name, ok, skipped, failed, bad_ids = fut.result()
            total_ok += ok
            total_skipped += skipped
            total_failed += failed
            all_bad.extend(bad_ids)
            done += 1
            elapsed = time.time() - t0
            rate = (total_ok + total_skipped) / max(elapsed, 1e-3)
            print(
                f"[{done}/{len(shards)}] {name}: "
                f"ok={ok} skip={skipped} fail={failed}  "
                f"(cum: {total_ok}/{total_skipped}/{total_failed} @ {rate:.0f}/s, {elapsed:.0f}s)"
            )

    print("-" * 60)
    print(f"TOTAL: ok={total_ok} skipped={total_skipped} failed={total_failed}")
    print(f"Wall: {time.time() - t0:.0f}s")
    if all_bad:
        print(f"\nFirst 20 failures:")
        for sid, msg in all_bad[:20]:
            print(f"  {sid}: {msg}")


if __name__ == "__main__":
    main()
