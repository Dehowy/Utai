"""Fix pitch[phonemes==SP]=0 in preprocessed tensors.

Bug: preprocessing scripts (all per-dataset) carried MIDI note over SP phoneme frames.
This caused model training to see SP frames with non-zero pitch input, conflicting with
SP=0 force in trainer. Run 8 inference-side fix was in model.forward (pitch.clone +
pitch[phonemes==SP_ID]=0 at line 96), but training data still had the bug.

This script fixes at source: iterates pitch tensors, masks positions where corresponding
phoneme == SP_ID (3 in our vocab), sets those pitch values to 0, saves back.

Idempotent: if already fixed (all SP positions are 0), no-op.
Backup: overwritten tensors are first copied to data/processed/<ds>/pitch.pre_sp_fix/.

Usage:
    py -3.10 scripts/fix_sp_pitch.py --datasets all
    py -3.10 scripts/fix_sp_pitch.py --datasets ace_opencpop  # after ACE SOFA realign
    py -3.10 scripts/fix_sp_pitch.py --datasets kiritan itako oniku ofuton natsume jvs_music
"""

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_BASE = REPO_ROOT / "data" / "processed"

# SP token id in our vocab (verified: src/model/score2hubert.py line 94 hardcodes SP_ID=3,
# and all datasets use the same data/processed/ace_opencpop/vocab.yaml)
SP_ID = 3

ALL_DATASETS = [
    "ace_opencpop",
    "kiritan",
    "itako",
    "oniku",
    "ofuton",
    "natsume",
    "jvs_music",
]


def verify_sp_id(vocab_path: Path) -> int:
    """Read vocab.yaml and confirm SP maps to SP_ID."""
    with open(vocab_path, encoding="utf-8") as f:
        vocab = yaml.safe_load(f)
    tokens = vocab.get("tokens") or vocab.get("phonemes") or vocab
    if isinstance(tokens, dict):
        items = list(tokens.items())
    elif isinstance(tokens, list):
        items = [(t, i) for i, t in enumerate(tokens)]
    else:
        return -1
    for tok, idx in items:
        if tok == "SP":
            return int(idx)
    return -1


def _fix_one(args_tuple):
    """Process one pitch tensor. Runs in worker process. Returns status tuple."""
    pitch_path_str, ph_path_str, backup_path_str, do_backup, dry_run = args_tuple
    pitch_path = Path(pitch_path_str)
    ph_path = Path(ph_path_str)
    if not ph_path.exists():
        return "missing_ph", 0

    pitch = torch.load(pitch_path, weights_only=False)
    phonemes = torch.load(ph_path, weights_only=False)

    if pitch.shape[0] != phonemes.shape[0]:
        return "mismatch", 0

    sp_mask = phonemes == SP_ID
    if not sp_mask.any():
        return "clean", 0

    leaked = int((pitch[sp_mask] != 0).sum().item())
    if leaked == 0:
        return "clean", 0

    if not dry_run:
        if do_backup and backup_path_str:
            bp = Path(backup_path_str)
            if not bp.exists():
                bp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(pitch_path), str(bp))
        pitch_fixed = pitch.clone()
        pitch_fixed[sp_mask] = 0
        torch.save(pitch_fixed, str(pitch_path))
    return "changed", leaked


def fix_dataset(dataset: str, dry_run: bool = False, backup: bool = False,
                workers: int = 8) -> dict:
    ds_dir = PROCESSED_BASE / dataset
    if not ds_dir.exists():
        print(f"[{dataset}] SKIP: {ds_dir} not found")
        return {"dataset": dataset, "status": "missing"}

    vocab_path = ds_dir / "vocab.yaml"
    if vocab_path.exists():
        sp_in_vocab = verify_sp_id(vocab_path)
        if sp_in_vocab != -1 and sp_in_vocab != SP_ID:
            print(f"[{dataset}] WARN: vocab.yaml says SP={sp_in_vocab}, expected {SP_ID}")

    phoneme_dir = ds_dir / "phonemes"
    pitch_dir = ds_dir / "pitch"
    backup_dir = ds_dir / "pitch.pre_sp_fix"

    if not phoneme_dir.exists() or not pitch_dir.exists():
        print(f"[{dataset}] SKIP: no phonemes or pitch dir")
        return {"dataset": dataset, "status": "missing_tensors"}

    pitch_files = sorted(pitch_dir.glob("*.pt"))
    print(f"[{dataset}] {len(pitch_files)} pitch tensors, {workers} workers")

    tasks = [(str(p), str(phoneme_dir / f"{p.stem}.pt"),
              str(backup_dir / f"{p.stem}.pt") if backup else "", backup, dry_run)
             for p in pitch_files]

    counts = {"changed": 0, "clean": 0, "missing_ph": 0, "mismatch": 0}
    sp_fixed = 0
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for status, leaked in ex.map(_fix_one, tasks, chunksize=50):
            counts[status] = counts.get(status, 0) + 1
            sp_fixed += leaked
            done += 1
            if done % 20000 == 0:
                rate = done / max(time.time() - t0, 1e-3)
                print(f"  {done}/{len(tasks)} @ {rate:.0f}/s")

    elapsed = time.time() - t0
    print(
        f"[{dataset}] DONE {elapsed:.0f}s: changed={counts['changed']} "
        f"clean={counts['clean']} missing_ph={counts['missing_ph']} "
        f"mismatch={counts['mismatch']} sp_zeroed={sp_fixed}"
    )
    return {
        "dataset": dataset, "status": "ok", "total": len(pitch_files),
        "changed": counts["changed"], "sp_positions_fixed": sp_fixed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["all"],
                   help=f"subset of {ALL_DATASETS} or 'all'")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--backup", action="store_true",
                   help="backup pre-fix tensors (slow — redundant since fix is deterministic)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    targets = ALL_DATASETS if "all" in args.datasets else args.datasets
    print(f"SP_ID={SP_ID}, targets: {targets}, dry_run={args.dry_run}, "
          f"backup={args.backup}, workers={args.workers}")

    results = []
    for ds in targets:
        results.append(fix_dataset(ds, dry_run=args.dry_run, backup=args.backup,
                                   workers=args.workers))

    print("-" * 60)
    total_changed = sum(r.get("changed", 0) for r in results)
    total_sp_fixed = sum(r.get("sp_positions_fixed", 0) for r in results)
    print(f"TOTAL: {total_changed} tensors modified, "
          f"{total_sp_fixed} SP-positions zeroed")


if __name__ == "__main__":
    main()
