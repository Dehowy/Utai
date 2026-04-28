"""Per-phoneme contamination audit across datasets.

For each phoneme id (3-79), computes the HuBERT centroid per dataset,
then the pairwise cross-dataset cosine similarity. Low cross-dataset agreement
= candidate for phoneme-level contamination.

Also audits silence tokens (SP=3, AP=4, br=5) specifically — checks if their
GT HuBERT is actually silence-like (low magnitude) per dataset.

Usage: venv/Scripts/python.exe scripts/diag_phoneme_contamination.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    'ace_opencpop': 'data/processed/ace_opencpop',
    'utau': 'data/processed/utau',
    'kiritan': 'data/processed/kiritan',
    'itako': 'data/processed/itako',
    'oniku': 'data/processed/oniku',
    'ofuton': 'data/processed/ofuton',
    'natsume': 'data/processed/natsume',
    'jvs_music': 'data/processed/jvs_music',
}

MAX_FILES = 5000
N_PHONES = 80
DIM = 768
MIN_FRAMES = 30

# Special tokens for silence audit (from phoneme_v1.py)
# 0=<pad>, 1=<sos>, 2=<eos>, 3=SP, 4=AP, 5=br, 6=<mask>
SILENCE_TOKENS = {3: 'SP', 4: 'AP', 5: 'br'}


def cos_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def collect_per_phoneme(ds_path, max_files=MAX_FILES):
    """Per phoneme: sum of hubert, count of frames, + per phoneme sum of ||hubert||^2 for magnitude."""
    hub_dir = Path(ds_path) / 'hubert'
    ph_dir = Path(ds_path) / 'phonemes'
    dur_dir = Path(ds_path) / 'duration'

    if not hub_dir.exists():
        return None

    files = [f.replace('.pt', '') for f in os.listdir(hub_dir) if f.endswith('.pt')]
    if len(files) > max_files:
        rng = np.random.RandomState(42)
        files = list(rng.choice(files, max_files, replace=False))

    acc_sum = np.zeros((N_PHONES, DIM), dtype=np.float64)
    acc_n = np.zeros(N_PHONES, dtype=np.int64)
    acc_mag_sum = np.zeros(N_PHONES, dtype=np.float64)  # sum of ||hubert_t||^2

    for stem in files:
        try:
            hub = torch.load(f'{hub_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.float64)
            phs = torch.load(f'{ph_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
            durs = torch.load(f'{dur_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
        except Exception:
            continue

        T = hub.shape[0]
        ph_exp = np.repeat(phs, durs)
        L = min(T, len(ph_exp))
        ph_exp = ph_exp[:L]
        hub_clip = hub[:L]

        # Accumulate per phoneme
        np.add.at(acc_n, ph_exp, 1)
        np.add.at(acc_sum, ph_exp, hub_clip)
        mag_sq = (hub_clip ** 2).sum(axis=1)  # [L]
        np.add.at(acc_mag_sum, ph_exp, mag_sq)

    centroids = np.full((N_PHONES, DIM), np.nan, dtype=np.float64)
    valid = acc_n >= MIN_FRAMES
    if valid.any():
        centroids[valid] = acc_sum[valid] / acc_n[valid][:, None]

    mean_mag = np.zeros(N_PHONES, dtype=np.float64)
    mean_mag[valid] = np.sqrt(acc_mag_sum[valid] / acc_n[valid])

    return {
        'centroids': centroids,
        'n_frames': acc_n,
        'mean_mag': mean_mag,
        'valid': valid,
        'n_files': len(files),
    }


def load_phoneme_names():
    """Build id→IPA name from default PhonemeVocab."""
    sys.path.insert(0, str(REPO_ROOT))
    from src.data.phoneme_v1 import PhonemeVocab
    v = PhonemeVocab()
    return v.id2phone


def main():
    id2phone = load_phoneme_names()
    print("=" * 100)
    print("Per-phoneme cross-dataset contamination audit")
    print("=" * 100)

    all_stats = {}
    for name, path in DATASETS.items():
        full = REPO_ROOT / path
        if not full.exists():
            continue
        t0 = time.time()
        s = collect_per_phoneme(str(full))
        if s is None:
            continue
        all_stats[name] = s
        print(f"  {name:15s} files={s['n_files']:>5d} "
              f"valid_phones={s['valid'].sum():>3d} "
              f"[{time.time()-t0:.0f}s]")

    if len(all_stats) < 2:
        print("Need at least 2 datasets")
        return

    ds_names = list(all_stats.keys())

    # ───────────────────────────────────────────────────────────
    # Part 1: Per-phoneme cross-dataset agreement
    # ───────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("[Part 1] Per-phoneme cross-dataset centroid agreement")
    print("=" * 100)
    print()
    print("For each phoneme, computes MIN and MEAN cos across all dataset pairs that have it.")
    print("Low MIN = at least one dataset disagrees. High MIN = all datasets agree.")
    print()

    phone_agreement = []  # list of (phoneme_id, min_cos, mean_cos, n_pairs, total_frames, datasets_with_phone)

    for ph_id in range(N_PHONES):
        # For which datasets does this phoneme have enough data?
        valid_ds = [n for n in ds_names if all_stats[n]['valid'][ph_id]]
        if len(valid_ds) < 2:
            continue

        # Compute all pair cos
        pairs = []
        for i in range(len(valid_ds)):
            for j in range(i + 1, len(valid_ds)):
                d1 = valid_ds[i]
                d2 = valid_ds[j]
                c = cos_sim(
                    all_stats[d1]['centroids'][ph_id],
                    all_stats[d2]['centroids'][ph_id]
                )
                pairs.append(((d1, d2), c))

        cos_vals = [p[1] for p in pairs]
        worst_pair = min(pairs, key=lambda x: x[1])
        min_cos = min(cos_vals)
        mean_cos = np.mean(cos_vals)
        total_frames = sum(all_stats[d]['n_frames'][ph_id] for d in valid_ds)

        phone_agreement.append({
            'id': ph_id,
            'name': id2phone.get(ph_id, f'<id={ph_id}>'),
            'min_cos': min_cos,
            'mean_cos': mean_cos,
            'n_datasets': len(valid_ds),
            'n_pairs': len(pairs),
            'total_frames': int(total_frames),
            'worst_pair': worst_pair[0],
            'worst_cos': worst_pair[1],
        })

    # Sort by worst (min_cos) ascending — worst disagreement first
    phone_agreement.sort(key=lambda x: x['min_cos'])

    print(f"{'ID':>3s} {'Phone':>6s} {'Min':>7s} {'Mean':>7s} {'#DS':>3s} {'#Pairs':>6s} {'Frames':>8s} {'Worst pair':>30s}")
    print("-" * 90)

    # Show all phonemes
    for p in phone_agreement:
        wpair = f"{p['worst_pair'][0][:9]}—{p['worst_pair'][1][:9]}"
        print(f"{p['id']:>3d} {p['name']:>6s} {p['min_cos']:>7.4f} {p['mean_cos']:>7.4f} "
              f"{p['n_datasets']:>3d} {p['n_pairs']:>6d} {p['total_frames']:>8d} {wpair:>30s}")

    # Highlight WORST 10 by min_cos (with at least 4 datasets & 1000 frames)
    print()
    print("WORST 10 by cross-dataset disagreement (≥4 datasets, ≥1000 frames):")
    significant = [p for p in phone_agreement if p['n_datasets'] >= 4 and p['total_frames'] >= 1000]
    for p in significant[:10]:
        wpair = f"{p['worst_pair'][0]} vs {p['worst_pair'][1]}"
        print(f"  /{p['name']}/ (id={p['id']}): min_cos={p['min_cos']:.4f} "
              f"(worst: {wpair}), n_frames={p['total_frames']}")

    # ───────────────────────────────────────────────────────────
    # Part 2: Silence-token audit
    # ───────────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("[Part 2] Silence token audit (SP=3, AP=4, br=5)")
    print("=" * 100)
    print()
    print("Clean silence should have LOW mean ||HuBERT||. High magnitude = vowel/consonant residue.")
    print("Cross-dataset cos of silence centroid should also be high — silence is universal.")
    print()

    for sil_id, sil_name in SILENCE_TOKENS.items():
        print(f"\n[Token {sil_id} = {sil_name!r}]")
        # Per dataset: mean magnitude
        print(f"  {'Dataset':15s} {'n_frames':>9s} {'mean_||H||':>12s}")
        print("  " + "-" * 40)
        for n in ds_names:
            s = all_stats[n]
            nf = s['n_frames'][sil_id]
            mag = s['mean_mag'][sil_id] if s['valid'][sil_id] else float('nan')
            if nf == 0:
                continue
            print(f"  {n:15s} {int(nf):>9d} {mag:>12.3f}")

        # Pairwise cos of silence centroids
        valid_ds = [n for n in ds_names if all_stats[n]['valid'][sil_id]]
        if len(valid_ds) >= 2:
            print(f"  Pairwise centroid cos (across {len(valid_ds)} datasets):")
            pairs = []
            for i in range(len(valid_ds)):
                for j in range(i + 1, len(valid_ds)):
                    d1, d2 = valid_ds[i], valid_ds[j]
                    c = cos_sim(all_stats[d1]['centroids'][sil_id],
                                all_stats[d2]['centroids'][sil_id])
                    pairs.append((d1, d2, c))
            pairs.sort(key=lambda x: x[2])
            for d1, d2, c in pairs[:5]:
                print(f"    {d1:15s} vs {d2:15s}: cos={c:.4f}")
            if len(pairs) > 5:
                print(f"    ... ({len(pairs) - 5} more pairs)")

    # ───────────────────────────────────────────────────────────
    # Save json
    # ───────────────────────────────────────────────────────────
    out = {
        'datasets': ds_names,
        'phone_agreement': phone_agreement,
        'silence': {
            sil_name: {
                n: {
                    'n_frames': int(all_stats[n]['n_frames'][sil_id]),
                    'mean_mag': float(all_stats[n]['mean_mag'][sil_id]) if all_stats[n]['valid'][sil_id] else None,
                }
                for n in ds_names
            }
            for sil_id, sil_name in SILENCE_TOKENS.items()
        },
    }
    out_path = REPO_ROOT / 'docs/phoneme_contamination_audit.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWritten: {out_path}")


if __name__ == '__main__':
    main()
