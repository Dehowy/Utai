"""Comprehensive cross-dataset diagnostic for ALL JA datasets vs ACE.

For each dataset (UTAU, Kiritan, Itako, ONIKU, OFUTON, Natsume, JVS-MuSiC):
  1. Per-phoneme centroid HuBERT cos vs ACE-zh (key indicator: GTSinger-style contamination)
  2. Pairwise centroid cos BETWEEN JA datasets (internal consistency)
  3. Per-phoneme conditional std (within-bucket variance)
  4. Pitch error (RMVPE F0 vs MIDI annotation)
  5. Phoneme distribution

Output: docs/dataset_audit_report.md
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import os
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Datasets to audit
DATASETS = {
    'ace_opencpop': 'data/processed/ace_opencpop',     # zh baseline
    'utau': 'data/processed/utau',                     # ja
    'kiritan': 'data/processed/kiritan',               # ja
    'itako': 'data/processed/itako',                   # ja
    'oniku': 'data/processed/oniku',                   # ja
    'ofuton': 'data/processed/ofuton',                 # ja
    'natsume': 'data/processed/natsume',               # ja
    'jvs_music': 'data/processed/jvs_music',           # ja (multi-singer)
}

MAX_FILES = 5000      # cap per dataset for accumulation speed
N_PHONES = 80
N_PITCHES = 128
N_BUCKETS = N_PHONES * N_PITCHES
DIM = 768
MIN_FRAMES = 30       # min frames per (phoneme, pitch) bucket to compute centroid

# Vowels for focused per-phoneme reporting
VOWELS = {17: 'a', 7: 'i', 24: 'u', 21: 'o', 10: 'e', 11: 'ɛ', 14: 'ə', 25: 'ɯ'}
# Common consonants
CONSONANTS = {32: 'k', 49: 's', 63: 'n', 67: 'l', 62: 'm', 30: 'tʰ', 65: 'ŋ', 70: 'j'}

# Pitch range we focus on (typical singing)
PITCH_RANGE = list(range(50, 80))  # D3-G5


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def cosine_sim(a, b):
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def collect_per_bucket_stats(ds_path, max_files=MAX_FILES):
    """Walk one dataset, return (acc_sum, acc_sq, acc_n, pitch_errors, phon_count)."""
    hub_dir = Path(ds_path) / 'hubert'
    ph_dir = Path(ds_path) / 'phonemes'
    pi_dir = Path(ds_path) / 'pitch'
    dur_dir = Path(ds_path) / 'duration'
    f0_dir = Path(ds_path) / 'f0'

    if not hub_dir.exists():
        return None

    files = [f.replace('.pt', '') for f in os.listdir(hub_dir) if f.endswith('.pt')]
    if len(files) > max_files:
        rng = np.random.RandomState(42)
        files = list(rng.choice(files, max_files, replace=False))

    acc_sum = np.zeros((N_BUCKETS, DIM), dtype=np.float64)
    acc_sq = np.zeros((N_BUCKETS, DIM), dtype=np.float64)
    acc_n = np.zeros(N_BUCKETS, dtype=np.int64)
    phon_count = Counter()
    pitch_errors_cents = []
    n_total_frames = 0

    for fi, stem in enumerate(files):
        try:
            hub = torch.load(f'{hub_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.float64)
            phs = torch.load(f'{ph_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
            pis = torch.load(f'{pi_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
            durs = torch.load(f'{dur_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
            f0 = torch.load(f'{f0_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.float64)
        except Exception:
            continue

        T = hub.shape[0]
        ph_exp = np.repeat(phs, durs)
        pi_exp = np.repeat(pis, durs)
        L = min(T, len(ph_exp), len(f0))
        ph_exp = ph_exp[:L]
        pi_exp = pi_exp[:L]
        hub_clip = hub[:L]
        f0_clip = f0[:L]

        # Pitch error per voiced+pitch-labeled frame
        voiced_mask = (f0_clip > 30.0) & (pi_exp > 0)
        if voiced_mask.any():
            f0v = f0_clip[voiced_mask]
            pv = pi_exp[voiced_mask].astype(np.float64)
            f0_midi = midi_to_hz(pv)
            cents = 1200.0 * np.log2(f0v / f0_midi)
            pitch_errors_cents.append(cents)

        # Centroid bucketing: skip silence/special tokens (0-6)
        mask = (ph_exp > 6) & (pi_exp > 0) & (pi_exp < 128)
        ph_exp = ph_exp[mask]
        pi_exp = pi_exp[mask]
        hub_clip = hub_clip[mask]

        if len(ph_exp) == 0:
            continue
        n_total_frames += len(ph_exp)

        bucket = ph_exp * 128 + pi_exp
        np.add.at(acc_n, bucket, 1)
        np.add.at(acc_sum, bucket, hub_clip)
        np.add.at(acc_sq, bucket, hub_clip ** 2)

        for p in ph_exp:
            phon_count[int(p)] += 1

    pitch_errors_cents = np.concatenate(pitch_errors_cents) if pitch_errors_cents else np.array([])
    return {
        'acc_sum': acc_sum,
        'acc_sq': acc_sq,
        'acc_n': acc_n,
        'phon_count': phon_count,
        'pitch_errors': pitch_errors_cents,
        'n_files': len(files),
        'n_frames': n_total_frames,
    }


def per_bucket_centroid(stats):
    """Return per-bucket centroid array [N_BUCKETS, DIM] (NaN where insufficient)."""
    centroids = np.full((N_BUCKETS, DIM), np.nan, dtype=np.float64)
    valid = stats['acc_n'] >= MIN_FRAMES
    if valid.any():
        centroids[valid] = stats['acc_sum'][valid] / stats['acc_n'][valid][:, None]
    return centroids


def cos_per_bucket(c1, c2, n1, n2, min_frames=MIN_FRAMES):
    """For each bucket valid in both, compute cos. Returns dict of bucket→cos."""
    result = {}
    for b in range(N_BUCKETS):
        if n1[b] < min_frames or n2[b] < min_frames:
            continue
        if np.isnan(c1[b, 0]) or np.isnan(c2[b, 0]):
            continue
        result[b] = cosine_sim(c1[b], c2[b])
    return result


def summarize_phoneme_cos(cos_dict, phone_set):
    """For phones in phone_set, average cos across the pitch range we care about."""
    out = {}
    for ph_id, ph_name in phone_set.items():
        vals = []
        for pi in PITCH_RANGE:
            b = ph_id * 128 + pi
            if b in cos_dict:
                vals.append(cos_dict[b])
        if vals:
            out[ph_name] = (np.mean(vals), len(vals))
    return out


def main():
    print("=" * 80)
    print("Comprehensive cross-dataset audit (Score2HuBERT v2)")
    print("=" * 80)

    # --- 1. Collect stats per dataset ---
    print("\n[Step 1] Collecting per-bucket HuBERT stats + pitch errors...")
    all_stats = {}
    for name, path in DATASETS.items():
        full_path = REPO_ROOT / path
        if not full_path.exists():
            print(f"  {name}: SKIP (not found)")
            continue
        t0 = time.time()
        s = collect_per_bucket_stats(str(full_path))
        if s is None:
            continue
        elapsed = time.time() - t0
        all_stats[name] = s
        valid_buckets = (s['acc_n'] >= MIN_FRAMES).sum()
        n_phon = len(s['phon_count'])
        print(f"  {name:15s}  files={s['n_files']:>5d}  frames={s['n_frames']:>9d}  "
              f"valid_buckets={valid_buckets:>4d}  unique_phones={n_phon:>3d}  [{elapsed:.0f}s]")

    if 'ace_opencpop' not in all_stats:
        print("ERROR: ACE not found, cannot do baseline comparison")
        return

    # --- 2. Centroid alignment vs ACE ---
    print("\n" + "=" * 80)
    print("[Step 2] Per-phoneme centroid cos vs ACE (smoking gun for GT-style contamination)")
    print("=" * 80)
    ace = all_stats['ace_opencpop']
    ace_centroids = per_bucket_centroid(ace)

    print(f"\n{'Dataset':15s}  {'AllBuckets':>11s}  {'Vowels':>9s}  {'/a/':>7s}  {'/i/':>7s}  {'/u/':>7s}  {'/o/':>7s}  {'/e/':>7s}  {'/n/':>7s}  {'/k/':>7s}")
    print('-' * 110)

    summary = []
    for name, s in all_stats.items():
        if name == 'ace_opencpop':
            continue
        cents = per_bucket_centroid(s)
        cos = cos_per_bucket(ace_centroids, cents, ace['acc_n'], s['acc_n'])
        if not cos:
            print(f"{name:15s}  no common buckets")
            continue
        all_vals = list(cos.values())
        vowel_cos = summarize_phoneme_cos(cos, VOWELS)
        cons_cos = summarize_phoneme_cos(cos, CONSONANTS)
        all_specific = {**vowel_cos, **cons_cos}

        all_mean = np.mean(all_vals)
        vowels_mean = np.mean([v[0] for v in vowel_cos.values()]) if vowel_cos else float('nan')

        line = f"{name:15s}  {all_mean:>11.4f}  {vowels_mean:>9.4f}"
        for tag in ['a', 'i', 'u', 'o', 'e', 'n', 'k']:
            if tag in all_specific:
                line += f"  {all_specific[tag][0]:>7.4f}"
            else:
                line += f"  {'-':>7s}"
        print(line)
        summary.append({
            'name': name, 'all_mean': all_mean, 'vowels_mean': vowels_mean,
            'specific': all_specific, 'n_buckets': len(cos),
        })

    print(f"\nReference: v1 cross-language same-phoneme cos = 0.876 (baseline)")
    print(f"Below baseline (esp. for vowels) = potential contamination warning")

    # --- 3. Pairwise centroid cos between JA datasets ---
    print("\n" + "=" * 80)
    print("[Step 3] Pairwise centroid cos BETWEEN JA datasets (vowels only)")
    print("=" * 80)
    ja_names = [n for n in all_stats if n != 'ace_opencpop']
    print(f"\n{'':15s}  " + "  ".join(f"{n[:9]:>9s}" for n in ja_names))
    for n1 in ja_names:
        c1 = per_bucket_centroid(all_stats[n1])
        n1_cnt = all_stats[n1]['acc_n']
        row = f"{n1:15s}  "
        for n2 in ja_names:
            if n1 == n2:
                row += f"  {'---':>9s}"
                continue
            c2 = per_bucket_centroid(all_stats[n2])
            n2_cnt = all_stats[n2]['acc_n']
            cos = cos_per_bucket(c1, c2, n1_cnt, n2_cnt)
            vowel_cos = summarize_phoneme_cos(cos, VOWELS)
            if vowel_cos:
                v = np.mean([x[0] for x in vowel_cos.values()])
                row += f"  {v:>9.4f}"
            else:
                row += f"  {'-':>9s}"
        print(row)

    # --- 4. Pitch error stats ---
    print("\n" + "=" * 80)
    print("[Step 4] Pitch error: RMVPE F0 vs MIDI annotation (cents)")
    print("=" * 80)
    print(f"{'Dataset':15s}  {'N_voiced':>9s}  {'Median':>7s}  {'Mean':>7s}  {'P90':>7s}  {'>100c':>7s}  {'>200c':>7s}")
    print('-' * 75)
    for name, s in all_stats.items():
        pe = s['pitch_errors']
        if len(pe) == 0:
            print(f"{name:15s}  no voiced frames")
            continue
        ape = np.abs(pe)
        print(f"{name:15s}  {len(pe):>9d}  {np.median(ape):>6.1f}c  {ape.mean():>6.1f}c  "
              f"{np.percentile(ape, 90):>6.1f}c  {100*(ape>100).sum()/len(ape):>6.1f}%  "
              f"{100*(ape>200).sum()/len(ape):>6.1f}%")

    # --- 5. Phoneme coverage ---
    print("\n" + "=" * 80)
    print("[Step 5] Phoneme coverage (number of unique phoneme IDs with > 100 frames)")
    print("=" * 80)
    print(f"{'Dataset':15s}  {'Total Frames':>14s}  {'Unique Phones':>14s}  {'Phones >100f':>14s}")
    print('-' * 65)
    for name, s in all_stats.items():
        n_above = sum(1 for c in s['phon_count'].values() if c >= 100)
        print(f"{name:15s}  {s['n_frames']:>14d}  {len(s['phon_count']):>14d}  {n_above:>14d}")

    # Save summary as JSON for downstream use
    out_path = REPO_ROOT / 'docs' / 'dataset_audit_report.json'
    out_path.parent.mkdir(exist_ok=True)
    save_summary = {}
    for s in summary:
        save_summary[s['name']] = {
            'all_mean_cos_vs_ace': float(s['all_mean']),
            'vowels_mean_cos_vs_ace': float(s['vowels_mean']),
            'n_buckets': int(s['n_buckets']),
            'per_phoneme_cos': {k: [float(v[0]), int(v[1])] for k, v in s['specific'].items()},
        }
    save_summary['_meta'] = {
        'reference_v1_cross_lang_baseline': 0.876,
        'min_frames_per_bucket': MIN_FRAMES,
        'max_files_per_dataset': MAX_FILES,
        'pitch_range': [PITCH_RANGE[0], PITCH_RANGE[-1]],
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_summary, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Detailed numbers saved to {out_path}")


if __name__ == '__main__':
    main()
