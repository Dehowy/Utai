"""Per-phoneme-per-pitch conditional variance diagnostic.

Computes within-group HuBERT std for each (phoneme, pitch_midi) bucket,
using vectorized accumulation (no nested Python loops per file).

Usage: python scripts/diag_conditional_variance.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
import os
import time
import yaml

DATASETS = {
    'ace_opencpop': 'data/processed/ace_opencpop',
    'gtsinger_zh': 'data/processed/gtsinger_zh',
    'gtsinger_ja': 'data/processed/gtsinger_ja',
    'gtsinger_en': 'data/processed/gtsinger_en',
}

MAX_FILES = 8000   # cap for ACE (100K)
MIN_FRAMES = 30    # minimum frames per bucket to compute stats
N_PHONES = 80
N_PITCHES = 128
N_BUCKETS = N_PHONES * N_PITCHES  # 10240
DIM = 768

# Load vocab
with open('data/processed/ace_opencpop/vocab.yaml', encoding='utf-8') as f:
    vocab = yaml.safe_load(f)
i2p = {int(v): k for k, v in vocab['phone2id'].items()}


def process_dataset(ds_name, ds_path):
    t0 = time.time()
    hub_dir = f'{ds_path}/hubert'
    ph_dir = f'{ds_path}/phonemes'
    pi_dir = f'{ds_path}/pitch'
    dur_dir = f'{ds_path}/duration'

    files = [f.replace('.pt', '') for f in os.listdir(hub_dir) if f.endswith('.pt')]
    if len(files) > MAX_FILES:
        rng = np.random.RandomState(42)
        files = list(rng.choice(files, MAX_FILES, replace=False))

    # Accumulate with flat bucket index: bucket = phone_id * 128 + pitch_midi
    acc_sum = np.zeros((N_BUCKETS, DIM), dtype=np.float64)
    acc_sq = np.zeros((N_BUCKETS, DIM), dtype=np.float64)
    acc_n = np.zeros(N_BUCKETS, dtype=np.int64)

    n_frames_total = 0
    n_skipped = 0

    for fi, stem in enumerate(files):
        if fi % 1000 == 0 and fi > 0:
            elapsed = time.time() - t0
            print(f'  {ds_name}: {fi}/{len(files)} files, {elapsed:.0f}s', flush=True)
        try:
            hub = torch.load(f'{hub_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.float64)
            phs = torch.load(f'{ph_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
            pis = torch.load(f'{pi_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
            durs = torch.load(f'{dur_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
        except Exception:
            n_skipped += 1
            continue

        T_frame = hub.shape[0]
        ph_exp = np.repeat(phs, durs)
        pi_exp = np.repeat(pis, durs)
        L = min(T_frame, len(ph_exp))
        ph_exp = ph_exp[:L]
        pi_exp = pi_exp[:L]
        hub_clip = hub[:L]

        # Filter: skip special tokens (0-6) and pitch=0
        mask = (ph_exp > 6) & (pi_exp > 0) & (pi_exp < 128)
        ph_exp = ph_exp[mask]
        pi_exp = pi_exp[mask]
        hub_clip = hub_clip[mask]

        if len(ph_exp) == 0:
            continue

        n_frames_total += len(ph_exp)

        # Vectorized bucket accumulation
        bucket = ph_exp * 128 + pi_exp  # [L_filtered]

        # np.add.at for scatter accumulation
        np.add.at(acc_n, bucket, 1)
        np.add.at(acc_sum, bucket, hub_clip)
        np.add.at(acc_sq, bucket, hub_clip ** 2)

    elapsed = time.time() - t0

    # Compute within-group std for buckets with enough frames
    valid = acc_n >= MIN_FRAMES
    n_valid = valid.sum()

    group_stds = np.full(N_BUCKETS, np.nan)
    if n_valid > 0:
        n_safe = acc_n[valid].astype(np.float64)
        mean = acc_sum[valid] / n_safe[:, None]
        var = acc_sq[valid] / n_safe[:, None] - mean ** 2
        var = np.maximum(var, 0)
        std = np.sqrt(var)
        avg_std_per_group = std.mean(axis=1)  # avg across 768 dims
        group_stds[valid] = avg_std_per_group

    valid_stds = group_stds[~np.isnan(group_stds)]

    print(f'\n{ds_name:20s}  files={len(files):>5d}(skipped {n_skipped})  '
          f'frames={n_frames_total:>8d}  buckets(n>={MIN_FRAMES})={n_valid:>4d}  '
          f'[{elapsed:.1f}s]', flush=True)

    if len(valid_stds) > 0:
        print(f'  within-group std:  mean={valid_stds.mean():.4f}  '
              f'median={np.median(valid_stds):.4f}  '
              f'p10={np.percentile(valid_stds,10):.4f}  '
              f'p90={np.percentile(valid_stds,90):.4f}  '
              f'max={valid_stds.max():.4f}', flush=True)

    return acc_sum, acc_sq, acc_n, group_stds


# Run all datasets
all_results = {}
for ds_name, ds_path in DATASETS.items():
    all_results[ds_name] = process_dataset(ds_name, ds_path)

# ---- Vowel breakdown ----
vowels = {17: 'a', 7: 'i', 24: 'u', 21: 'o', 10: 'e', 14: 'ə', 12: 'æ'}
pitch_names = {60: 'C4', 62: 'D4', 64: 'E4', 65: 'F4', 67: 'G4', 69: 'A4', 71: 'B4', 72: 'C5'}

print('\n' + '=' * 110)
print('Per-vowel conditional std (common pitch range G3-C5, MIDI 55-72)')
print('=' * 110)

for ph_id, ph_name in vowels.items():
    print(f'\n--- /{ph_name}/ (id={ph_id}) ---')
    for ds_name in DATASETS:
        _, _, acc_n, group_stds = all_results[ds_name]
        stds, ns = [], []
        for pi in range(55, 73):
            bucket = ph_id * 128 + pi
            if acc_n[bucket] >= MIN_FRAMES:
                stds.append(group_stds[bucket])
                ns.append(acc_n[bucket])
        if stds:
            arr = np.array(stds)
            total_n = sum(ns)
            # weighted average by frame count
            wavg = np.average(arr, weights=ns)
            print(f'  {ds_name:20s}  pitches={len(stds):>2d}  frames={total_n:>6d}  '
                  f'avg_std={wavg:.4f}  range=[{arr.min():.4f}, {arr.max():.4f}]')
        else:
            print(f'  {ds_name:20s}  (insufficient data)')

# ---- Ratio table: each dataset's conditional std vs ACE ----
print('\n' + '=' * 110)
print('Conditional std RATIO vs ACE-Opencpop (higher = more within-group spread = worse for det_head)')
print('=' * 110)

ace_sum, ace_sq, ace_n, ace_stds = all_results['ace_opencpop']

for ds_name in ['gtsinger_zh', 'gtsinger_ja', 'gtsinger_en']:
    _, _, ds_n, ds_stds = all_results[ds_name]
    # find common buckets with enough data in BOTH
    both_valid = (ace_n >= MIN_FRAMES) & (ds_n >= MIN_FRAMES) & (~np.isnan(ace_stds)) & (~np.isnan(ds_stds))
    n_common = both_valid.sum()
    if n_common > 0:
        ratio = ds_stds[both_valid] / ace_stds[both_valid]
        print(f'\n  {ds_name} vs ace_opencpop:  common_buckets={n_common}')
        print(f'    std ratio:  mean={ratio.mean():.3f}  median={np.median(ratio):.3f}  '
              f'p10={np.percentile(ratio,10):.3f}  p90={np.percentile(ratio,90):.3f}')

        # breakdown by vowels
        for ph_id, ph_name in {17: 'a', 7: 'i', 24: 'u', 21: 'o', 10: 'e'}.items():
            vowel_ratios = []
            for pi in range(55, 73):
                bucket = ph_id * 128 + pi
                if ace_n[bucket] >= MIN_FRAMES and ds_n[bucket] >= MIN_FRAMES:
                    vowel_ratios.append(ds_stds[bucket] / ace_stds[bucket])
            if vowel_ratios:
                vr = np.array(vowel_ratios)
                print(f'      /{ph_name}/: ratio mean={vr.mean():.3f} [{vr.min():.3f}-{vr.max():.3f}] (n={len(vowel_ratios)} pitches)')

print('\n--- Interpretation ---')
print('  ratio ~1.0: same conditional spread as ACE -> safe for det_head')
print('  ratio 1.0-1.3: mild increase, probably OK')
print('  ratio 1.3-1.5: concerning, det_head will average more')
print('  ratio >1.5: significant blurring risk, probably should not mix')
print('\nDone.', flush=True)
