"""Pitch error diagnostic: RMVPE F0 vs MIDI annotation (in cents).

For each dataset, computes cent deviation = 1200 * log2(f0_rmvpe / f0_midi)
on voiced frames with non-zero pitch labels.

Excludes: gtsinger_zh (dropped from training set)

Usage: python scripts/diag_pitch_error.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
import os
import time

DATASETS = {
    'ace_opencpop': 'data/processed/ace_opencpop',
    'gtsinger_ja': 'data/processed/gtsinger_ja',
    'gtsinger_en': 'data/processed/gtsinger_en',
}

MAX_FILES = 8000


def midi_to_hz(midi):
    """MIDI note number to Hz."""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def process_dataset(ds_name, ds_path):
    t0 = time.time()
    f0_dir = f'{ds_path}/f0'
    pi_dir = f'{ds_path}/pitch'
    dur_dir = f'{ds_path}/duration'

    files = [f.replace('.pt', '') for f in os.listdir(f0_dir) if f.endswith('.pt')]
    if len(files) > MAX_FILES:
        rng = np.random.RandomState(42)
        files = list(rng.choice(files, MAX_FILES, replace=False))

    all_cents = []
    all_abs_cents = []
    n_voiced = 0
    n_total_frames = 0
    n_files = 0
    n_large_error = 0  # >100 cents
    n_very_large = 0   # >200 cents

    for fi, stem in enumerate(files):
        if fi % 2000 == 0 and fi > 0:
            print(f'  {ds_name}: {fi}/{len(files)}', flush=True)
        try:
            f0 = torch.load(f'{f0_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.float64)
            pis = torch.load(f'{pi_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
            durs = torch.load(f'{dur_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
        except Exception:
            continue

        n_files += 1
        T = len(f0)

        # Expand pitch to frame level
        pi_exp = np.repeat(pis, durs)[:T]
        if len(pi_exp) < T:
            pi_exp = np.pad(pi_exp, (0, T - len(pi_exp)))

        n_total_frames += T

        # Voiced + non-rest mask
        mask = (f0 > 30.0) & (pi_exp > 0)  # f0>30Hz to filter noise
        n_voiced += mask.sum()

        if mask.sum() == 0:
            continue

        f0_v = f0[mask]
        pi_v = pi_exp[mask].astype(np.float64)
        f0_midi = midi_to_hz(pi_v)

        # Cents deviation
        cents = 1200.0 * np.log2(f0_v / f0_midi)
        abs_cents = np.abs(cents)

        all_cents.append(cents)
        all_abs_cents.append(abs_cents)
        n_large_error += (abs_cents > 100).sum()
        n_very_large += (abs_cents > 200).sum()

    elapsed = time.time() - t0
    all_cents = np.concatenate(all_cents) if all_cents else np.array([])
    all_abs_cents = np.concatenate(all_abs_cents) if all_abs_cents else np.array([])

    print(f'\n{ds_name}', flush=True)
    print(f'  files={n_files}  total_frames={n_total_frames}  voiced_frames={n_voiced}  [{elapsed:.0f}s]')

    if len(all_cents) > 0:
        print(f'  Signed error (cents):')
        print(f'    mean={all_cents.mean():.1f}  median={np.median(all_cents):.1f}  '
              f'std={all_cents.std():.1f}')
        print(f'    p5={np.percentile(all_cents,5):.1f}  p25={np.percentile(all_cents,25):.1f}  '
              f'p75={np.percentile(all_cents,75):.1f}  p95={np.percentile(all_cents,95):.1f}')

        print(f'  Absolute error (cents):')
        print(f'    mean={all_abs_cents.mean():.1f}  median={np.median(all_abs_cents):.1f}  '
              f'std={all_abs_cents.std():.1f}')
        print(f'    p50={np.percentile(all_abs_cents,50):.1f}  p75={np.percentile(all_abs_cents,75):.1f}  '
              f'p90={np.percentile(all_abs_cents,90):.1f}  p95={np.percentile(all_abs_cents,95):.1f}  '
              f'p99={np.percentile(all_abs_cents,99):.1f}')

        pct_100 = 100.0 * n_large_error / len(all_abs_cents)
        pct_200 = 100.0 * n_very_large / len(all_abs_cents)
        print(f'  |error| > 100c: {n_large_error} ({pct_100:.1f}%)')
        print(f'  |error| > 200c: {n_very_large} ({pct_200:.1f}%)')

        # Histogram bins
        bins = [0, 10, 25, 50, 100, 200, 500, 1000, 99999]
        labels = ['0-10c', '10-25c', '25-50c', '50-100c', '100-200c', '200-500c', '500-1000c', '>1000c']
        hist, _ = np.histogram(all_abs_cents, bins=bins)
        print(f'  Distribution:')
        for label, count in zip(labels, hist):
            pct = 100.0 * count / len(all_abs_cents)
            bar = '#' * int(pct / 2)
            print(f'    {label:>10s}: {count:>8d} ({pct:>5.1f}%) {bar}')

    return all_cents, all_abs_cents


# Run
print('Pitch error diagnostic: RMVPE F0 vs MIDI annotation\n', flush=True)
results = {}
for ds_name, ds_path in DATASETS.items():
    results[ds_name] = process_dataset(ds_name, ds_path)

# Comparison summary
print('\n' + '=' * 80)
print('SUMMARY — absolute pitch error (cents)')
print('=' * 80)
print(f'{"Dataset":20s} {"Mean":>8s} {"Median":>8s} {"P90":>8s} {"P95":>8s} {">100c":>8s} {">200c":>8s}')
print('-' * 80)
for ds_name in DATASETS:
    _, abs_c = results[ds_name]
    if len(abs_c) > 0:
        pct100 = 100.0 * (abs_c > 100).sum() / len(abs_c)
        pct200 = 100.0 * (abs_c > 200).sum() / len(abs_c)
        print(f'{ds_name:20s} {abs_c.mean():>7.1f}c {np.median(abs_c):>7.1f}c '
              f'{np.percentile(abs_c,90):>7.1f}c {np.percentile(abs_c,95):>7.1f}c '
              f'{pct100:>7.1f}% {pct200:>7.1f}%')

print('\nInterpretation:')
print('  median < 25c: tight (synthetic or well-tuned)')
print('  median 25-50c: normal human singing')
print('  median 50-100c: loose singing or annotation offset')
print('  median > 100c: annotation problem (wrong pitch labels or bad alignment)')
print('\nDone.', flush=True)
