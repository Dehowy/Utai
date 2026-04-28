"""Cross-dataset phoneme centroid alignment diagnostic.

For each (phoneme, pitch) bucket, computes the HuBERT centroid (mean vector)
per dataset, then measures cosine similarity between ACE and GTSinger centroids.

If phoneme labels are consistent: centroid cos should be high (>0.95)
If phoneme labels are inconsistent: centroid cos will drop significantly

Usage: python scripts/diag_centroid_alignment.py
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

MAX_FILES = 8000
MIN_FRAMES = 50  # stricter — need stable centroids
N_PHONES = 80
N_PITCHES = 128
N_BUCKETS = N_PHONES * N_PITCHES
DIM = 768

with open('data/processed/ace_opencpop/vocab.yaml', encoding='utf-8') as f:
    vocab = yaml.safe_load(f)
i2p = {int(v): k for k, v in vocab['phone2id'].items()}
p2i = vocab['phone2id']


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

    acc_sum = np.zeros((N_BUCKETS, DIM), dtype=np.float64)
    acc_n = np.zeros(N_BUCKETS, dtype=np.int64)

    for fi, stem in enumerate(files):
        if fi % 2000 == 0 and fi > 0:
            print(f'  {ds_name}: {fi}/{len(files)}', flush=True)
        try:
            hub = torch.load(f'{hub_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.float64)
            phs = torch.load(f'{ph_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
            pis = torch.load(f'{pi_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
            durs = torch.load(f'{dur_dir}/{stem}.pt', map_location='cpu', weights_only=True).numpy().astype(np.int32)
        except Exception:
            continue

        T_frame = hub.shape[0]
        ph_exp = np.repeat(phs, durs)
        pi_exp = np.repeat(pis, durs)
        L = min(T_frame, len(ph_exp))

        mask = (ph_exp[:L] > 6) & (pi_exp[:L] > 0) & (pi_exp[:L] < 128)
        bucket = ph_exp[:L][mask] * 128 + pi_exp[:L][mask]
        hub_m = hub[:L][mask]

        np.add.at(acc_n, bucket, 1)
        np.add.at(acc_sum, bucket, hub_m)

    elapsed = time.time() - t0
    print(f'  {ds_name}: done, {elapsed:.0f}s', flush=True)
    return acc_sum, acc_n


def cosine_sim(a, b):
    """Cosine similarity between two 1D vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


# ---- Collect ----
print("Loading datasets...\n", flush=True)
all_data = {}
for ds_name, ds_path in DATASETS.items():
    all_data[ds_name] = process_dataset(ds_name, ds_path)

ace_sum, ace_n = all_data['ace_opencpop']

# ---- Cross-dataset centroid cosine similarity ----
print('\n' + '=' * 110)
print('CROSS-DATASET CENTROID COSINE SIMILARITY')
print('For each (phoneme, pitch) bucket with enough data in BOTH datasets,')
print('compute cos(ACE_centroid, Other_centroid). Low cos = label inconsistency.')
print('=' * 110)

# Focus phonemes: vowels + common consonants
focus_phones = {
    17: 'a', 7: 'i', 24: 'u', 21: 'o', 10: 'e', 14: 'ə',
    63: 'n', 49: 's', 32: 'k', 67: 'l', 62: 'm', 61: 'h',
    55: 'ʃ', 69: 'ɹ', 28: 'b', 31: 'd', 34: 'g',
}

pitch_range = range(55, 73)  # G3 to C5

for cmp_name in ['gtsinger_zh', 'gtsinger_ja', 'gtsinger_en']:
    cmp_sum, cmp_n = all_data[cmp_name]

    print(f'\n{"=" * 80}')
    print(f'  {cmp_name} vs ace_opencpop')
    print(f'{"=" * 80}')

    # Global: all common buckets
    both_valid = (ace_n >= MIN_FRAMES) & (cmp_n >= MIN_FRAMES)
    n_common = both_valid.sum()

    if n_common == 0:
        print('  No common buckets!')
        continue

    # Compute centroid cos for all common buckets
    all_cos = []
    for bucket_idx in np.where(both_valid)[0]:
        ace_centroid = ace_sum[bucket_idx] / ace_n[bucket_idx]
        cmp_centroid = cmp_sum[bucket_idx] / cmp_n[bucket_idx]
        c = cosine_sim(ace_centroid, cmp_centroid)
        all_cos.append(c)

    all_cos = np.array(all_cos)
    print(f'  ALL common buckets (n={n_common}):')
    print(f'    centroid cos:  mean={all_cos.mean():.4f}  median={np.median(all_cos):.4f}  '
          f'p5={np.percentile(all_cos,5):.4f}  p10={np.percentile(all_cos,10):.4f}  '
          f'min={all_cos.min():.4f}')

    # Per-phoneme breakdown for focus phones
    print(f'\n  Per-phoneme centroid cos (pitch range G3-C5):')
    print(f'  {"phone":8s} {"pitches":>7s} {"mean_cos":>9s} {"min_cos":>9s} {"max_cos":>9s} {"ACE_n":>7s} {"CMP_n":>7s}')
    print(f'  {"-"*60}')

    for ph_id in sorted(focus_phones.keys()):
        ph_name = focus_phones[ph_id]
        cos_vals = []
        ace_total = 0
        cmp_total = 0
        for pi in pitch_range:
            bucket = ph_id * 128 + pi
            if ace_n[bucket] >= MIN_FRAMES and cmp_n[bucket] >= MIN_FRAMES:
                ace_c = ace_sum[bucket] / ace_n[bucket]
                cmp_c = cmp_sum[bucket] / cmp_n[bucket]
                cos_vals.append(cosine_sim(ace_c, cmp_c))
                ace_total += ace_n[bucket]
                cmp_total += cmp_n[bucket]
        if cos_vals:
            arr = np.array(cos_vals)
            flag = '  <<<' if arr.mean() < 0.90 else ('  <' if arr.mean() < 0.95 else '')
            print(f'  /{ph_name:6s}  {len(cos_vals):>7d}  {arr.mean():>9.4f}  {arr.min():>9.4f}  '
                  f'{arr.max():>9.4f}  {ace_total:>7d}  {cmp_total:>7d}{flag}')

    # Also show the WORST buckets overall
    print(f'\n  Bottom 20 buckets by centroid cos:')
    worst_indices = np.argsort(all_cos)[:20]
    valid_buckets = np.where(both_valid)[0]
    print(f'  {"bucket":30s} {"cos":>7s} {"ACE_n":>7s} {"CMP_n":>7s}')
    for wi in worst_indices:
        bucket_idx = valid_buckets[wi]
        ph_id = bucket_idx // 128
        pi_id = bucket_idx % 128
        ph_name = i2p.get(ph_id, f'?{ph_id}')
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note = f'{note_names[pi_id % 12]}{pi_id // 12 - 1}'
        print(f'  /{ph_name:6s} @ {note:4s} (MIDI {pi_id:>3d})  '
              f'{all_cos[wi]:>7.4f}  {ace_n[bucket_idx]:>7d}  {cmp_n[bucket_idx]:>7d}')


# ---- Same-language control: GTSinger ZH vs ACE (both Chinese) ----
print('\n' + '=' * 110)
print('CONTROL: If labels are consistent, same-language centroids should be most similar.')
print('Expected: gtsinger_zh >> gtsinger_en/ja (since zh shares language with ACE)')
print('If gtsinger_zh centroid cos ≈ gtsinger_en centroid cos, then phoneme labels')
print('may be driving alignment more than language identity.')
print('=' * 110)

# Summary comparison
for cmp_name in ['gtsinger_zh', 'gtsinger_ja', 'gtsinger_en']:
    cmp_sum, cmp_n = all_data[cmp_name]
    # just vowels
    vowel_cos = []
    for ph_id in [17, 7, 24, 21, 10]:  # a i u o e
        for pi in pitch_range:
            bucket = ph_id * 128 + pi
            if ace_n[bucket] >= MIN_FRAMES and cmp_n[bucket] >= MIN_FRAMES:
                ace_c = ace_sum[bucket] / ace_n[bucket]
                cmp_c = cmp_sum[bucket] / cmp_n[bucket]
                vowel_cos.append(cosine_sim(ace_c, cmp_c))
    if vowel_cos:
        arr = np.array(vowel_cos)
        print(f'  {cmp_name:20s}  vowel centroids vs ACE:  mean={arr.mean():.4f}  '
              f'median={np.median(arr):.4f}  min={arr.min():.4f}')

print('\nDone.', flush=True)
