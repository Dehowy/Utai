"""4-angle phoneme contamination audit (post-SOFA-aware).

Extends v1 diag_phoneme_contamination.py with language-stratified analyses:
    1. Intra-lang horizontal:   same lang, different datasets, same phoneme (cohesion)
    2. Intra-lang vertical:     same dataset, different phonemes (separability)
    3. Cross-lang horizontal:   different lang, same phoneme (IPA transfer health)
    4. Cross-lang vertical + confusion alerts:
       cases where cross-lang cos(same phoneme) < cross-lang cos(different phoneme)

Plus SP/AP/br silence-token magnitude audit.

Loads pre-SOFA snapshot if `--baseline-json` provided; emits delta table.

Usage:
    py -3.10 scripts/diag_phoneme_contamination_v2.py --tag post_sofa
    py -3.10 scripts/diag_phoneme_contamination_v2.py --tag post_sofa \\
        --baseline-json docs/phoneme_contamination_audit.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding='utf-8')

from src.data.phoneme_v1 import PhonemeVocab  # noqa: E402

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

LANG_OF = {
    'ace_opencpop': 'zh',
    'utau': 'ja',
    'kiritan': 'ja',
    'itako': 'ja',
    'oniku': 'ja',
    'ofuton': 'ja',
    'natsume': 'ja',
    'jvs_music': 'ja',
}

MAX_FILES = 5000
N_PHONES = 80
DIM = 768
MIN_FRAMES = 30

SILENCE_TOKENS = {3: 'SP', 4: 'AP', 5: 'br'}


def cos_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def collect_per_phoneme(ds_path, max_files=MAX_FILES):
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
    acc_sqsum_perdim = np.zeros((N_PHONES, DIM), dtype=np.float64)
    acc_n = np.zeros(N_PHONES, dtype=np.int64)
    acc_mag_sum = np.zeros(N_PHONES, dtype=np.float64)

    for stem in files:
        try:
            hub = torch.load(f'{hub_dir}/{stem}.pt', map_location='cpu',
                             weights_only=True).numpy().astype(np.float64)
            phs = torch.load(f'{ph_dir}/{stem}.pt', map_location='cpu',
                             weights_only=True).numpy().astype(np.int32)
            durs = torch.load(f'{dur_dir}/{stem}.pt', map_location='cpu',
                              weights_only=True).numpy().astype(np.int32)
        except Exception:
            continue
        T = hub.shape[0]
        ph_exp = np.repeat(phs, durs)
        L = min(T, len(ph_exp))
        ph_exp = ph_exp[:L]
        hub_clip = hub[:L]
        np.add.at(acc_n, ph_exp, 1)
        np.add.at(acc_sum, ph_exp, hub_clip)
        mag_sq = (hub_clip ** 2).sum(axis=1)
        np.add.at(acc_mag_sum, ph_exp, mag_sq)
        # Per-dim squared accumulator for intra-cluster variance
        np.add.at(acc_sqsum_perdim, ph_exp, hub_clip ** 2)

    centroids = np.full((N_PHONES, DIM), np.nan, dtype=np.float64)
    valid = acc_n >= MIN_FRAMES
    if valid.any():
        centroids[valid] = acc_sum[valid] / acc_n[valid][:, None]
    mean_mag = np.zeros(N_PHONES, dtype=np.float64)
    mean_mag[valid] = np.sqrt(acc_mag_sum[valid] / acc_n[valid])
    # Intra-cluster RMS std: sqrt(mean across dims of var per dim)
    cluster_std = np.zeros(N_PHONES, dtype=np.float64)
    if valid.any():
        mean_sq_perdim = acc_sqsum_perdim[valid] / acc_n[valid][:, None]
        var_perdim = mean_sq_perdim - centroids[valid] ** 2
        var_perdim = np.clip(var_perdim, 0.0, None)  # numerical safety
        cluster_std[valid] = np.sqrt(var_perdim.mean(axis=-1))
    return {
        'centroids': centroids, 'n_frames': acc_n,
        'mean_mag': mean_mag, 'cluster_std': cluster_std,
        'valid': valid, 'n_files': len(files),
    }


def angle_intra_lang_horizontal(stats, lang_to_ds, id2phone):
    """For each lang, for each phoneme: cos across that lang's datasets."""
    out = {}
    for lang, ds_list in lang_to_ds.items():
        ds_in_lang = [d for d in ds_list if d in stats]
        if len(ds_in_lang) < 2:
            continue
        per_phone = []
        for pid in range(N_PHONES):
            valid_ds = [d for d in ds_in_lang if stats[d]['valid'][pid]]
            if len(valid_ds) < 2:
                continue
            pairs = []
            for i in range(len(valid_ds)):
                for j in range(i + 1, len(valid_ds)):
                    c = cos_sim(stats[valid_ds[i]]['centroids'][pid],
                                stats[valid_ds[j]]['centroids'][pid])
                    pairs.append(((valid_ds[i], valid_ds[j]), c))
            cos_vals = [p[1] for p in pairs]
            worst = min(pairs, key=lambda x: x[1])
            total_frames = sum(int(stats[d]['n_frames'][pid]) for d in valid_ds)
            per_phone.append({
                'id': pid,
                'name': id2phone.get(pid, f'<{pid}>'),
                'min_cos': min(cos_vals),
                'mean_cos': float(np.mean(cos_vals)),
                'n_datasets': len(valid_ds),
                'n_pairs': len(pairs),
                'total_frames': total_frames,
                'worst_pair': list(worst[0]),
                'worst_cos': worst[1],
            })
        per_phone.sort(key=lambda x: x['min_cos'])
        out[lang] = per_phone
    return out


def angle_intra_lang_vertical(stats, id2phone, top_confuse=10):
    """Per dataset: phoneme separation matrix. Mean off-diagonal cos = how similar
    different phonemes look (lower = better). Plus most-confusable phoneme pairs.
    """
    out = {}
    for ds_name, s in stats.items():
        valid_ids = [pid for pid in range(N_PHONES) if s['valid'][pid]]
        # Skip silence tokens for separability (they're supposed to be similar = silence)
        valid_ids = [pid for pid in valid_ids if pid not in SILENCE_TOKENS]
        if len(valid_ids) < 2:
            continue
        pair_cos = []
        for i in range(len(valid_ids)):
            for j in range(i + 1, len(valid_ids)):
                pid_i, pid_j = valid_ids[i], valid_ids[j]
                c = cos_sim(s['centroids'][pid_i], s['centroids'][pid_j])
                pair_cos.append(((pid_i, pid_j), c))
        cos_vals = [p[1] for p in pair_cos]
        # Most confusable = highest cos between different phonemes
        pair_cos_sorted = sorted(pair_cos, key=lambda x: -x[1])
        confusable = [
            {'ph_a': id2phone.get(p[0][0], f'<{p[0][0]}>'),
             'ph_b': id2phone.get(p[0][1], f'<{p[0][1]}>'),
             'cos': p[1]}
            for p in pair_cos_sorted[:top_confuse]
        ]
        out[ds_name] = {
            'n_phones_compared': len(valid_ids),
            'mean_off_diag_cos': float(np.mean(cos_vals)),
            'median_off_diag_cos': float(np.median(cos_vals)),
            'max_off_diag_cos': float(np.max(cos_vals)),
            'most_confusable_pairs': confusable,
        }
    return out


def angle_cross_lang_horizontal(stats, lang_to_ds, id2phone):
    """Each phoneme: cos between dataset(zh) vs dataset(ja). Mean + min."""
    langs = list(lang_to_ds.keys())
    if 'zh' not in langs or 'ja' not in langs:
        return []
    zh_ds = [d for d in lang_to_ds['zh'] if d in stats]
    ja_ds = [d for d in lang_to_ds['ja'] if d in stats]
    if not zh_ds or not ja_ds:
        return []
    out = []
    for pid in range(N_PHONES):
        valid_zh = [d for d in zh_ds if stats[d]['valid'][pid]]
        valid_ja = [d for d in ja_ds if stats[d]['valid'][pid]]
        if not valid_zh or not valid_ja:
            continue
        pairs = []
        for dz in valid_zh:
            for dj in valid_ja:
                c = cos_sim(stats[dz]['centroids'][pid], stats[dj]['centroids'][pid])
                pairs.append(((dz, dj), c))
        cos_vals = [p[1] for p in pairs]
        worst = min(pairs, key=lambda x: x[1])
        total_frames_zh = sum(int(stats[d]['n_frames'][pid]) for d in valid_zh)
        total_frames_ja = sum(int(stats[d]['n_frames'][pid]) for d in valid_ja)
        out.append({
            'id': pid,
            'name': id2phone.get(pid, f'<{pid}>'),
            'min_cos': min(cos_vals),
            'mean_cos': float(np.mean(cos_vals)),
            'n_pairs': len(pairs),
            'total_frames_zh': total_frames_zh,
            'total_frames_ja': total_frames_ja,
            'worst_pair': list(worst[0]),
            'worst_cos': worst[1],
        })
    out.sort(key=lambda x: x['min_cos'])
    return out


def angle_cross_lang_vertical_confusion(stats, lang_to_ds, id2phone, threshold_margin=0.05):
    """Confusion alerts: cases where cos(zh_ph, ja_OTHER_ph) > cos(zh_ph, ja_same_ph) - margin.

    Uses lang-averaged centroids: zh = mean over zh datasets' centroids, ja = mean over ja.
    """
    if 'zh' not in lang_to_ds or 'ja' not in lang_to_ds:
        return []
    zh_ds = [d for d in lang_to_ds['zh'] if d in stats]
    ja_ds = [d for d in lang_to_ds['ja'] if d in stats]
    if not zh_ds or not ja_ds:
        return []

    # Per-language centroid: average of dataset centroids (where valid)
    zh_cen = np.full((N_PHONES, DIM), np.nan)
    ja_cen = np.full((N_PHONES, DIM), np.nan)
    zh_valid = np.zeros(N_PHONES, dtype=bool)
    ja_valid = np.zeros(N_PHONES, dtype=bool)
    for pid in range(N_PHONES):
        zhs = [stats[d]['centroids'][pid] for d in zh_ds if stats[d]['valid'][pid]]
        if zhs:
            zh_cen[pid] = np.mean(zhs, axis=0)
            zh_valid[pid] = True
        jas = [stats[d]['centroids'][pid] for d in ja_ds if stats[d]['valid'][pid]]
        if jas:
            ja_cen[pid] = np.mean(jas, axis=0)
            ja_valid[pid] = True

    shared = [pid for pid in range(N_PHONES) if zh_valid[pid] and ja_valid[pid]]
    if len(shared) < 2:
        return []

    alerts = []
    for pid in shared:
        same_cos = cos_sim(zh_cen[pid], ja_cen[pid])
        # Find any other JA phoneme that is closer to ZH[pid] than JA[pid]
        for pid_other in shared:
            if pid_other == pid: continue
            other_cos = cos_sim(zh_cen[pid], ja_cen[pid_other])
            if other_cos > same_cos - threshold_margin:
                alerts.append({
                    'zh_ph': id2phone.get(pid, f'<{pid}>'),
                    'ja_same_ph': id2phone.get(pid, f'<{pid}>'),
                    'cos_same': same_cos,
                    'ja_other_ph': id2phone.get(pid_other, f'<{pid_other}>'),
                    'cos_other': other_cos,
                    'risk_margin': other_cos - same_cos,
                })
    alerts.sort(key=lambda x: -x['risk_margin'])
    return alerts


def angle_variance(stats, lang_to_ds, id2phone):
    """Per-phoneme intra-cluster RMS std (tightness) + inter-dataset centroid spread.

    High intra = same dataset's same phoneme is scattered (hard for det_head to regress).
    High inter (relative to intra) = datasets disagree on this phoneme.
    """
    out = {}
    for lang, ds_list in lang_to_ds.items():
        ds_in_lang = [d for d in ds_list if d in stats]
        if not ds_in_lang:
            continue
        per_phone = []
        for pid in range(N_PHONES):
            valid_ds = [d for d in ds_in_lang if stats[d]['valid'][pid]]
            if len(valid_ds) < 1:
                continue
            intra_stds = {d: float(stats[d]['cluster_std'][pid]) for d in valid_ds}
            mean_intra = float(np.mean(list(intra_stds.values())))
            if len(valid_ds) >= 2:
                cens = np.stack([stats[d]['centroids'][pid] for d in valid_ds])
                mean_c = cens.mean(axis=0)
                inter_std = float(np.sqrt(((cens - mean_c) ** 2).mean()))
            else:
                inter_std = None
            # snr-style: inter / (inter + mean_intra). High = centroid spread dominates
            # intra scatter → cross-dataset disagreement >> within-dataset noise (BAD for det_head).
            ratio = None
            if inter_std is not None and (mean_intra + inter_std) > 1e-6:
                ratio = inter_std / (mean_intra + inter_std)
            per_phone.append({
                'id': pid,
                'name': id2phone.get(pid, f'<{pid}>'),
                'mean_intra_std': mean_intra,
                'inter_centroid_std': inter_std,
                'inter_over_total': ratio,
                'intra_per_dataset': intra_stds,
                'n_datasets': len(valid_ds),
            })
        # Sort by highest intra first (worst-cluster phonemes)
        per_phone.sort(key=lambda x: -x['mean_intra_std'])
        out[lang] = per_phone
    return out


def print_variance(data, max_show=15):
    print()
    print("=" * 100)
    print("[Angle 5] VARIANCE — intra-cluster (within phoneme-dataset) + inter-centroid (across datasets)")
    print("(High intra_std = same phoneme scatters inside one dataset. det_head needs tight clusters.")
    print(" High inter/total = cross-dataset centroids disagree more than within-dataset noise allows.)")
    print("=" * 100)
    for lang, per_phone in data.items():
        print(f"\n--- {lang.upper()} top-{max_show} phonemes by intra_std (worst clusters first) ---")
        print(f"{'Phone':>6s} {'IntraRMS':>10s} {'InterStd':>10s} {'I/Tot':>7s} {'#DS':>4s}")
        for p in per_phone[:max_show]:
            inter = f"{p['inter_centroid_std']:>10.4f}" if p['inter_centroid_std'] is not None else f"{'n/a':>10s}"
            ratio = f"{p['inter_over_total']:>7.3f}" if p['inter_over_total'] is not None else f"{'n/a':>7s}"
            print(f"{p['name']:>6s} {p['mean_intra_std']:>10.4f} {inter} {ratio} {p['n_datasets']:>4d}")


def silence_audit(stats):
    out = {}
    for sil_id, sil_name in SILENCE_TOKENS.items():
        per_ds = {}
        for n, s in stats.items():
            nf = int(s['n_frames'][sil_id])
            if nf == 0:
                continue
            mag = float(s['mean_mag'][sil_id]) if s['valid'][sil_id] else None
            per_ds[n] = {'n_frames': nf, 'mean_mag': mag}
        valid_ds = [n for n in per_ds if stats[n]['valid'][sil_id]]
        pairs = []
        for i in range(len(valid_ds)):
            for j in range(i + 1, len(valid_ds)):
                d1, d2 = valid_ds[i], valid_ds[j]
                c = cos_sim(stats[d1]['centroids'][sil_id], stats[d2]['centroids'][sil_id])
                pairs.append({'ds1': d1, 'ds2': d2, 'cos': c})
        pairs.sort(key=lambda x: x['cos'])
        out[sil_name] = {'per_dataset': per_ds, 'pairwise_cos': pairs}
    return out


def print_intra_lang_horizontal(data, max_show=15):
    print()
    print("=" * 100)
    print("[Angle 1] INTRA-LANG HORIZONTAL — same lang, different datasets, same phoneme")
    print("(High cos = good intra-lang cohesion. Low = some dataset disagrees inside its own language.)")
    print("=" * 100)
    for lang, per_phone in data.items():
        print(f"\n--- {lang.upper()} ({per_phone[0]['n_datasets']} datasets in lang) ---")
        print(f"{'Phone':>6s} {'Min':>7s} {'Mean':>7s} {'#Pairs':>6s} {'Frames':>8s} {'Worst pair':>30s}")
        for p in per_phone[:max_show]:
            wp = f"{p['worst_pair'][0][:9]}—{p['worst_pair'][1][:9]}"
            print(f"{p['name']:>6s} {p['min_cos']:>7.3f} {p['mean_cos']:>7.3f} "
                  f"{p['n_pairs']:>6d} {p['total_frames']:>8d} {wp:>30s}")
        if len(per_phone) > max_show:
            print(f"  ... ({len(per_phone) - max_show} more)")


def print_intra_lang_vertical(data, max_show=10):
    print()
    print("=" * 100)
    print("[Angle 2] INTRA-LANG VERTICAL — same dataset, different phonemes (separability)")
    print("(High mean off-diag = phonemes look alike inside dataset = poor separation. "
          "Worry if confusable_cos > 0.7.)")
    print("=" * 100)
    print(f"\n{'Dataset':<15s} {'#Phones':>8s} {'Mean':>7s} {'Median':>7s} {'Max':>7s}")
    for n, d in data.items():
        print(f"{n:<15s} {d['n_phones_compared']:>8d} "
              f"{d['mean_off_diag_cos']:>7.3f} {d['median_off_diag_cos']:>7.3f} "
              f"{d['max_off_diag_cos']:>7.3f}")
    print()
    print("Most confusable phoneme pairs (per dataset, top-5 each):")
    for n, d in data.items():
        pairs = ', '.join(f"{p['ph_a']}-{p['ph_b']}({p['cos']:.2f})"
                          for p in d['most_confusable_pairs'][:5])
        print(f"  {n:<15s}: {pairs}")


def print_cross_lang_horizontal(data, max_show=20):
    print()
    print("=" * 100)
    print("[Angle 3] CROSS-LANG HORIZONTAL — same phoneme, ZH vs JA datasets")
    print("(High = IPA cross-lang assumption holds. Low = lang-specific divergence on this phoneme.)")
    print("=" * 100)
    if not data:
        print("(No cross-lang shared phonemes.)"); return
    print(f"{'Phone':>6s} {'Min':>7s} {'Mean':>7s} {'#Pairs':>6s} {'ZH frm':>8s} {'JA frm':>8s} {'Worst':>30s}")
    for p in data[:max_show]:
        wp = f"{p['worst_pair'][0][:9]}—{p['worst_pair'][1][:9]}"
        print(f"{p['name']:>6s} {p['min_cos']:>7.3f} {p['mean_cos']:>7.3f} "
              f"{p['n_pairs']:>6d} {p['total_frames_zh']:>8d} {p['total_frames_ja']:>8d} {wp:>30s}")
    if len(data) > max_show:
        print(f"... ({len(data) - max_show} more)")


def print_cross_lang_confusion(alerts, max_show=15):
    print()
    print("=" * 100)
    print("[Angle 4] CROSS-LANG VERTICAL — phoneme confusion alerts")
    print("(zh_ph appears closer to a DIFFERENT ja_ph than to its same-name ja_ph. "
          "Risk of model misidentifying phonemes across languages.)")
    print("=" * 100)
    if not alerts:
        print("No alerts (no cases where cross-lang ph_diff cos > ph_same cos - margin)."); return
    print(f"{'ZH ph':>6s} -> {'JA same':>8s} {'cos':>7s} | {'JA other':>8s} {'cos':>7s} {'risk_margin':>10s}")
    for a in alerts[:max_show]:
        print(f"{a['zh_ph']:>6s} -> {a['ja_same_ph']:>8s} {a['cos_same']:>7.3f} | "
              f"{a['ja_other_ph']:>8s} {a['cos_other']:>7.3f} {a['risk_margin']:>+10.3f}")
    if len(alerts) > max_show:
        print(f"... ({len(alerts) - max_show} more alerts)")


def print_silence(silence):
    print()
    print("=" * 100)
    print("[Silence] SP / AP / br token magnitude + cross-dataset cos")
    print("(Clean silence: low ||H||, high cos. Currently ~9.0 magnitude — silence is not silent.)")
    print("=" * 100)
    for sil_name, d in silence.items():
        print(f"\n[{sil_name}]")
        print(f"  {'Dataset':<15s} {'n_frames':>9s} {'mean_||H||':>12s}")
        for n, info in d['per_dataset'].items():
            mag_str = f"{info['mean_mag']:>12.3f}" if info['mean_mag'] is not None else f"{'n/a':>12s}"
            print(f"  {n:<15s} {info['n_frames']:>9d} {mag_str}")
        if d['pairwise_cos']:
            worst = d['pairwise_cos'][:3]
            best = d['pairwise_cos'][-3:]
            print(f"  Worst pairs: " + '; '.join(f"{p['ds1']}-{p['ds2']}={p['cos']:.3f}" for p in worst))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='post_sofa', help='output filename suffix')
    parser.add_argument('--baseline-json', default=None,
                        help='compare against pre-SOFA JSON (optional)')
    parser.add_argument('--max-files', type=int, default=MAX_FILES)
    args = parser.parse_args()

    vocab = PhonemeVocab()
    id2phone = vocab.id2phone

    print("=" * 100)
    print(f"4-angle phoneme contamination audit — tag={args.tag}")
    print("=" * 100)

    stats = {}
    for name, path in DATASETS.items():
        full = REPO_ROOT / path
        if not full.exists():
            continue
        t0 = time.time()
        s = collect_per_phoneme(str(full), max_files=args.max_files)
        if s is None:
            continue
        stats[name] = s
        print(f"  {name:<15s} files={s['n_files']:>5d} "
              f"valid_phones={int(s['valid'].sum()):>3d} [{time.time() - t0:.0f}s]")

    if len(stats) < 2:
        print("Need at least 2 datasets")
        return

    lang_to_ds = defaultdict(list)
    for n in stats:
        lang_to_ds[LANG_OF.get(n, 'unk')].append(n)
    print()
    print("Language groups:")
    for l, ds in lang_to_ds.items():
        print(f"  {l}: {ds}")

    a1 = angle_intra_lang_horizontal(stats, lang_to_ds, id2phone)
    a2 = angle_intra_lang_vertical(stats, id2phone)
    a3 = angle_cross_lang_horizontal(stats, lang_to_ds, id2phone)
    a4 = angle_cross_lang_vertical_confusion(stats, lang_to_ds, id2phone)
    a5 = angle_variance(stats, lang_to_ds, id2phone)
    sil = silence_audit(stats)

    print_intra_lang_horizontal(a1)
    print_intra_lang_vertical(a2)
    print_cross_lang_horizontal(a3)
    print_cross_lang_confusion(a4)
    print_variance(a5)
    print_silence(sil)

    out = {
        'tag': args.tag,
        'datasets_by_lang': {l: list(ds) for l, ds in lang_to_ds.items()},
        'intra_lang_horizontal': a1,
        'intra_lang_vertical': a2,
        'cross_lang_horizontal': a3,
        'cross_lang_confusion_alerts': a4,
        'variance': a5,
        'silence': sil,
    }
    out_path = REPO_ROOT / 'docs' / f'phoneme_contamination_audit_{args.tag}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWritten: {out_path}")

    # Optional baseline diff
    if args.baseline_json:
        base_path = REPO_ROOT / args.baseline_json
        if base_path.exists():
            print()
            print("=" * 100)
            print(f"DIFF vs baseline: {args.baseline_json}")
            print("=" * 100)
            with open(base_path, encoding='utf-8') as f:
                base = json.load(f)
            # v1 baseline schema only had `phone_agreement` (== combined intra+cross horizontal)
            base_phone = {p['name']: p for p in base.get('phone_agreement', [])}
            print(f"\n{'Phone':>6s} {'pre min':>8s} {'post min(zh-ja)':>17s} {'delta':>8s}")
            for p in a3[:30]:
                pre = base_phone.get(p['name'])
                if pre is None: continue
                delta = p['min_cos'] - pre['min_cos']
                print(f"{p['name']:>6s} {pre['min_cos']:>8.3f} {p['min_cos']:>17.3f} {delta:>+8.3f}")


if __name__ == '__main__':
    main()
