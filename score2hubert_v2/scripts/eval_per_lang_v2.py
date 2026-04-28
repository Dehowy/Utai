"""Per-language + per-dataset validation metrics (v2).

Fixes v1 bugs and adds per-dataset breakdown:
  - F0 filter matches trainer: (f0_target > 30 & note_pitch > 0 & valid)
    → rest-voiced frames (model correctly predicts 0, GT has carryover) are excluded.
    This fixes the "ZH f0_mae ~2800c" artifact present in v1.
  - Per-dataset breakdown: JVS-MuSiC (100 singers) shouldn't drag JA aggregate down;
    single-singer datasets (Kiritan, Itako, ONIKU, OFUTON, Natsume) shouldn't inflate it.
  - Reports f0_mae in cents (not just log-space L1) for perceptual intuition.

Usage:
    python scripts/eval_per_lang_v2.py --ckpt checkpoints/best.pt
    python scripts/eval_per_lang_v2.py --ckpt checkpoints/final.pt
"""
import sys, argparse, yaml, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

from src.model.score2hubert import Score2HuBERT
from src.data.dataset import Score2HuBERTDataset
from src.data.collate import collate_fn


def make_filter(cfg_data):
    f = cfg_data.get('entry_filters', {})
    singing_only = f.get('singing_only', False)
    def _filter(e):
        if e.get('dataset', '') == 'gtsinger':
            return False
        if singing_only and e.get('singing_ratio', 1.0) <= 0:
            return False
        return True
    return _filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/best.pt')
    ap.add_argument('--config', default='configs/train.yaml')
    ap.add_argument('--batch-size', type=int, default=48)
    ap.add_argument('--max-batches', type=int, default=0, help='0=all val batches')
    ap.add_argument('--out-json', default=None, help='optional output path for per-dataset json')
    args = ap.parse_args()

    device = 'cuda'

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    mc = cfg['model']
    model = Score2HuBERT(
        n_phones=mc['n_phones'], hidden_dim=mc['hidden_dim'],
        hubert_dim=mc['hubert_dim'], n_enc_layers=mc['n_enc_layers'],
        n_heads=mc['n_heads'], enc_kernel=mc['enc_kernel'],
        n_det_blocks=mc['n_det_blocks'], det_kernel=mc['det_kernel'],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    step = ckpt.get('step', '?')
    print(f'[eval_v2] Loaded: {args.ckpt} (step={step})')

    dc = cfg['data']
    val_ds = Score2HuBERTDataset(
        data_dirs=dc['val_dirs'], split='test',
        max_frames=dc['max_frames'], entry_filter=make_filter(dc),
    )
    # Attach dataset-name per entry from data_dir basename
    dataset_of = [Path(e['_data_dir']).name for e in val_ds.entries]

    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )
    print(f'[eval_v2] Val: {len(val_ds)} samples, {len(loader)} batches')

    # Stats per dataset
    ds_stats = defaultdict(lambda: {
        'n': 0, 'lang': None,
        'hub_l1': 0.0, 'hub_cos': 0.0,
        'f0_log_l1': 0.0, 'f0_mae_cents': 0.0, 'f0_n_voiced': 0,
        'voicing_bce': 0.0,
    })

    sample_idx = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            langs = batch['languages']
            with autocast('cuda'):
                out = model(
                    phonemes=batch['phonemes'].to(device),
                    pitch=batch['pitch'].to(device),
                    durations=batch['duration'].to(device),
                    phone_mask=batch['phone_mask'].to(device),
                    f0_target=batch['f0'].to(device),
                )
            frame_mask = batch['frame_mask'].to(device)
            valid = ~frame_mask
            hubert_t = batch['hubert'].to(device)
            hubert_p = out['hubert_pred']
            f0_t = batch['f0'].to(device)
            f0_p = out['f0_pred']
            note_pitch = out['note_pitch_frame']
            voicing_logit = out['voicing_logit']

            # SP=0 force (Run 8): per-batch sp_frame_mask via phoneme expansion
            SP_ID = 3
            phs = batch['phonemes'].to(device)
            durs = batch['duration'].to(device)
            phm = batch['phone_mask'].to(device)
            B = hubert_p.shape[0]
            sp_frame_mask = torch.zeros(B, hubert_t.shape[1], dtype=torch.bool, device=device)
            for b in range(B):
                vp = ~phm[b]
                pb = phs[b][vp]
                db = durs[b][vp]
                if db.sum() == 0:
                    continue
                exp = torch.repeat_interleave(pb, db)
                L = min(hubert_t.shape[1], int(exp.shape[0]))
                sp_frame_mask[b, :L] = (exp[:L] == SP_ID)
            hubert_t = hubert_t.clone()
            hubert_t[sp_frame_mask] = 0.0

            cos = F.cosine_similarity(hubert_p, hubert_t, dim=-1)
            for b in range(B):
                ds_name = dataset_of[sample_idx]
                sample_idx += 1
                v = valid[b]
                if v.sum() == 0:
                    continue
                # HuBERT L1 — includes SP frames (target=0)
                hub_l1 = F.l1_loss(hubert_p[b][v], hubert_t[b][v]).item()
                # HuBERT cos — excludes SP frames (cos vs 0 is undefined)
                non_sp_v = v & ~sp_frame_mask[b]
                hub_cos_sim = cos[b][non_sp_v].mean().item() if non_sp_v.any() else 0.0
                # F0 with correct filter
                voiced = (f0_t[b] > 30.0) & v & (note_pitch[b] > 0)
                n_voiced = int(voiced.sum().item())
                if n_voiced > 0:
                    lf_p = torch.log1p(f0_p[b][voiced])
                    lf_t = torch.log1p(f0_t[b][voiced])
                    f0_log_l1 = F.l1_loss(lf_p, lf_t).item()
                    cents = (1200.0 * torch.log2(
                        f0_p[b][voiced].clamp(min=1e-3) / f0_t[b][voiced].clamp(min=1e-3)
                    )).abs().mean().item()
                else:
                    f0_log_l1 = 0.0
                    cents = 0.0
                # Voicing BCE per sample
                voicing_tgt = (f0_t[b] > 30.0).float()
                v_bce = F.binary_cross_entropy_with_logits(
                    voicing_logit[b][v], voicing_tgt[v]
                ).item()

                s = ds_stats[ds_name]
                s['n'] += 1
                s['lang'] = langs[b]
                s['hub_l1'] += hub_l1
                s['hub_cos'] += hub_cos_sim
                s['f0_log_l1'] += f0_log_l1 if n_voiced > 0 else 0
                s['f0_mae_cents'] += cents if n_voiced > 0 else 0
                s['f0_n_voiced'] += 1 if n_voiced > 0 else 0
                s['voicing_bce'] += v_bce

            if (bi + 1) % 20 == 0:
                print(f'  batch {bi+1}/{len(loader)}', flush=True)

    # Compute averages + aggregate by language
    results = {'checkpoint': str(args.ckpt), 'step': step, 'by_dataset': {}, 'by_lang': {}}
    print()
    header = f'{"Dataset":18s} {"Lang":5s} {"N":>5s} {"hub_l1":>8s} {"cos_sim":>9s} {"f0_log":>8s} {"f0_mae_c":>10s} {"voicing":>9s}'
    print(header)
    print('-' * len(header))

    lang_agg = defaultdict(lambda: {'n': 0, 'hub_l1': 0.0, 'hub_cos': 0.0,
                                     'f0_log_l1': 0.0, 'f0_mae_cents': 0.0,
                                     'f0_n_voiced': 0, 'voicing_bce': 0.0})

    for ds_name in sorted(ds_stats.keys()):
        s = ds_stats[ds_name]
        n = s['n']
        if n == 0:
            continue
        hub_l1 = s['hub_l1'] / n
        hub_cos = s['hub_cos'] / n
        n_voiced = max(s['f0_n_voiced'], 1)
        f0_log = s['f0_log_l1'] / n_voiced
        f0_c = s['f0_mae_cents'] / n_voiced
        v_bce = s['voicing_bce'] / n
        lang = s['lang']
        print(f'{ds_name:18s} {lang:5s} {n:>5d} {hub_l1:>8.4f} {hub_cos:>9.4f} '
              f'{f0_log:>8.4f} {f0_c:>10.2f} {v_bce:>9.4f}')
        results['by_dataset'][ds_name] = {
            'lang': lang, 'n': n,
            'hub_l1': hub_l1, 'hub_cos_sim': hub_cos,
            'f0_log_l1': f0_log, 'f0_mae_cents': f0_c,
            'voicing_bce': v_bce,
        }
        # Aggregate into lang
        la = lang_agg[lang]
        la['n'] += n
        la['hub_l1'] += s['hub_l1']
        la['hub_cos'] += s['hub_cos']
        la['f0_log_l1'] += s['f0_log_l1']
        la['f0_mae_cents'] += s['f0_mae_cents']
        la['f0_n_voiced'] += s['f0_n_voiced']
        la['voicing_bce'] += s['voicing_bce']

    print('-' * len(header))
    for lang in sorted(lang_agg.keys()):
        a = lang_agg[lang]
        n = a['n']
        n_voiced = max(a['f0_n_voiced'], 1)
        print(f'{"[AGG] " + lang:18s} {lang:5s} {n:>5d} '
              f'{a["hub_l1"]/n:>8.4f} {a["hub_cos"]/n:>9.4f} '
              f'{a["f0_log_l1"]/n_voiced:>8.4f} {a["f0_mae_cents"]/n_voiced:>10.2f} '
              f'{a["voicing_bce"]/n:>9.4f}')
        results['by_lang'][lang] = {
            'n': n,
            'hub_l1': a['hub_l1']/n, 'hub_cos_sim': a['hub_cos']/n,
            'f0_log_l1': a['f0_log_l1']/n_voiced,
            'f0_mae_cents': a['f0_mae_cents']/n_voiced,
            'voicing_bce': a['voicing_bce']/n,
        }

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'\nWritten: {args.out_json}')


if __name__ == '__main__':
    main()
