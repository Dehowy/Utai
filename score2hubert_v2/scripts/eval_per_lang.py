"""Per-language validation metrics for any Score2HuBERT v2 checkpoint.

Runs alongside training (loads latest.pt or best.pt) without interrupting it.
Reports val loss, hub_l1, hub_cos_sim, f0_l1 separately for zh/en/ja.

Usage:
    python scripts/eval_per_lang.py [--ckpt checkpoints/latest.pt] [--max-batches 50]
"""
import sys, os, argparse, yaml
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
    """Match trainer filter — always rejects GTSinger (dropped 2026-04-13)."""
    f = cfg_data.get('entry_filters', {})
    singing_only = f.get('singing_only', False)
    def _filter(e):
        # Hard reject: GTSinger permanently dropped
        if e.get('dataset', '') == 'gtsinger':
            return False
        if singing_only and e.get('singing_ratio', 1.0) <= 0:
            return False
        return True
    return _filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/latest.pt')
    ap.add_argument('--config', default='configs/train.yaml')
    ap.add_argument('--max-batches', type=int, default=0, help='0=all val batches')
    ap.add_argument('--batch-size', type=int, default=32,
                    help='Lower than training to coexist with live training')
    args = ap.parse_args()

    device = 'cuda'

    # Load config + checkpoint
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
    print(f'Loaded: {args.ckpt} (step={step})')

    # Build val dataset
    dc = cfg['data']
    val_ds = Score2HuBERTDataset(
        data_dirs=dc['val_dirs'], split='test',
        max_frames=dc['max_frames'], entry_filter=make_filter(dc),
    )
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
        generator=torch.Generator().manual_seed(42),
    )
    print(f'Val: {len(val_ds)} samples, {len(loader)} batches of {args.batch_size}')

    # Per-lang accumulators
    stats = defaultdict(lambda: {'n': 0, 'l1': 0.0, 'cos': 0.0, 'f0': 0.0, 'total': 0.0})

    # Loss weights (from config, but we also use them here to match trainer)
    lw = cfg['loss']
    w_l1 = lw['hubert_l1']; w_cos = lw['hubert_cos']
    w_f0 = lw['f0_l1']; w_v = lw['voicing_bce']

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            langs = batch['languages']  # list[str]
            with autocast('cuda'):
                out = model(
                    phonemes=batch['phonemes'].to(device),
                    pitch=batch['pitch'].to(device),
                    durations=batch['duration'].to(device),
                    phone_mask=batch['phone_mask'].to(device),
                    f0_target=batch['f0'].to(device),
                )
            frame_mask = batch['frame_mask'].to(device)
            valid = ~frame_mask  # [B, T]
            hubert_t = batch['hubert'].to(device)
            hubert_p = out['hubert_pred']
            f0_t = batch['f0'].to(device)
            f0_p = out['f0_pred']

            # Per-sample metrics
            cos = F.cosine_similarity(hubert_p, hubert_t, dim=-1)  # [B, T]
            B = hubert_p.shape[0]
            for b in range(B):
                lang = langs[b]
                v = valid[b]
                if v.sum() == 0:
                    continue
                l1 = F.l1_loss(hubert_p[b][v], hubert_t[b][v]).item()
                c = cos[b][v].mean().item()  # avg cos sim
                voiced = (f0_t[b] > 30.0) & v
                if voiced.any():
                    lf_p = torch.log1p(f0_p[b][voiced])
                    lf_t = torch.log1p(f0_t[b][voiced])
                    f0l = F.l1_loss(lf_p, lf_t).item()
                else:
                    f0l = 0.0
                s = stats[lang]
                s['n'] += 1
                s['l1'] += l1
                s['cos'] += c
                s['f0'] += f0l
                # total loss (approximation, voicing bce skipped for speed)
                s['total'] += w_l1 * l1 + w_cos * (1 - c) + w_f0 * f0l

            if (bi + 1) % 10 == 0:
                print(f'  batch {bi+1}/{len(loader) if not args.max_batches else args.max_batches}', flush=True)

    # Report
    print()
    print(f'{"Lang":8s} {"N":>6s} {"total_loss":>11s} {"hub_l1":>8s} {"cos_sim":>9s} {"f0_l1":>8s}')
    print('-' * 55)
    total_n = 0
    total_acc = defaultdict(float)
    for lang in sorted(stats.keys()):
        s = stats[lang]
        n = s['n']
        if n == 0: continue
        total_n += n
        for k in ['l1', 'cos', 'f0', 'total']:
            total_acc[k] += s[k]
        print(f'{lang:8s} {n:>6d} {s["total"]/n:>11.4f} {s["l1"]/n:>8.4f} {s["cos"]/n:>9.4f} {s["f0"]/n:>8.4f}')
    print('-' * 55)
    if total_n > 0:
        print(f'{"AGGREG":8s} {total_n:>6d} {total_acc["total"]/total_n:>11.4f} '
              f'{total_acc["l1"]/total_n:>8.4f} {total_acc["cos"]/total_n:>9.4f} '
              f'{total_acc["f0"]/total_n:>8.4f}')


if __name__ == '__main__':
    main()
