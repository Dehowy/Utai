#!/usr/bin/env python3
"""
Profile per-step time breakdown for Score2HuBERT training.

Measures wall-clock time spent in each phase of a training step:
  - data: waiting for next batch from DataLoader
  - h2d:  host-to-device transfer
  - fwd:  model forward pass
  - bwd:  backward pass
  - opt:  optimizer step + scaler update
  - sync: implicit GPU-CPU syncs from .item() inside forward
  - other: residual (Python overhead, scheduler, etc.)

Also reports samples/sec and est. GPU utilization.

Usage:
    venv/Scripts/python.exe scripts/profile_step.py \\
        --config configs/pretrain_opencpop.yaml \\
        --steps 100 --warmup 10

    # Compare against num_workers=2:
    venv/Scripts/python.exe scripts/profile_step.py \\
        --config configs/pretrain_opencpop.yaml \\
        --steps 100 --warmup 10 \\
        --num-workers 2 --persistent-workers --prefetch-factor 4
"""

import sys
import time
import argparse
import logging
import statistics
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Project imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.score2hubert import Score2HuBERT
from src.data.dataset import Score2HuBERTDataset
from src.data.collate import collate_fn

logging.basicConfig(level=logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/pretrain_opencpop.yaml")
    p.add_argument("--steps", type=int, default=100, help="Number of measured steps")
    p.add_argument("--warmup", type=int, default=10, help="Warmup steps to discard")
    p.add_argument("--num-workers", type=int, default=None,
                   help="Override config num_workers")
    p.add_argument("--persistent-workers", action="store_true",
                   help="Enable persistent_workers (only if num_workers > 0)")
    p.add_argument("--prefetch-factor", type=int, default=None,
                   help="Set prefetch_factor (only if num_workers > 0)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override config batch_size")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed precision (force FP32)")
    return p.parse_args()


def build_dataloader(config, args):
    data_cfg = config["data"]
    train_cfg = config["training"]

    dataset = Score2HuBERTDataset(
        processed_dir=data_cfg["processed_dir"],
        split="train",
        max_frames=data_cfg["max_frames"],
        min_frames=data_cfg["min_frames"],
        augmentation=data_cfg.get("augmentation"),
    )

    batch_size = args.batch_size or train_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else train_cfg["num_workers"]

    # CLI overrides win; otherwise fall back to config
    persistent = args.persistent_workers or train_cfg.get("persistent_workers", False)
    prefetch = args.prefetch_factor if args.prefetch_factor is not None else train_cfg.get("prefetch_factor")

    # Build kwargs (only pass worker-related args when workers > 0)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    if num_workers > 0:
        if persistent:
            loader_kwargs["persistent_workers"] = True
        if prefetch is not None:
            loader_kwargs["prefetch_factor"] = prefetch

    loader = DataLoader(dataset, **loader_kwargs)
    return loader, dataset, batch_size, num_workers, persistent, prefetch


def main():
    args = parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA not available"

    print("=" * 70)
    print("Score2HuBERT Step Profiler")
    print("=" * 70)
    print(f"Config:        {args.config}")
    print(f"Device:        {torch.cuda.get_device_name(0)}")

    # Build model
    print("\nBuilding model...")
    model = Score2HuBERT(config).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params:  {n_params/1e6:.1f}M")

    # Build optimizer + scaler
    opt_cfg = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        weight_decay=opt_cfg["weight_decay"],
        eps=opt_cfg["eps"],
    )
    use_amp = (not args.no_amp) and config["training"].get("fp16", True)
    scaler = GradScaler("cuda", enabled=use_amp)

    # Build dataloader
    loader, dataset, batch_size, num_workers, persistent, prefetch = build_dataloader(config, args)
    print(f"Dataset:       {len(dataset)} train samples")
    print(f"Batch size:    {batch_size}")
    print(f"num_workers:   {num_workers}")
    if num_workers > 0:
        print(f"persistent:    {persistent}")
        print(f"prefetch_fac:  {prefetch}")
    print(f"AMP (FP16):    {use_amp}")
    print(f"Steps:         {args.steps} (warmup {args.warmup})")
    print()

    # Phase timers (lists of seconds per step, after warmup)
    t_data, t_h2d, t_fwd, t_bwd, t_opt, t_total = [], [], [], [], [], []

    data_iter = iter(loader)
    grad_clip = config["training"].get("grad_clip", 1.0)

    print("Starting profile loop...")
    print("-" * 70)
    overall_start = time.perf_counter()

    for step in range(args.warmup + args.steps):
        # ---- Phase: data load (waiting for next batch) ----
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        t1 = time.perf_counter()

        # ---- Phase: host-to-device transfer ----
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # ---- Phase: forward ----
        with autocast("cuda", enabled=use_amp):
            var_targets = {}
            if "energy" in batch:
                var_targets["energy"] = batch["energy"]
            if "breathiness" in batch:
                var_targets["breathiness"] = batch["breathiness"]

            output = model(
                phonemes=batch["phonemes"],
                pitch=batch["pitch"],
                duration=batch["duration"],
                singing_ratio=batch["singing_ratio"],
                hubert_target=batch["hubert"],
                f0_target=batch["f0"],
                phone_mask=batch["phone_mask"],
                frame_mask=batch["frame_mask"],
                lang_id=batch["lang_id"],
                variance_targets=var_targets if var_targets else None,
                tech=batch.get("tech"),
            )
            loss = output["loss"]
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # ---- Phase: backward ----
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        # ---- Phase: optimizer step ----
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t5 = time.perf_counter()

        # Skip warmup
        if step >= args.warmup:
            t_data.append(t1 - t0)
            t_h2d.append(t2 - t1)
            t_fwd.append(t3 - t2)
            t_bwd.append(t4 - t3)
            t_opt.append(t5 - t4)
            t_total.append(t5 - t0)

        # Progress
        if (step + 1) % 25 == 0:
            print(f"  step {step+1:4d}/{args.warmup + args.steps}  "
                  f"last_total={t5-t0:.3f}s  "
                  f"loss={loss.item():.3f}")

    overall_elapsed = time.perf_counter() - overall_start
    print("-" * 70)

    # Stats
    def stat(xs):
        return {
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "p90": sorted(xs)[int(len(xs) * 0.9)] if len(xs) > 1 else xs[0],
        }

    s_data = stat(t_data)
    s_h2d = stat(t_h2d)
    s_fwd = stat(t_fwd)
    s_bwd = stat(t_bwd)
    s_opt = stat(t_opt)
    s_total = stat(t_total)

    # Per-phase percent of median total
    median_total = s_total["median"]
    pct_data = 100 * s_data["median"] / median_total
    pct_h2d = 100 * s_h2d["median"] / median_total
    pct_fwd = 100 * s_fwd["median"] / median_total
    pct_bwd = 100 * s_bwd["median"] / median_total
    pct_opt = 100 * s_opt["median"] / median_total
    accounted = pct_data + pct_h2d + pct_fwd + pct_bwd + pct_opt

    samples_per_sec = batch_size / median_total

    # GPU "ideal" if all CPU/data overhead vanished
    gpu_busy = s_fwd["median"] + s_bwd["median"] + s_opt["median"]
    est_util = 100 * gpu_busy / median_total

    print()
    print("=" * 70)
    print("Per-step time breakdown (median over measured steps)")
    print("=" * 70)
    print(f"{'phase':<8} {'median(ms)':>11} {'mean(ms)':>11} {'p90(ms)':>11} {'%':>7}")
    print("-" * 70)
    print(f"{'data':<8} {s_data['median']*1000:>11.1f} {s_data['mean']*1000:>11.1f} {s_data['p90']*1000:>11.1f} {pct_data:>6.1f}%")
    print(f"{'h2d':<8} {s_h2d['median']*1000:>11.1f} {s_h2d['mean']*1000:>11.1f} {s_h2d['p90']*1000:>11.1f} {pct_h2d:>6.1f}%")
    print(f"{'fwd':<8} {s_fwd['median']*1000:>11.1f} {s_fwd['mean']*1000:>11.1f} {s_fwd['p90']*1000:>11.1f} {pct_fwd:>6.1f}%")
    print(f"{'bwd':<8} {s_bwd['median']*1000:>11.1f} {s_bwd['mean']*1000:>11.1f} {s_bwd['p90']*1000:>11.1f} {pct_bwd:>6.1f}%")
    print(f"{'opt':<8} {s_opt['median']*1000:>11.1f} {s_opt['mean']*1000:>11.1f} {s_opt['p90']*1000:>11.1f} {pct_opt:>6.1f}%")
    print("-" * 70)
    print(f"{'TOTAL':<8} {s_total['median']*1000:>11.1f} {s_total['mean']*1000:>11.1f} {s_total['p90']*1000:>11.1f} {accounted:>6.1f}%")
    print()
    print(f"Throughput:           {samples_per_sec:.1f} samples/sec")
    print(f"GPU busy (fwd+bwd+opt): {gpu_busy*1000:.1f} ms / step")
    print(f"Est. GPU utilization: {est_util:.1f}%  (rest is data/transfer/python)")
    print()

    # Memory
    print(f"GPU memory peak:      {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"GPU memory reserved:  {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
    print()

    print(f"Wall-clock total ({args.warmup + args.steps} steps incl. warmup): {overall_elapsed:.1f}s")


if __name__ == "__main__":
    main()
