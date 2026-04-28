"""Profile training step stages + top ops to diagnose Run 5 slowdown.

Reports:
1. Per-stage timing (data load / forward / backward / optimizer) averaged over N steps
2. torch.profiler top ops by CUDA self-time and CPU self-time

Run:
    PYTHONIOENCODING=utf-8 venv/Scripts/python.exe scripts/profile_step.py --config configs/train.yaml
"""

import argparse
import sys
import time
import logging
from pathlib import Path

import torch
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--profile-steps', type=int, default=50)
    parser.add_argument('--profiler-steps', type=int, default=10)
    args = parser.parse_args()

    # Keep trainer's own INFO logs visible but silent for everything else
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # ALT: seed BEFORE init (reproducible init), then re-randomize CUDA RNG
    # to avoid cuDNN LSTM slow path
    torch.manual_seed(42)
    trainer = Trainer(args.config)
    trainer.model.train()

    # Re-randomize CUDA RNG (and CPU RNG) — model is already built with seed=42
    import time as _t
    torch.manual_seed(int(_t.time() * 1000) & 0x7FFFFFFF)

    train_iter = iter(trainer.train_loader)

    # --- trainer.train()-equivalent loop (NO stage timing, NO profiler) ---
    print(f"\n[trainer.train() replica — seed init + randomize CUDA] 50 steps...", flush=True)
    t_rep_start = time.time()
    running_loss = {}
    log_count = 0
    for i in range(50):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(trainer.train_loader)
            batch = next(train_iter)

        try:
            loss_dict = trainer.train_step(batch)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                continue
            raise
        for k, v in loss_dict.items():
            running_loss[k] = running_loss.get(k, 0) + v
        log_count += 1

    torch.cuda.synchronize()
    t_rep_elapsed = time.time() - t_rep_start
    print(f"[trainer.train() replica] 50 steps in {t_rep_elapsed:.2f}s "
          f"({50/t_rep_elapsed:.2f} step/s)", flush=True)

    # --- warmup ---
    print(f"\n[warmup] {args.warmup} steps...", flush=True)
    t_warm_start = time.time()
    for _ in range(args.warmup):
        batch = next(train_iter)
        _ = trainer.train_step(batch)
    torch.cuda.synchronize()
    print(f"[warmup] done in {time.time()-t_warm_start:.1f}s", flush=True)

    # --- stage timing ---
    n = args.profile_steps
    print(f"\n[stage timing] {n} steps with CUDA sync between stages...", flush=True)

    times = {'data': 0.0, 'forward': 0.0, 'loss': 0.0, 'backward': 0.0, 'opt': 0.0}
    t_total_start = time.time()

    for i in range(n):
        t0 = time.time()
        batch = next(train_iter)
        torch.cuda.synchronize()
        t1 = time.time()
        times['data'] += t1 - t0

        trainer.optimizer.zero_grad()
        with autocast(enabled=trainer.cfg['training']['fp16']):
            output = trainer.model(
                phonemes=batch['phonemes'].to(trainer.device),
                pitch=batch['pitch'].to(trainer.device),
                durations=batch['duration'].to(trainer.device),
                phone_mask=batch['phone_mask'].to(trainer.device),
                f0_target=batch['f0'].to(trainer.device),
            )
        torch.cuda.synchronize()
        t2 = time.time()
        times['forward'] += t2 - t1

        with autocast(enabled=trainer.cfg['training']['fp16']):
            total_loss, _ = trainer.compute_loss(batch, output)
        torch.cuda.synchronize()
        t3 = time.time()
        times['loss'] += t3 - t2

        trainer.scaler.scale(total_loss).backward()
        torch.cuda.synchronize()
        t4 = time.time()
        times['backward'] += t4 - t3

        trainer.scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), trainer.cfg['training']['grad_clip']
        )
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        trainer.scheduler.step()
        torch.cuda.synchronize()
        t5 = time.time()
        times['opt'] += t5 - t4

    t_total_end = time.time()
    total = t_total_end - t_total_start

    print(f"\n=== Stage timing ({n} steps) ===")
    print(f"  total:     {total:.2f}s  ({n/total:.2f} step/s)")
    for k in ['data', 'forward', 'loss', 'backward', 'opt']:
        ms = times[k] * 1000 / n
        pct = times[k] / total * 100
        print(f"  {k:10s} {ms:7.1f} ms/step  ({pct:4.1f}%)")

    # --- torch.profiler ---
    print(f"\n[profiler] {args.profiler_steps} steps with CPU+CUDA activities...", flush=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(args.profiler_steps):
            batch = next(train_iter)
            _ = trainer.train_step(batch)
    torch.cuda.synchronize()

    print("\n=== Top 30 ops by CUDA self time ===")
    print(prof.key_averages().table(
        sort_by='self_cuda_time_total', row_limit=30
    ))
    print("\n=== Top 30 ops by CPU self time ===")
    print(prof.key_averages().table(
        sort_by='self_cpu_time_total', row_limit=30
    ))


if __name__ == '__main__':
    main()
