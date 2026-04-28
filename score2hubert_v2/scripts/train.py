"""Launch Score2HuBERT v2 training.

Usage: python scripts/train.py [--config configs/train.yaml] [--resume path/to/checkpoint.pt]
"""
import sys
import os
import argparse

# CUDA allocator: enable expandable segments (PyTorch 2.1+) to reduce fragmentation.
# MUST be set BEFORE any torch import. Added 2026-04-15 after Run 8 showed VRAM
# headroom <5% caused 30% slowdown + silent shared-memory spillover on long runs.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train.yaml')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

    trainer = Trainer(args.config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == '__main__':
    main()
