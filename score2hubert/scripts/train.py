#!/usr/bin/env python3
"""
Score2HuBERT Training Entry Point.

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/base.yaml --resume checkpoints/best.pt
    python scripts/train.py --config configs/pretrain_opencpop.yaml

Multi-phase training:
    1. Pre-train on Opencpop (Chinese):
       python scripts/train.py --config configs/pretrain_opencpop.yaml
    2. Fine-tune on Japanese:
       python scripts/train.py --config configs/finetune_ja.yaml --resume checkpoints/pretrain_best.pt
"""

import sys
import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import Trainer


def setup_logging(log_dir: str, level: str = "INFO"):
    """Configure logging to console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "train.log", mode="a"),
    ]

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Score2HuBERT model")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (full state restore)")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Warm-start from a checkpoint: load model+EMA only "
                             "(strict=False), keep fresh optimizer/scheduler/step. "
                             "Use when switching architecture or training mode.")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides (e.g., training.batch_size=32)")
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Apply overrides
    if args.override:
        overrides = OmegaConf.from_dotlist(args.override)
        config = OmegaConf.merge(config, overrides)

    # Convert to plain dict (some code expects dict)
    config = OmegaConf.to_container(config, resolve=True)

    # Setup logging
    setup_logging(config["paths"]["log_dir"])
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Score2HuBERT Training")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
    if args.init_from:
        logger.info(f"Initializing from: {args.init_from} (model+EMA only)")

    # Print key config values
    logger.info(f"Model: encoder={config['model']['encoder']['n_layers']}L "
                f"d={config['model']['encoder']['d_model']}, "
                f"flow={config['model']['flow_matching']['n_layers']}L")
    logger.info(f"Training: lr={config['training']['optimizer']['lr']}, "
                f"batch={config['training']['batch_size']}, "
                f"steps={config['training']['max_steps']}")

    # Train
    trainer = Trainer(config, resume_from=args.resume, init_from=args.init_from)
    trainer.train()


if __name__ == "__main__":
    main()
