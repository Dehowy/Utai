#!/usr/bin/env python3
"""
Diagnostic: build a Trainer in init_from mode and immediately call _validate
WITHOUT any training. This should be functionally identical to "training step 0
val" and tells us what trainer val produces with no drift at all.

If this gives f0_mae ≈ 47 cents → bug introduced by training step drift
If this gives f0_mae ≈ 104 cents → bug in Trainer init / EMA path itself
"""
import sys, yaml, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import Trainer

with open("configs/pretrain_opencpop_det.yaml") as f:
    cfg = yaml.safe_load(f)

# Use a separate ckpt dir (not used for anything, but trainer wants one)
cfg["paths"]["checkpoint_dir"] = "checkpoints/_diag_trainer_val"
cfg["paths"]["log_dir"] = "logs/_diag_trainer_val"

# Force num_workers=0 for clean test (no worker process complications)
cfg["training"]["num_workers"] = 0
cfg["training"]["persistent_workers"] = False
if "prefetch_factor" in cfg["training"]:
    del cfg["training"]["prefetch_factor"]

trainer = Trainer(cfg, init_from="checkpoints/pretrain_opencpop/best.pt")
print()
print("=" * 70)
print("Calling trainer._validate() at step 0 (no training)")
print("=" * 70)

# Build dataloader the trainer way
_, val_loader = trainer._build_dataloaders()
metrics = trainer._validate(val_loader)

print()
print(f"f0_mae_cents : {metrics.get('f0_mae_cents'):.2f}")
print(f"recon_l1     : {metrics.get('recon_l1'):.4f}")
print(f"recon_cos_sim: {metrics.get('recon_cos_sim'):.4f}")
print(f"total_loss   : {metrics.get('total_loss'):.4f}")
