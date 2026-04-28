"""Phase 2 step-0 baseline validation.

Loads Plan C checkpoint via init_from path, builds the multi_lang_4 val loader,
runs _validate() ONCE without any training. Records baseline metrics.

Hard threshold: if val_recon_l1 > 0.30, STOP and debug — something is broken
(too far from Plan C's 0.118 to be explained by language-distribution mismatch alone).

Expected: val_recon_l1 ~0.13-0.18 (slightly worse than Plan C's 0.118 due to
new languages + speech samples + new val split membership).
"""
import sys
import yaml
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.stdout.reconfigure(encoding="utf-8")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from src.training.trainer import Trainer

CONFIG = "configs/finetune_multi_det.yaml"
INIT_FROM = "checkpoints/pretrain_opencpop_det/best.pt"

print("=" * 70)
print("PHASE 2 STEP-0 BASELINE VALIDATION")
print("=" * 70)
print(f"Config:    {CONFIG}")
print(f"Init from: {INIT_FROM}")
print()

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

# Use a sandbox dir to avoid creating spurious files in the real training dir
cfg["paths"]["checkpoint_dir"] = "checkpoints/_step0_val_multi"
cfg["paths"]["log_dir"] = "logs/_step0_val_multi"

# Force num_workers=0 for clean test (no worker process complications)
cfg["training"]["num_workers"] = 0
cfg["training"]["persistent_workers"] = False
if "prefetch_factor" in cfg["training"]:
    del cfg["training"]["prefetch_factor"]

trainer = Trainer(cfg, init_from=INIT_FROM)
print()
print("=" * 70)
print("Running trainer._validate() at step 0 (no training)")
print("=" * 70)

_, val_loader = trainer._build_dataloaders()
metrics = trainer._validate(val_loader)

print()
print("=" * 70)
print("STEP-0 BASELINE METRICS")
print("=" * 70)
print(f"  f0_mae_cents : {metrics.get('f0_mae_cents', float('nan')):.2f}")
print(f"  recon_l1     : {metrics.get('recon_l1', float('nan')):.4f}")
print(f"  recon_cos_sim: {metrics.get('recon_cos_sim', float('nan')):.4f}")
print(f"  total_loss   : {metrics.get('total_loss', float('nan')):.4f}")
print()
print("Plan C reference (Chinese-only val): l1=0.1183, cos=0.854, f0=44.6")
print()

# Hard threshold check
val_l1 = metrics.get("recon_l1", float("inf"))
if val_l1 > 0.30:
    print("XX  HARD STOP: val_recon_l1 > 0.30")
    print("    This is too far from Plan C baseline (0.118) to be language drift alone.")
    print("    Investigate: vocab mismatch, ckpt load issue, or data corruption.")
    sys.exit(1)
elif val_l1 > 0.20:
    print("WARN  val_recon_l1 in {0.20..0.30}: worse than expected but possibly OK if")
    print("      multilingual + speech samples create more variance. Proceed cautiously.")
else:
    print("OK  val_recon_l1 looks healthy. Safe to launch full Phase 2 training.")
