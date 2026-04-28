#!/usr/bin/env python3
"""
Diagnostic: verify init_from + EMA merge actually loads Plan A's encoder/f0
state correctly. Runs on CPU to not interfere with training.

Loads Plan A best.pt the same way trainer does, manually applies EMA,
then runs forward on a few val batches and reports f0_mae.

Expected (if init_from works correctly):
- f0_mae ≈ Plan A's 47.3 cents (proves encoder/f0 EMA loaded)

If we instead see ~104 cents: there's a real bug in init_from / EMA
shadow merge / EMA.apply path that needs fixing before training continues.
"""
import sys, yaml, torch
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.score2hubert import Score2HuBERT
from src.data.dataset import Score2HuBERTDataset
from src.data.collate import collate_fn
from torch.utils.data import DataLoader

CKPT = "checkpoints/pretrain_opencpop/best.pt"
CFG_PATH = "configs/pretrain_opencpop_det.yaml"
import os
DEVICE = os.environ.get("DIAG_DEVICE", "cpu")
USE_AUTOCAST = os.environ.get("DIAG_AUTOCAST", "0") == "1"
N_BATCHES = int(os.environ.get("DIAG_BATCHES", "5"))

print("=" * 70)
print("EMA Load Diagnostic")
print("=" * 70)

# Load config (det mode — same as training)
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

# Build model
model = Score2HuBERT(cfg).to(DEVICE)
model.eval()

# Freeze flow_decoder (matches trainer behavior)
for p in model.flow_decoder.parameters():
    p.requires_grad = False

# Load checkpoint
print(f"\nLoading {CKPT}")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
print(f"  global_step: {ckpt.get('global_step')}")
print(f"  best_val_loss (legacy): {ckpt.get('best_val_loss')}")

# Load model state with strict=False (matches init_from)
missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
print(f"  loaded model_state: missing={len(missing)}, unexpected={len(unexpected)}")
print(f"  missing keys (first 5): {missing[:5]}")

# === Path A: forward without applying EMA (uses LIVE Plan A weights) ===
print("\n[A] Forward with LIVE weights (loaded from model_state)")

# Build a small val loader
val_dataset = Score2HuBERTDataset(
    processed_dir=cfg["data"]["processed_dir"],
    split="val",
    max_frames=cfg["data"]["max_frames"],
    min_frames=cfg["data"]["min_frames"],
)
DIAG_BATCH_SIZE = int(os.environ.get("DIAG_BATCH_SIZE", "8"))
val_loader = DataLoader(
    val_dataset, batch_size=DIAG_BATCH_SIZE, shuffle=False, num_workers=0,
    collate_fn=collate_fn, pin_memory=False,
)
print(f"  using batch_size={DIAG_BATCH_SIZE}")

def run_val(model, loader, max_batches):
    model.eval()
    metrics_sum = defaultdict(float)
    n = 0
    from torch.amp import autocast
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            var_targets = {}
            if "energy" in batch: var_targets["energy"] = batch["energy"]
            if "breathiness" in batch: var_targets["breathiness"] = batch["breathiness"]
            ctx = autocast("cuda", enabled=(USE_AUTOCAST and DEVICE == "cuda"))
            with ctx:
                out = model(
                    phonemes=batch["phonemes"], pitch=batch["pitch"],
                    duration=batch["duration"], singing_ratio=batch["singing_ratio"],
                    hubert_target=batch["hubert"], f0_target=batch["f0"],
                    phone_mask=batch["phone_mask"], frame_mask=batch["frame_mask"],
                    lang_id=batch["lang_id"],
                    variance_targets=var_targets if var_targets else None,
                )
            m = out.get("metrics", {})
            m["total_loss"] = out["loss"].item()
            for k, v in m.items():
                metrics_sum[k] += v
            n += 1
    return {k: v / max(1, n) for k, v in metrics_sum.items()}

print("  running 5 val batches with LIVE weights...")
metrics_live = run_val(model, val_loader, N_BATCHES)
print(f"  f0_mae      : {metrics_live.get('f0_mae_cents'):>8.2f} cents")
print(f"  recon_l1    : {metrics_live.get('recon_l1'):>8.4f}")
print(f"  recon_cos   : {metrics_live.get('recon_cos_sim'):>8.4f}")
print(f"  total_loss  : {metrics_live.get('total_loss'):>8.4f}")

# === Path B: forward with EMA applied (uses Plan A's EMA shadow) ===
print("\n[B] Forward with EMA shadow applied (the path val uses)")

# Apply EMA shadow manually (matches what trainer.EMA does)
if "ema_shadow" in ckpt:
    backup = {}
    n_applied = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in ckpt["ema_shadow"]:
            backup[name] = param.data.clone()
            param.data = ckpt["ema_shadow"][name].to(DEVICE)
            n_applied += 1
    print(f"  applied EMA to {n_applied} params")
else:
    print("  WARNING: no ema_shadow in ckpt!")

print("  running 5 val batches with EMA weights...")
metrics_ema = run_val(model, val_loader, N_BATCHES)
print(f"  f0_mae      : {metrics_ema.get('f0_mae_cents'):>8.2f} cents")
print(f"  recon_l1    : {metrics_ema.get('recon_l1'):>8.4f}")
print(f"  recon_cos   : {metrics_ema.get('recon_cos_sim'):>8.4f}")
print(f"  total_loss  : {metrics_ema.get('total_loss'):>8.4f}")

# Restore live weights (cleanup)
for name, val in backup.items():
    dict(model.named_parameters())[name].data = val

print("\n" + "=" * 70)
print("Interpretation:")
print(f"  Plan A's recorded f0_mae at best.pt: 47.3 cents")
print(f"  Live (model_state) f0_mae:            {metrics_live.get('f0_mae_cents'):.1f} cents")
print(f"  EMA (ema_shadow)   f0_mae:            {metrics_ema.get('f0_mae_cents'):.1f} cents")
print()
print("  - If EMA f0_mae ≈ 47: init_from works, training val anomaly is")
print("    purely the det_head EMA cold-start affecting cosine somehow")
print("  - If EMA f0_mae >> 47: init_from is buggy, need to fix before continuing")
print("  - If LIVE f0_mae ≈ 47 but EMA differs: EMA shadow is corrupted")
