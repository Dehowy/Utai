#!/usr/bin/env python3
"""
Diagnostic: load a training checkpoint and compare LIVE vs EMA val metrics.
The training checkpoint contains both model_state (live) and ema_shadow.

We expect:
  - LIVE val: should give ~Plan A f0_mae (~47) and reasonable det_head L1
  - EMA val: depends on how badly EMA blend is corrupting things

If LIVE >> EMA (i.e., live is much better), the trainer's EMA-based val
is misleading and we should switch to live-based val (or reset EMA).
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
from torch.amp import autocast

CKPT = "checkpoints/pretrain_opencpop_det/step_5000.pt"
CFG_PATH = "configs/pretrain_opencpop_det.yaml"
DEVICE = "cuda"
N_BATCHES = 156  # ~full val set at batch=32

print(f"Loading {CKPT}")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
print(f"  global_step: {ckpt.get('global_step')}")
print(f"  best_val_l1: {ckpt.get('best_val_l1', 'N/A')}")

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

model = Score2HuBERT(cfg).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
for p in model.flow_decoder.parameters():
    p.requires_grad = False

# Build val loader
val_dataset = Score2HuBERTDataset(
    processed_dir=cfg["data"]["processed_dir"],
    split="val",
    max_frames=cfg["data"]["max_frames"],
    min_frames=cfg["data"]["min_frames"],
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=0,
    collate_fn=collate_fn, pin_memory=False,
)

def run_val(model, loader, max_batches, label):
    metrics_sum = defaultdict(float)
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            var_targets = {}
            if "energy" in batch: var_targets["energy"] = batch["energy"]
            if "breathiness" in batch: var_targets["breathiness"] = batch["breathiness"]
            with autocast("cuda", enabled=True):
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
    res = {k: v / max(1, n) for k, v in metrics_sum.items()}
    print(f"\n[{label}] {n} batches:")
    print(f"  f0_mae       : {res.get('f0_mae_cents'):>8.2f} cents")
    print(f"  recon_l1     : {res.get('recon_l1'):>8.4f}")
    print(f"  recon_cos_sim: {res.get('recon_cos_sim'):>8.4f}")
    print(f"  total_loss   : {res.get('total_loss'):>8.4f}")
    return res

# Path A: use live (model_state) weights
print("\n" + "=" * 70)
print("[A] LIVE weights (no EMA)")
print("=" * 70)
res_live = run_val(model, val_loader, N_BATCHES, "LIVE")

# Path B: apply EMA shadow then val
print("\n" + "=" * 70)
print("[B] EMA-applied weights")
print("=" * 70)
backup = {}
n_applied = 0
for name, param in model.named_parameters():
    if param.requires_grad and name in ckpt["ema_shadow"]:
        backup[name] = param.data.clone()
        param.data = ckpt["ema_shadow"][name].to(DEVICE)
        n_applied += 1
print(f"  applied EMA to {n_applied} params")
res_ema = run_val(model, val_loader, N_BATCHES, "EMA")

# Restore
for name, val in backup.items():
    dict(model.named_parameters())[name].data = val

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"  f0_mae:    LIVE {res_live.get('f0_mae_cents'):.1f} vs EMA {res_ema.get('f0_mae_cents'):.1f} cents")
print(f"  recon_l1:  LIVE {res_live.get('recon_l1'):.4f} vs EMA {res_ema.get('recon_l1'):.4f}")
print(f"  recon_cos: LIVE {res_live.get('recon_cos_sim'):.4f} vs EMA {res_ema.get('recon_cos_sim'):.4f}")
print()
print("If LIVE is much better than EMA, switch to live-based val.")
print("If they're similar, the val number is just slow EMA convergence.")
