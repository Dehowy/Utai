#!/usr/bin/env python3
"""
Diagnostic: can a small deterministic head regress HuBERT features directly
from the frozen Score2HuBERT encoder output (frame_hidden + f0_emb)?

This answers a critical architectural question: does `frame_hidden` actually
contain enough phoneme identity information for the decoder to use?

If yes: a full Plan B architectural fix (adding cond-AdaLN to DiTBlock) has a
        real chance of working — the info is there, the flow decoder just
        wasn't wired to use it.
If no: the encoder or upstream is broken, and we need a bigger surgery.

Procedure:
- Load Plan A best.pt
- Freeze encoder, variance_adaptor, f0_predictor, f0_cond_proj
- Add a small trainable det_head: Linear → 3x (Conv1d + GN + GELU) → Linear
  (~2M params, directly regresses HuBERT from [frame_hidden | f0_emb])
- Train ONLY the det_head for N steps with L1 + 0.5*(1-cos) loss
- Print L1 and cos every 20 steps

Expected:
- If frame_hidden has phoneme info: L1 drops rapidly (0.30 -> 0.10), cos rises
  from ~0.45 toward 0.85+, clear trend in 200-500 steps.
- If frame_hidden lacks it: L1 plateaus around 0.20-0.25, cos stuck around
  0.55-0.65, no improving trend.

Usage:
    venv/Scripts/python.exe scripts/diag_det_head.py \\
        --model checkpoints/pretrain_opencpop/best.pt \\
        --steps 500
"""

import sys
import json
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from omegaconf import OmegaConf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.score2hubert import Score2HuBERT
from src.data.dataset import Score2HuBERTDataset
from src.data.collate import collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


class DetHead(nn.Module):
    """Small deterministic HuBERT head.

    Takes [frame_hidden; f0_emb] concat (2*d_model) and maps to HuBERT features.
    Intentionally small (~2M params) so training is fast and we can see if the
    signal is there with minimal capacity.
    """

    def __init__(self, d_in: int = 1024, d_out: int = 768, d_hidden: int = 512):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_hidden)
        self.blocks = nn.ModuleList()
        for _ in range(3):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(d_hidden, d_hidden, kernel_size=5, padding=2),
                nn.GroupNorm(8, d_hidden),
                nn.GELU(),
            ))
        self.out_proj = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_in] → [B, T, d_out]"""
        h = self.in_proj(x)                # [B, T, H]
        h = h.transpose(1, 2)              # [B, H, T] for conv
        for block in self.blocks:
            h = h + block(h)               # residual conv blocks
        h = h.transpose(1, 2)              # [B, T, H]
        return self.out_proj(h)            # [B, T, D_out]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/pretrain_opencpop.yaml")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Higher than main training — this head is small and fresh")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--log-every", type=int, default=20)
    # Audio generation after training (for perceptual evaluation)
    p.add_argument("--audio-output-dir", type=str, default=None,
                   help="If set, generate audio from 3 long test samples using the "
                        "trained det head (+ RVC) after training ends")
    p.add_argument("--rvc-model", type=str, default=None,
                   help="Path to RVC weights .pth (required with --audio-output-dir)")
    p.add_argument("--audio-num-samples", type=int, default=3)
    p.add_argument("--audio-min-frames", type=int, default=350)
    p.add_argument("--audio-seed", type=int, default=42,
                   help="Same default as eval_audio.py so samples match for comparison")
    return p.parse_args()


# -------------------- RVC helpers (inlined from eval_audio.py) --------------------


def load_rvc(rvc_model_path: str, device: str):
    """Load RVC's net_g for direct feature -> audio synthesis."""
    model_p = Path(rvc_model_path)
    rvc_root = None
    for parent in model_p.parents:
        if (parent / "infer").exists():
            rvc_root = parent
            break
    if rvc_root is None:
        raise RuntimeError(f"Can't find RVC root from {rvc_model_path}")

    rvc_infer = rvc_root / "infer"
    if str(rvc_infer) not in sys.path:
        sys.path.insert(0, str(rvc_infer))
    if str(rvc_root) not in sys.path:
        sys.path.insert(0, str(rvc_root))

    cpt = torch.load(rvc_model_path, map_location="cpu", weights_only=False)
    from lib.infer_pack.models import SynthesizerTrnMs768NSFsid

    sr = cpt.get("config", [0, 0, 0, 0, 0])[-1] or 40000
    config = cpt.get("config", None)
    if config:
        net_g = SynthesizerTrnMs768NSFsid(*config, is_half=False)
    else:
        net_g = SynthesizerTrnMs768NSFsid(
            1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2],
            512, [16, 16, 4, 4], 109, sr, 1,
        )
    net_g.load_state_dict(cpt.get("weight", {}), strict=False)
    net_g = net_g.to(device).eval()
    sid = torch.tensor([0], device=device)
    logger.info(f"Loaded RVC: {rvc_model_path} (sr={sr})")
    return net_g, sid, sr


def f0_to_coarse(f0_hz: torch.Tensor) -> torch.Tensor:
    """Convert F0 in Hz to RVC's coarse pitch index [1..255], 0 for unvoiced."""
    f0_mel = 1127.0 * torch.log(1.0 + f0_hz / 700.0)
    f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
    f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
    f0_mel_norm = (f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min)
    coarse = (f0_mel_norm * 254.0 + 1.0).long().clamp(1, 255)
    coarse[f0_hz < 50] = 0
    return coarse


@torch.no_grad()
def rvc_decode(net_g, sid, hubert_feat, f0_hz, device):
    """RVC interface matching pipeline.py:247 — feat upsampled 2x, NO permute, f0 2x."""
    T_min = min(hubert_feat.size(0), f0_hz.size(0))
    feat = hubert_feat[:T_min].unsqueeze(0).to(device).float()
    f0 = f0_hz[:T_min].to(device).float()

    feat_2x = F.interpolate(feat.permute(0, 2, 1), scale_factor=2, mode="nearest").permute(0, 2, 1)
    f0_2x = F.interpolate(f0.view(1, 1, -1), scale_factor=2, mode="linear", align_corners=False).view(-1)

    pitch_coarse = f0_to_coarse(f0_2x).unsqueeze(0)
    nsff0 = f0_2x.unsqueeze(0)
    p_len = torch.tensor([feat_2x.size(1)], device=device).long()

    audio = net_g.infer(feat_2x, p_len, pitch_coarse, nsff0, sid)[0][0, 0]
    return audio.float().cpu()


def save_audio(audio, sr, path):
    a = audio.numpy()
    a = a / (np.abs(a).max() + 1e-8) * 0.95
    sf.write(str(path), a, sr)


# -------------------- Main extraction + training pieces --------------------


@torch.no_grad()
def extract_conditions(model: Score2HuBERT, batch: dict) -> tuple:
    """
    Run the frozen encoder path to get (frame_hidden, f0_emb, hubert_target, frame_mask).

    Mirrors Score2HuBERT.forward() up through step 6 (condition construction),
    but without flow matching or reconstruction loss.
    """
    phonemes = batch["phonemes"]
    pitch = batch["pitch"]
    duration = batch["duration"]
    singing_ratio = batch["singing_ratio"]
    hubert_target = batch["hubert"]
    f0_target = batch["f0"]
    phone_mask = batch["phone_mask"]
    frame_mask = batch["frame_mask"]
    lang_id = batch["lang_id"]

    # 1. Encoder
    phone_hidden = model.encoder(
        phonemes, pitch, duration, singing_ratio,
        lang_id=lang_id, phone_mask=phone_mask,
    )

    # 2. Length regulation
    T_frame = hubert_target.size(1)
    frame_hidden = model.length_regulator(phone_hidden, duration, max_len=T_frame)

    # 3. Variance adaptor (eval mode, no targets needed here — just use predictions)
    var_targets = {}
    if "energy" in batch:
        var_targets["energy"] = batch["energy"]
    if "breathiness" in batch:
        var_targets["breathiness"] = batch["breathiness"]
    frame_hidden, _ = model.variance_adaptor(
        frame_hidden, targets=var_targets if var_targets else None, mask=frame_mask
    )

    # 4. F0 embedding (same as forward step 6)
    f0_log = torch.log(f0_target.clamp(min=1.0) + 1e-6).unsqueeze(-1)
    f0_emb = model.f0_cond_proj(f0_log)

    return frame_hidden, f0_emb, hubert_target, frame_mask


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info("Deterministic Head Diagnostic")
    logger.info("=" * 70)
    logger.info(f"Model:     {args.model}")
    logger.info(f"Steps:     {args.steps}")
    logger.info(f"Batch:     {args.batch_size}")
    logger.info(f"LR:        {args.lr}")

    # Load config
    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)

    # Load the frozen Score2HuBERT model
    logger.info("Loading model...")
    model = Score2HuBERT(config).to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)

    # Prefer EMA weights if present
    if "ema_shadow" in ckpt:
        state_dict = {}
        for name, param in model.named_parameters():
            if name in ckpt["ema_shadow"]:
                state_dict[name] = ckpt["ema_shadow"][name]
            else:
                state_dict[name] = ckpt["model_state"][name]
        for name, buf in model.named_buffers():
            if name in ckpt["model_state"]:
                state_dict[name] = ckpt["model_state"][name]
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"  Loaded EMA weights from step {ckpt.get('global_step', '?')}")
    else:
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"  Loaded live weights from step {ckpt.get('global_step', '?')}")

    model.eval()
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Build the trainable det head
    d_model = config["model"]["encoder"]["d_model"]
    d_hubert = config["model"]["flow_matching"]["d_hubert"]
    det_head = DetHead(d_in=2 * d_model, d_out=d_hubert, d_hidden=d_model).to(device)
    n_params = sum(p.numel() for p in det_head.parameters() if p.requires_grad)
    logger.info(f"DetHead params: {n_params/1e6:.2f}M "
                f"(d_in={2*d_model}, d_out={d_hubert}, d_hidden={d_model})")

    # Build dataloader
    data_cfg = config["data"]
    train_cfg = config["training"]
    dataset = Score2HuBERTDataset(
        processed_dir=data_cfg["processed_dir"],
        split="train",
        max_frames=data_cfg["max_frames"],
        min_frames=data_cfg["min_frames"],
        augmentation=data_cfg.get("augmentation"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=train_cfg.get("persistent_workers", False),
        prefetch_factor=train_cfg.get("prefetch_factor", 2),
    )

    # Optimizer — only det_head parameters
    optimizer = torch.optim.AdamW(det_head.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), weight_decay=0.01)
    scaler = GradScaler("cuda")

    data_iter = iter(loader)

    # Running stats (windowed over log_every)
    running_l1 = 0.0
    running_cos = 0.0
    running_init_l1 = 0.0  # L1 before any training, for baseline
    running_init_cos = 0.0
    log_count = 0

    logger.info("Training det head...")
    logger.info(f"{'step':>6} {'l1':>10} {'cos':>10} {'delta_l1':>12} {'delta_cos':>12}")
    logger.info("-" * 70)

    baseline_l1 = None
    baseline_cos = None

    for step in range(args.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Extract frozen conditions
        frame_hidden, f0_emb, hubert_target, frame_mask = extract_conditions(model, batch)

        # Forward det head
        with autocast("cuda", enabled=True):
            det_input = torch.cat([frame_hidden, f0_emb], dim=-1)  # [B, T, 2*d_model]
            pred_hubert = det_head(det_input)

            # Compute masked L1 and cosine
            mask = frame_mask.unsqueeze(-1).float()      # [B, T, 1]
            l1 = (torch.abs(pred_hubert - hubert_target) * mask).sum() / (mask.sum() * d_hubert)
            cos_sim = F.cosine_similarity(pred_hubert, hubert_target, dim=-1)  # [B, T]
            mask_f = frame_mask.float()
            cos = (cos_sim * mask_f).sum() / mask_f.sum()
            loss = l1 + 0.5 * (1 - cos)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(det_head.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_l1 += l1.item()
        running_cos += cos.item()
        log_count += 1

        # Baseline = first step's pre-training values
        if baseline_l1 is None:
            baseline_l1 = l1.item()
            baseline_cos = cos.item()

        if (step + 1) % args.log_every == 0:
            avg_l1 = running_l1 / log_count
            avg_cos = running_cos / log_count
            delta_l1 = avg_l1 - baseline_l1
            delta_cos = avg_cos - baseline_cos
            logger.info(
                f"{step+1:>6} {avg_l1:>10.4f} {avg_cos:>10.4f} "
                f"{delta_l1:>+12.4f} {delta_cos:>+12.4f}"
            )
            running_l1 = 0.0
            running_cos = 0.0
            log_count = 0

    logger.info("-" * 70)
    logger.info("Diagnostic training complete.")

    # -------------------- Optional audio generation --------------------
    if args.audio_output_dir:
        if args.rvc_model is None:
            raise ValueError("--audio-output-dir requires --rvc-model")

        logger.info("")
        logger.info("=" * 70)
        logger.info("Generating audio from trained det head")
        logger.info("=" * 70)

        out_dir = Path(args.audio_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load RVC
        net_g, sid, sr = load_rvc(args.rvc_model, device)

        # Pick same long samples as eval_audio.py (same seed, same min_frames)
        rng = random.Random(args.audio_seed)
        proc = Path(data_cfg["processed_dir"])
        with open(proc / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        test = [m for m in meta if m.get("split") == "test"]
        candidates = [m for m in test
                      if args.audio_min_frames <= m.get("n_frames", 0) <= 400]
        if len(candidates) < args.audio_num_samples:
            picks = candidates
        else:
            rng.shuffle(candidates)
            picks = candidates[:args.audio_num_samples]
        picks.sort(key=lambda m: -m["n_frames"])

        logger.info(f"Picked {len(picks)} samples (n_frames "
                    f"{picks[-1]['n_frames']}-{picks[0]['n_frames']})")

        det_head.eval()
        for i, m in enumerate(picks, start=1):
            sid_str = m["id"]
            n_frames = m["n_frames"]
            logger.info(f"[{i}/{len(picks)}] {sid_str} ({n_frames} frames "
                        f"≈ {n_frames/50:.1f}s)")

            # Load preprocessed inputs
            phonemes = torch.load(proc / "phonemes" / f"{sid_str}.pt", weights_only=True).long()
            pitch = torch.load(proc / "pitch" / f"{sid_str}.pt", weights_only=True).long()
            duration = torch.load(proc / "duration" / f"{sid_str}.pt", weights_only=True).long()
            gt_hubert = torch.load(proc / "hubert" / f"{sid_str}.pt", weights_only=True).float()
            gt_f0 = torch.load(proc / "f0" / f"{sid_str}.pt", weights_only=True).float()

            # Build a fake one-sample "batch" and push through extract_conditions
            batch_one = {
                "phonemes": phonemes.unsqueeze(0).to(device),
                "pitch": pitch.unsqueeze(0).to(device),
                "duration": duration.unsqueeze(0).to(device),
                "singing_ratio": torch.tensor([m.get("singing_ratio", 1.0)],
                                              dtype=torch.float32, device=device),
                "lang_id": torch.tensor([m.get("lang_id", 0)], dtype=torch.long, device=device),
                "hubert": gt_hubert.unsqueeze(0).to(device),
                "f0": gt_f0.unsqueeze(0).to(device),
                "phone_mask": torch.ones(1, phonemes.size(0), dtype=torch.bool, device=device),
                "frame_mask": torch.ones(1, gt_hubert.size(0), dtype=torch.bool, device=device),
            }

            frame_hidden, f0_emb, _, _ = extract_conditions(model, batch_one)

            # Det head inference
            with torch.no_grad():
                det_input = torch.cat([frame_hidden, f0_emb], dim=-1)
                pred_hubert = det_head(det_input)[0]  # [T, 768]

            # Generate 3 audio variants:
            # - gt_ceiling: GT hubert + GT f0 (baseline)
            # - det_pred_full: det head hubert + GT f0 (pure det head, isolates content)
            # - det_pred_predf0: det head hubert + pred f0 (not implemented — no f0 predictor run)
            # Keep it simple: just gt_ceiling + det_pred for direct comparison
            T_common = min(pred_hubert.size(0), gt_hubert.size(0), gt_f0.size(0))

            variants = {
                "gt_ceiling":  (gt_hubert[:T_common],  gt_f0[:T_common]),
                "det_pred":    (pred_hubert[:T_common], gt_f0[:T_common]),
            }
            for name, (h, f) in variants.items():
                audio = rvc_decode(net_g, sid, h, f, device)
                out_path = out_dir / f"sample{i}_{name}.wav"
                save_audio(audio, sr, out_path)
                logger.info(f"    saved {out_path.name} ({len(audio)/sr:.2f}s)")

        logger.info("")
        logger.info(f"Audio outputs in: {out_dir}")

    logger.info("")
    logger.info("Interpretation guide:")
    logger.info("  L1/cos: L1 drop ≥ 0.10 and cos rise ≥ 0.10 → frame_hidden HAS phoneme info")
    logger.info("          → Plan B (architectural cond-AdaLN) is viable")
    logger.info("  Audio:  if det_pred has audible lyrics → frame_hidden info is USABLE for content")
    logger.info("          → commit to Plan B")
    logger.info("          if det_pred is also '啊啊啊' → encoder needs upgrade, Plan B won't help")


if __name__ == "__main__":
    main()
