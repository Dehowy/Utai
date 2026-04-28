#!/usr/bin/env python3
"""
Generate ablation audio outputs for a Score2HuBERT checkpoint.

For each picked test sample, produces 4 audio variants by combining
ground-truth and predicted HuBERT/F0:

  - gt_ceiling : GT HuBERT  + GT F0    → RVC ceiling (best possible RVC quality)
  - pred_full  : pred HuBERT + pred F0 → full Score2HuBERT pipeline
  - h_pred_f0gt: pred HuBERT + GT F0   → isolates HuBERT prediction quality
  - hgt_f0pred : GT HuBERT  + pred F0  → isolates F0 prediction quality

If `pred_full` and `h_pred_f0gt` sound similar to `gt_ceiling`, the model
works. If `h_pred_f0gt` is humming-without-lyrics, HuBERT prediction lost
phonetic content. If `hgt_f0pred` is bad, F0 prediction broke.

Uses the SAME RVC inference interface as RVC's official pipeline.py:
  feat is upsampled 2x to 100fps (RVC's expected internal rate),
  passed as [B, T, 768] (NOT permuted), with f0 also at 100fps.

Usage:
    venv/Scripts/python.exe scripts/eval_audio.py \\
        --model checkpoints/pretrain_opencpop/best.pt \\
        --rvc-model RVC/RVC20240604Nvidia/assets/weights/yachiyo_runami.pth \\
        --output-dir test_outputs_plan_a \\
        --num-samples 5 \\
        --min-frames 300 \\
        --steps 4 --solver euler
"""

import sys
import argparse
import json
import random
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.score2hubert import Score2HuBERT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="Score2HuBERT checkpoint path")
    p.add_argument("--rvc-model", type=str, required=True,
                   help="RVC weights .pth")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--processed-dir", type=str,
                   default="data/processed/ace_opencpop")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--min-frames", type=int, default=300,
                   help="Pick samples with at least this many HuBERT frames "
                        "(50fps). 300 ≈ 6 seconds.")
    p.add_argument("--max-frames", type=int, default=400,
                   help="Upper bound — leave room for the longest samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=4,
                   help="Flow matching ODE steps (Run 1 found 4 best by metric)")
    p.add_argument("--solver", type=str, default="euler",
                   choices=["euler", "midpoint", "rk4"])
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="CFG guidance (Run 1 found 1.0 best — Plan A may differ)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_model(model_path: str, device: str) -> Tuple[Score2HuBERT, dict]:
    """Load Score2HuBERT with EMA weights if present.

    Handles checkpoints from older training runs that lack newer modules
    (e.g., loading a Plan A flow-mode ckpt into a model that now has det_head).
    Missing keys keep their fresh PyTorch initialization.
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = Score2HuBERT(config).to(device)

    if "ema_shadow" in ckpt:
        state_dict = {}
        for name, param in model.named_parameters():
            if name in ckpt["ema_shadow"]:
                state_dict[name] = ckpt["ema_shadow"][name]
            elif name in ckpt["model_state"]:
                state_dict[name] = ckpt["model_state"][name]
            # else: missing in both — keep fresh init by skipping
        for name, buf in model.named_buffers():
            if name in ckpt["model_state"]:
                state_dict[name] = ckpt["model_state"][name]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(f"  {len(missing)} keys kept at fresh init: {missing[:3]}...")
        logger.info(f"Loaded EMA weights from {model_path}")
    else:
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            logger.info(f"  {len(missing)} keys kept at fresh init: {missing[:3]}...")
        logger.info(f"Loaded model weights from {model_path}")

    model.eval()
    # New ckpts use best_val_l1; fall back to old best_val_loss for backwards compat
    best_val = ckpt.get('best_val_l1', ckpt.get('best_val_loss', '?'))
    logger.info(f"  step={ckpt.get('global_step', '?')}, "
                f"mode={ckpt.get('config', {}).get('model', {}).get('mode', 'flow')}, "
                f"best_val_l1={best_val}")
    return model, config


def load_rvc(rvc_model_path: str, device: str):
    """Load RVC's net_g for direct feature → audio synthesis."""
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
def rvc_decode(
    net_g,
    sid: torch.Tensor,
    hubert_feat: torch.Tensor,  # [T, 768] at 50fps
    f0_hz: torch.Tensor,        # [T] in Hz, at 50fps
    device: str,
) -> torch.Tensor:
    """
    Direct RVC decoding matching the official pipeline.py interface:
      - feat upsampled 2x to 100fps
      - feat passed as [B, T, 768] (NOT permuted)
      - f0 also upsampled 2x to match feat length
      - pitch_coarse and nsff0 derived from upsampled f0
    """
    # Truncate to common length first to avoid size mismatch
    T_min = min(hubert_feat.size(0), f0_hz.size(0))
    feat = hubert_feat[:T_min].unsqueeze(0).to(device).float()  # [1, T, 768]
    f0 = f0_hz[:T_min].to(device).float()                       # [T]

    # Upsample feat 2x: [1, T, 768] → [1, 2T, 768]
    feat_2x = F.interpolate(
        feat.permute(0, 2, 1), scale_factor=2, mode="nearest"
    ).permute(0, 2, 1)

    # Upsample F0 2x linearly: [T] → [2T]
    f0_2x = F.interpolate(
        f0.view(1, 1, -1), scale_factor=2, mode="linear", align_corners=False
    ).view(-1)

    pitch_coarse = f0_to_coarse(f0_2x).unsqueeze(0)  # [1, 2T]
    nsff0 = f0_2x.unsqueeze(0)                       # [1, 2T]
    p_len = torch.tensor([feat_2x.size(1)], device=device).long()

    # NOTE: feat passed as [B, T, 768], NOT permuted to [B, 768, T]
    audio = net_g.infer(feat_2x, p_len, pitch_coarse, nsff0, sid)[0][0, 0]
    return audio.float().cpu()


def save_audio(audio: torch.Tensor, sr: int, path: Path):
    """Normalize and save WAV."""
    a = audio.numpy()
    a = a / (np.abs(a).max() + 1e-8) * 0.95
    sf.write(str(path), a, sr)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model + RVC
    model, config = load_model(args.model, device)
    net_g, sid, sr = load_rvc(args.rvc_model, device)

    # Load test split metadata and pick long samples
    proc = Path(args.processed_dir)
    with open(proc / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    test = [m for m in meta if m.get("split") == "test"]
    candidates = [
        m for m in test
        if args.min_frames <= m.get("n_frames", 0) <= args.max_frames
    ]
    if len(candidates) < args.num_samples:
        logger.warning(
            f"Only {len(candidates)} samples have {args.min_frames}-{args.max_frames} frames; "
            f"requested {args.num_samples}. Lower --min-frames if needed."
        )
        picks = candidates
    else:
        random.shuffle(candidates)
        picks = candidates[:args.num_samples]
    picks.sort(key=lambda m: -m["n_frames"])  # longest first

    logger.info(f"Picked {len(picks)} samples (n_frames range: "
                f"{picks[-1]['n_frames']}-{picks[0]['n_frames']}, "
                f"≈{picks[-1]['n_frames']/50:.1f}-{picks[0]['n_frames']/50:.1f}s)")

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

        # Phase 3: load GT tech flags if present; else zeros
        tech_path = proc / "tech" / f"{sid_str}.pt"
        if tech_path.exists():
            tech = torch.load(tech_path, weights_only=True).float()
        else:
            tech = torch.zeros(phonemes.size(0), 6, dtype=torch.float32)

        # Add batch dim
        phonemes_b = phonemes.unsqueeze(0).to(device)
        pitch_b = pitch.unsqueeze(0).to(device)
        duration_b = duration.unsqueeze(0).to(device)
        sr_b = torch.tensor([m.get("singing_ratio", 1.0)], dtype=torch.float32).to(device)
        lang_b = torch.tensor([m.get("lang_id", 0)], dtype=torch.long).to(device)
        tech_b = tech.unsqueeze(0).to(device)  # [1, T_phone, 6]

        # Run model.synthesize() — use predicted F0 and predicted HuBERT
        with torch.no_grad():
            output = model.synthesize(
                phonemes=phonemes_b,
                pitch=pitch_b,
                duration=duration_b,
                singing_ratio=sr_b,
                lang_id=lang_b,
                n_steps=args.steps,
                solver=args.solver,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                tech=tech_b,
            )
        pred_hubert = output["hubert"][0].float()  # [T_pred, 768]
        pred_f0 = output["f0"][0].float()          # [T_pred]

        # Some samples may have slight length difference — truncate to common length
        T_common = min(pred_hubert.size(0), pred_f0.size(0),
                       gt_hubert.size(0), gt_f0.size(0))
        pred_hubert = pred_hubert[:T_common]
        pred_f0 = pred_f0[:T_common]
        gt_h = gt_hubert[:T_common]
        gt_f = gt_f0[:T_common]

        # Generate 4 ablations
        variants = {
            "gt_ceiling":  (gt_h,        gt_f),
            "pred_full":   (pred_hubert, pred_f0),
            "h_pred_f0gt": (pred_hubert, gt_f),
            "hgt_f0pred":  (gt_h,        pred_f0),
        }
        for name, (h, f) in variants.items():
            audio = rvc_decode(net_g, sid, h, f, device)
            out_path = out_dir / f"sample{i}_{name}.wav"
            save_audio(audio, sr, out_path)
            logger.info(f"    saved {out_path.name} "
                        f"({len(audio)/sr:.2f}s)")

    logger.info(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
