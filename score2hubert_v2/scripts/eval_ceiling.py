"""Quick GT ceiling test — no model needed.

Loads preprocessed (HuBERT, F0, phonemes, duration) for sample audios from each
specified dataset, runs them directly through RVC, saves WAV.

If ceiling sounds clean → data is good
If ceiling sounds muddy/garbled → data quality issue (GT-style contamination)

Usage:
    python scripts/eval_ceiling.py --datasets jvs_music oniku kiritan \
        --output-dir test_outputs/ceiling_audit --num-samples 4
"""
import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_rvc(rvc_model_path: str, device: str):
    model_p = Path(rvc_model_path)
    rvc_root = None
    for parent in model_p.parents:
        if (parent / "infer").exists():
            rvc_root = parent
            break
    for p in [rvc_root / "infer", rvc_root]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    cpt = torch.load(rvc_model_path, map_location="cpu", weights_only=False)
    from lib.infer_pack.models import SynthesizerTrnMs768NSFsid
    cfg = cpt.get("config", None)
    sr = cfg[-1] if cfg else 40000
    net_g = SynthesizerTrnMs768NSFsid(*cfg, is_half=False)
    net_g.load_state_dict(cpt.get("weight", {}), strict=False)
    net_g = net_g.to(device).eval()
    sid = torch.tensor([0], device=device)
    logger.info(f"Loaded RVC: {rvc_model_path} (sr={sr})")
    return net_g, sid, sr


def f0_to_coarse(f0_hz: torch.Tensor) -> torch.Tensor:
    f0_mel = 1127.0 * torch.log(1.0 + f0_hz / 700.0)
    f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
    f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
    f0_mel_norm = (f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min)
    coarse = (f0_mel_norm * 254.0 + 1.0).long().clamp(1, 255)
    coarse[f0_hz < 50] = 0
    return coarse


@torch.no_grad()
def rvc_decode(net_g, sid, hubert_feat, f0_hz, device):
    T = min(hubert_feat.size(0), f0_hz.size(0))
    feat = hubert_feat[:T].unsqueeze(0).to(device).float()
    f0 = f0_hz[:T].to(device).float()
    feat_2x = F.interpolate(feat.permute(0, 2, 1), scale_factor=2, mode="nearest").permute(0, 2, 1)
    f0_2x = F.interpolate(f0.view(1, 1, -1), scale_factor=2, mode="linear", align_corners=False).view(-1)
    pitch_coarse = f0_to_coarse(f0_2x).unsqueeze(0)
    nsff0 = f0_2x.unsqueeze(0)
    p_len = torch.tensor([feat_2x.size(1)], device=device).long()
    audio = net_g.infer(feat_2x, p_len, pitch_coarse, nsff0, sid)[0][0, 0]
    return audio.float().cpu()


def save_audio(audio: torch.Tensor, sr: int, path: Path):
    a = audio.numpy()
    a = a / (np.abs(a).max() + 1e-8) * 0.95
    sf.write(str(path), a, sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="dataset names under data/processed/ to test")
    ap.add_argument("--output-dir", default="test_outputs/ceiling_audit")
    ap.add_argument("--rvc-model",
                    default="RVC/RVC20240604Nvidia/assets/weights/yachiyo_runami_e350_s10500.pth")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--min-frames", type=int, default=200)
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    net_g, sid, sr = load_rvc(args.rvc_model, args.device)

    for ds_name in args.datasets:
        ds_dir = REPO_ROOT / "data" / "processed" / ds_name
        if not ds_dir.exists():
            logger.warning(f"{ds_name}: not found")
            continue

        with open(ds_dir / "metadata.json", encoding="utf-8") as f:
            meta = json.load(f)

        # Filter to test split if exists, in frame range
        test_entries = [e for e in meta if e.get("split") == "test"]
        candidates = [e for e in test_entries
                      if args.min_frames <= e["n_frames"] <= args.max_frames]
        if len(candidates) < args.num_samples:
            # fallback: use all
            candidates = [e for e in meta
                          if args.min_frames <= e["n_frames"] <= args.max_frames]

        # Pick longest N for clearer listening
        candidates.sort(key=lambda e: -e["n_frames"])
        picks = candidates[:args.num_samples]

        out_dir = out_root / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n=== {ds_name}: {len(picks)} samples ===")
        for i, m in enumerate(picks, 1):
            sid_str = m["id"]
            n_f = m["n_frames"]
            try:
                hubert = torch.load(ds_dir / "hubert" / f"{sid_str}.pt", weights_only=True).float()
                f0 = torch.load(ds_dir / "f0" / f"{sid_str}.pt", weights_only=True).float()
            except Exception as e:
                logger.warning(f"  [{i}] {sid_str}: load failed: {e}")
                continue

            audio = rvc_decode(net_g, sid, hubert, f0, args.device)
            out_path = out_dir / f"{sid_str}_ceiling.wav"
            save_audio(audio, sr, out_path)
            logger.info(f"  [{i}] {sid_str} ({n_f}f / {n_f/50:.1f}s): {out_path.name}")

    logger.info(f"\nDone. Listen to: {out_root}")


if __name__ == "__main__":
    main()
