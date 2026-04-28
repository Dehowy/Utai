"""
RVC Compensation Analysis and Feature Adjustment.

Problem:
  When our generated HuBERT features are fed into RVC's decoder (net_g),
  certain fine details may be smoothed or lost due to the decoder's
  learned response characteristics.

Solution:
  1. Extract HuBERT features H_gt from ground truth audio
  2. Feed H_gt + F0 into RVC's net_g → reconstructed audio
  3. Re-extract HuBERT from reconstructed audio → H_roundtrip
  4. Compute offset: Δ = H_gt - H_roundtrip (what net_g drops)
  5. Adjusted target: H_target = H_gt + α * Δ (compensate for smoothing)

This makes our model "over-express" details that net_g tends to lose,
so the final output after RVC is closer to the intended features.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import logging
import json

logger = logging.getLogger(__name__)


class RVCCompensationAnalyzer:
    """
    Analyze RVC's feature transformation characteristics and compute
    compensation offsets for training data adjustment.
    """

    def __init__(
        self,
        rvc_model_path: str,
        hubert_model_path: str,
        rvc_config_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device
        self.rvc_model_path = rvc_model_path
        self.hubert_model_path = hubert_model_path

        # Load RVC model components
        self._load_rvc(rvc_model_path, rvc_config_path)

        # Load HuBERT for re-extraction
        from .extract_hubert import HuBERTExtractor
        self.hubert_extractor = HuBERTExtractor(
            hubert_model_path, layer=12, device=device
        )

    def _load_rvc(self, model_path: str, config_path: Optional[str] = None):
        """Load RVC's synthesizer (net_g) for roundtrip analysis."""
        import sys

        # Determine RVC root and add to path
        model_p = Path(model_path)
        # Try to find RVC root from model path
        rvc_root = None
        for parent in model_p.parents:
            if (parent / "infer").exists():
                rvc_root = parent
                break

        if rvc_root is None:
            raise RuntimeError(f"Cannot find RVC root from {model_path}")

        rvc_infer = rvc_root / "infer"
        if str(rvc_infer) not in sys.path:
            sys.path.insert(0, str(rvc_infer))

        # Load model checkpoint
        cpt = torch.load(model_path, map_location="cpu", weights_only=False)

        # Determine model version and config
        tgt_sr = cpt.get("config", [0, 0, 0, 0, 0])[-1]
        if tgt_sr == 0:
            tgt_sr = 40000  # default

        self.tgt_sr = tgt_sr
        self.version = cpt.get("version", "v2")

        # Load synthesizer
        from lib.infer_pack.models import SynthesizerTrnMs768NSFsid

        if_f0 = cpt.get("f0", 1)
        config = cpt.get("config", None)

        if config:
            net_g = SynthesizerTrnMs768NSFsid(
                *config, is_half=False
            )
        else:
            # Default V2 config
            net_g = SynthesizerTrnMs768NSFsid(
                1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2],
                512, [16, 16, 4, 4], 109, tgt_sr, 1,
            )

        # Load weights
        state_dict = cpt.get("weight", {})
        net_g.load_state_dict(state_dict, strict=False)
        net_g = net_g.to(self.device).eval()

        self.net_g = net_g
        self.sid = torch.tensor([0], device=self.device)

        logger.info(f"Loaded RVC model: version={self.version}, sr={tgt_sr}")

    @torch.no_grad()
    def roundtrip(
        self,
        hubert_features: torch.Tensor,  # [T, 768]
        f0: torch.Tensor,               # [T] Hz
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform roundtrip: HuBERT features → RVC decoder → audio → HuBERT features.

        Returns:
            audio: reconstructed audio tensor
            h_roundtrip: re-extracted HuBERT features [T', 768]
        """
        T = hubert_features.size(0)

        # Prepare features for RVC decoder
        # RVC upsamples features 2x: [T, 768] → interpolate → [2T, 768]
        feat = hubert_features.unsqueeze(0).to(self.device)  # [1, T, 768]
        feat = F.interpolate(
            feat.permute(0, 2, 1),  # [1, 768, T]
            scale_factor=2,
            mode="nearest",
        ).permute(0, 2, 1)  # [1, 2T, 768]

        # Prepare F0
        f0_hz = f0.to(self.device).float()

        # Convert F0 to coarse (RVC format)
        f0_mel = 1127.0 * torch.log(1.0 + f0_hz / 700.0)
        f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
        f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
        f0_mel_norm = (f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min)
        f0_coarse = (f0_mel_norm * 254.0 + 1.0).long().clamp(1, 255)
        f0_coarse[f0_hz < 50] = 0

        # Expand F0 to match upsampled feature length
        f0_for_rvc = f0_hz.unsqueeze(0)  # [1, T]
        f0_coarse_for_rvc = f0_coarse.unsqueeze(0)  # [1, T]

        # Ensure lengths match
        feat_len = feat.size(1)
        f0_len = f0_for_rvc.size(1)
        if feat_len > f0_len * 2:
            feat = feat[:, :f0_len * 2, :]
        elif feat_len < f0_len * 2:
            f0_for_rvc = f0_for_rvc[:, :feat_len // 2]
            f0_coarse_for_rvc = f0_coarse_for_rvc[:, :feat_len // 2]

        # RVC decoder inference
        phone_lengths = torch.tensor([feat.size(1)], device=self.device)
        pitch = f0_coarse_for_rvc
        pitchf = f0_for_rvc

        audio = self.net_g.infer(
            feat.permute(0, 2, 1),  # [1, 768, T]
            phone_lengths,
            pitch, pitchf, self.sid,
        )[0][0, 0]  # [samples]

        audio = audio.float().cpu()

        # Re-extract HuBERT from reconstructed audio
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio.numpy(), self.tgt_sr)

        h_roundtrip = self.hubert_extractor.extract(tmp_path)

        # Clean up
        Path(tmp_path).unlink(missing_ok=True)

        return audio, h_roundtrip

    def compute_compensation(
        self,
        hubert_dir: Path,
        f0_dir: Path,
        output_dir: Path,
        alpha: float = 0.35,
        max_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-dimension compensation statistics across the dataset.

        For each sample:
        1. Load H_gt and F0
        2. Roundtrip through RVC
        3. Compute Δ = H_gt - H_roundtrip
        4. Save adjusted features: H_target = H_gt + α * Δ

        Also computes global statistics for analysis.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        hubert_files = sorted(hubert_dir.rglob("*.pt"))
        if max_samples:
            hubert_files = hubert_files[:max_samples]

        # Accumulators for statistics
        delta_sum = torch.zeros(768)
        delta_sq_sum = torch.zeros(768)
        n_frames_total = 0
        per_dim_deltas = []

        for h_path in tqdm(hubert_files, desc="Computing RVC compensation"):
            # Load features
            h_gt = torch.load(h_path, weights_only=True)
            f0_path = f0_dir / h_path.relative_to(hubert_dir)
            if not f0_path.exists():
                continue

            f0 = torch.load(f0_path, weights_only=True)

            # Align lengths
            min_len = min(h_gt.size(0), f0.size(0))
            h_gt = h_gt[:min_len]
            f0 = f0[:min_len]

            try:
                _, h_roundtrip = self.roundtrip(h_gt, f0)
            except Exception as e:
                logger.warning(f"Roundtrip failed for {h_path}: {e}")
                continue

            # Align lengths after roundtrip
            min_len = min(h_gt.size(0), h_roundtrip.size(0))
            h_gt = h_gt[:min_len]
            h_roundtrip = h_roundtrip[:min_len]

            # Compute delta
            delta = h_gt - h_roundtrip  # [T, 768]

            # Update statistics
            delta_sum += delta.sum(0)
            delta_sq_sum += (delta ** 2).sum(0)
            n_frames_total += min_len

            per_dim_deltas.append(delta.mean(0))

            # Save compensated features
            h_compensated = h_gt + alpha * delta
            out_path = output_dir / h_path.relative_to(hubert_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(h_compensated, out_path)

        # Compute global statistics
        if n_frames_total > 0:
            delta_mean = delta_sum / n_frames_total
            delta_var = delta_sq_sum / n_frames_total - delta_mean ** 2
            delta_std = delta_var.clamp(min=1e-8).sqrt()
        else:
            delta_mean = torch.zeros(768)
            delta_std = torch.ones(768)

        stats = {
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "n_frames": n_frames_total,
            "n_files": len(hubert_files),
            "alpha": alpha,
            "delta_mean_norm": delta_mean.norm().item(),
            "delta_std_mean": delta_std.mean().item(),
            "per_dim_mean_abs": delta_mean.abs().mean().item(),
            # Top dimensions with largest compensation
            "top_compensated_dims": delta_mean.abs().topk(20).indices.tolist(),
        }

        # Save statistics
        torch.save(stats, output_dir / "compensation_stats.pt")

        # Save human-readable summary
        summary = {k: v if not isinstance(v, torch.Tensor) else v.tolist()
                   for k, v in stats.items()}
        with open(output_dir / "compensation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"RVC Compensation: {n_frames_total} frames processed, "
            f"mean |Δ|={stats['per_dim_mean_abs']:.6f}, "
            f"α={alpha}"
        )

        return stats

    @staticmethod
    def apply_compensation(
        hubert_features: torch.Tensor,
        compensation_stats: dict,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply pre-computed compensation to HuBERT features.

        Can be used at inference time to adjust generated features.
        """
        delta_mean = compensation_stats["delta_mean"]
        if alpha is None:
            alpha = compensation_stats.get("alpha", 0.35)

        # Simple mean-shift compensation
        return hubert_features + alpha * delta_mean.to(hubert_features.device)
