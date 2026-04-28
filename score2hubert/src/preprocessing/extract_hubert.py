"""
HuBERT feature extraction for Score2HuBERT training data.

Uses the EXACT same HuBERT model as RVC (hubert_base.pt) to ensure
feature distribution compatibility. Extracts Layer 12 features (768-dim)
at 100fps (16kHz / 160 hop).

Critical: feature distribution must match what RVC's decoder expects.
"""

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class HuBERTExtractor:
    """
    Extract HuBERT features matching RVC's internal representation.

    Uses fairseq's HuBERT-Base model, Layer 12, 768-dim features.
    Frame rate: 100fps (hop_length=160 at 16kHz).
    """

    def __init__(
        self,
        model_path: str,
        layer: int = 12,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.device = device
        self.layer = layer
        self.use_fp16 = use_fp16 and device == "cuda"
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load HuBERT model via fairseq (same as RVC)."""
        from fairseq import checkpoint_utils

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [model_path], suffix=""
        )
        model = models[0].to(self.device)
        model.eval()

        if self.use_fp16:
            model = model.half()

        return model

    @torch.no_grad()
    def extract(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Extract HuBERT features from an audio file.

        Args:
            audio_path: path to audio file (any format supported by soundfile/librosa)

        Returns:
            features: [T_frame, 768] float32 tensor
        """
        # Load and resample to 16kHz mono
        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo to mono
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        if self.use_fp16:
            audio_tensor = audio_tensor.half()

        # Pad for stability (match RVC behavior)
        audio_tensor = F.pad(audio_tensor, (40, 40), mode="reflect")

        # Extract features
        # RVC V2 uses layer 12 direct output (no final_proj)
        features = self.model.extract_features(
            source=audio_tensor,
            padding_mask=None,
            output_layer=self.layer,
        )

        # features is a tuple: (feature_tensor, padding_mask)
        feat = features[0].squeeze(0)  # [T, 768]

        # Convert back to float32 for storage
        feat = feat.float().cpu()

        return feat

    @torch.no_grad()
    def extract_batch(self, audio_paths: list, batch_size: int = 8) -> list:
        """Extract features for multiple audio files."""
        results = []
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            for path in batch_paths:
                try:
                    feat = self.extract(path)
                    results.append(feat)
                except Exception as e:
                    logger.error(f"Failed to extract HuBERT from {path}: {e}")
                    results.append(None)
        return results

    def process_directory(
        self,
        audio_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: tuple = (".wav", ".flac", ".ogg", ".mp3"),
    ):
        """
        Process all audio files in a directory and save HuBERT features.

        Args:
            audio_dir: directory containing audio files
            output_dir: directory to save .pt feature files
        """
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_files = []
        for ext in extensions:
            audio_files.extend(audio_dir.rglob(f"*{ext}"))

        logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")

        for audio_path in tqdm(audio_files, desc="Extracting HuBERT"):
            # Generate output filename (preserve relative structure)
            rel_path = audio_path.relative_to(audio_dir)
            out_path = output_dir / rel_path.with_suffix(".pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                continue  # skip if already extracted

            try:
                feat = self.extract(audio_path)
                torch.save(feat, out_path)
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")


def compute_hubert_statistics(feature_dir: Union[str, Path]) -> dict:
    """
    Compute mean and std of HuBERT features across a dataset.

    Useful for:
    1. Verifying our features match RVC's expected distribution
    2. Feature normalization (if needed)
    3. Diagnosing distribution shift issues
    """
    feature_dir = Path(feature_dir)
    pt_files = list(feature_dir.rglob("*.pt"))

    if not pt_files:
        raise ValueError(f"No .pt files found in {feature_dir}")

    # Online mean/variance computation
    n_total = 0
    running_sum = None
    running_sq_sum = None

    for pt_file in tqdm(pt_files, desc="Computing HuBERT statistics"):
        feat = torch.load(pt_file, weights_only=True)  # [T, 768]
        n = feat.size(0)

        if running_sum is None:
            running_sum = feat.sum(0)
            running_sq_sum = (feat ** 2).sum(0)
        else:
            running_sum += feat.sum(0)
            running_sq_sum += (feat ** 2).sum(0)
        n_total += n

    mean = running_sum / n_total
    var = running_sq_sum / n_total - mean ** 2
    std = var.clamp(min=1e-8).sqrt()

    stats = {
        "mean": mean,
        "std": std,
        "n_frames": n_total,
        "n_files": len(pt_files),
        "dim": mean.size(0),
        "mean_norm": mean.norm().item(),
        "std_mean": std.mean().item(),
    }

    logger.info(
        f"HuBERT stats: {n_total} frames, {len(pt_files)} files, "
        f"mean_norm={stats['mean_norm']:.4f}, std_mean={stats['std_mean']:.4f}"
    )

    return stats
