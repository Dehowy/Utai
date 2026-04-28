"""
PyTorch Dataset for Score2HuBERT training.

Each sample contains:
- phonemes: list of phoneme IDs
- pitch: MIDI note per phoneme
- duration: frame count per phoneme
- singing_ratio: float in [0, 1]
- language: language ID
- hubert: HuBERT features [T_frame, 768]
- f0: F0 contour [T_frame] in Hz
- energy: energy envelope [T_frame] (optional)
- breathiness: breathiness [T_frame] (optional)

Data is loaded from preprocessed .pt files in the processed directory.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Dict
import json
import random


class Score2HuBERTDataset(Dataset):
    """
    Dataset for Score2HuBERT training.

    Expected directory structure:
        processed_dir/
        ├── metadata.json       # list of {id, language, singing_ratio, ...}
        ├── hubert/             # {id}.pt → [T_frame, 768] float32
        ├── f0/                 # {id}.pt → [T_frame] float32 (Hz)
        ├── phonemes/           # {id}.pt → [T_phone] int64
        ├── pitch/              # {id}.pt → [T_phone] int64 (MIDI)
        ├── duration/           # {id}.pt → [T_phone] int64 (frames)
        ├── energy/             # {id}.pt → [T_frame] float32 (optional)
        └── breathiness/        # {id}.pt → [T_frame] float32 (optional)
    """

    def __init__(
        self,
        processed_dir: str,
        split: str = "train",  # "train" | "val" | "test"
        max_frames: int = 1200,
        min_frames: int = 30,
        augmentation: Optional[dict] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.augmentation = augmentation or {}

        # Load metadata
        meta_path = self.processed_dir / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)

        # Filter by split
        self.samples = [m for m in all_metadata if m.get("split", "train") == split]

        # Further filter by frame count
        self.samples = [
            m for m in self.samples
            if self.min_frames <= m.get("n_frames", self.max_frames) <= self.max_frames
        ]

        # Check which optional features exist
        self.has_energy = (self.processed_dir / "energy").exists()
        self.has_breathiness = (self.processed_dir / "breathiness").exists()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.samples[idx]
        sid = meta["id"]

        # Load core features
        hubert = torch.load(self.processed_dir / "hubert" / f"{sid}.pt", weights_only=True)
        f0 = torch.load(self.processed_dir / "f0" / f"{sid}.pt", weights_only=True)
        phonemes = torch.load(self.processed_dir / "phonemes" / f"{sid}.pt", weights_only=True)
        pitch = torch.load(self.processed_dir / "pitch" / f"{sid}.pt", weights_only=True)
        duration = torch.load(self.processed_dir / "duration" / f"{sid}.pt", weights_only=True)

        # Technique flags (Phase 3): per-phoneme [T_phone, 6] binary flags for
        # GTSinger's 6 singing techniques (mix_voice, falsetto, breathy, pharyngeal,
        # vibrato, glissando). Samples without an explicit tech file (ace_opencpop,
        # any pre-Phase-3 data, speech entries) get all-zeros of the correct shape.
        tech_path = self.processed_dir / "tech" / f"{sid}.pt"
        if tech_path.exists():
            tech = torch.load(tech_path, weights_only=True).float()  # [T_phone, 6]
            # Defensive: align to phoneme count if mismatch from preprocessing quirks
            if tech.size(0) != phonemes.size(0):
                target_T = phonemes.size(0)
                if tech.size(0) > target_T:
                    tech = tech[:target_T]
                else:
                    pad = torch.zeros(target_T - tech.size(0), tech.size(1), dtype=torch.float32)
                    tech = torch.cat([tech, pad], dim=0)
        else:
            tech = torch.zeros(phonemes.size(0), 6, dtype=torch.float32)

        # Discrete HuBERT tokens (Phase 5): K-means codebook indices per frame.
        # Optional — only present after running scripts/quantize_hubert.py.
        token_path = self.processed_dir / "hubert_tokens" / f"{sid}.pt"
        if token_path.exists():
            hubert_tokens = torch.load(token_path, weights_only=True).long()  # [T_frame]
            # Align to continuous hubert length
            if hubert_tokens.size(0) != hubert.size(0):
                T = hubert.size(0)
                hubert_tokens = hubert_tokens[:T] if hubert_tokens.size(0) > T else \
                    F.pad(hubert_tokens, (0, T - hubert_tokens.size(0)), value=0)
        else:
            hubert_tokens = None

        sample = {
            "id": sid,
            "phonemes": phonemes.long(),
            "pitch": pitch.long(),
            "duration": duration.long(),
            "hubert": hubert.float(),
            "f0": f0.float(),
            "singing_ratio": torch.tensor(meta.get("singing_ratio", 1.0)).float(),
            "lang_id": torch.tensor(meta.get("lang_id", 0)).long(),
            "tech": tech,  # [T_phone, 6]
            "n_phones": phonemes.size(0),
            "n_frames": hubert.size(0),
        }
        if hubert_tokens is not None:
            sample["hubert_tokens"] = hubert_tokens

        # Optional features
        if self.has_energy:
            energy_path = self.processed_dir / "energy" / f"{sid}.pt"
            if energy_path.exists():
                sample["energy"] = torch.load(energy_path, weights_only=True).float()

        if self.has_breathiness:
            breath_path = self.processed_dir / "breathiness" / f"{sid}.pt"
            if breath_path.exists():
                sample["breathiness"] = torch.load(breath_path, weights_only=True).float()

        # Data augmentation (training only)
        if self.split == "train":
            sample = self._augment(sample)

        return sample

    def _augment(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentation."""
        aug_cfg = self.augmentation

        # Pitch shift augmentation
        ps_cfg = aug_cfg.get("pitch_shift", {})
        if ps_cfg.get("enabled", False) and random.random() < ps_cfg.get("prob", 0.5):
            lo, hi = ps_cfg.get("range_semitones", [-12, 12])
            shift = random.randint(lo, hi)
            if shift != 0:
                sample = self._pitch_shift(sample, shift)

        # Time stretch augmentation
        ts_cfg = aug_cfg.get("time_stretch", {})
        if ts_cfg.get("enabled", False) and random.random() < ts_cfg.get("prob", 0.3):
            lo, hi = ts_cfg.get("range", [0.9, 1.1])
            factor = random.uniform(lo, hi)
            if abs(factor - 1.0) > 0.01:
                sample = self._time_stretch(sample, factor)

        return sample

    def _pitch_shift(self, sample: dict, semitones: int) -> dict:
        """
        Pitch shift augmentation.

        Shifts MIDI pitch and F0, but does NOT modify HuBERT features
        (HuBERT content features are largely pitch-invariant).
        """
        sample = sample.copy()

        # Shift MIDI pitch
        sample["pitch"] = (sample["pitch"] + semitones).clamp(0, 127)

        # Shift F0
        ratio = 2.0 ** (semitones / 12.0)
        f0 = sample["f0"]
        voiced = f0 > 0
        f0_shifted = f0.clone()
        f0_shifted[voiced] = f0[voiced] * ratio
        sample["f0"] = f0_shifted

        return sample

    def _time_stretch(self, sample: dict, factor: float) -> dict:
        """
        Time stretch augmentation.

        Adjusts durations and resamples frame-level features.
        Factor > 1 = slower, < 1 = faster.
        """
        sample = sample.copy()

        # Scale durations
        new_dur = (sample["duration"].float() * factor).round().long().clamp(min=1)
        sample["duration"] = new_dur

        # Resample frame-level features
        new_n_frames = new_dur.sum().item()
        for key in ["hubert", "f0", "energy", "breathiness"]:
            if key in sample and sample[key] is not None:
                feat = sample[key]
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0).unsqueeze(0).float()
                    feat = torch.nn.functional.interpolate(
                        feat, size=new_n_frames, mode="linear", align_corners=False
                    )
                    sample[key] = feat.squeeze()
                elif feat.dim() == 2:
                    feat = feat.unsqueeze(0).permute(0, 2, 1).float()
                    feat = torch.nn.functional.interpolate(
                        feat, size=new_n_frames, mode="linear", align_corners=False
                    )
                    sample[key] = feat.squeeze(0).permute(1, 0)

        sample["n_frames"] = new_n_frames
        return sample
