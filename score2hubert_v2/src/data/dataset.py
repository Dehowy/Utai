"""Score2HuBERT v2 dataset — loads preprocessed .pt files.

Each data directory contains:
    hubert/{id}.pt, f0/{id}.pt, phonemes/{id}.pt, pitch/{id}.pt, duration/{id}.pt
    metadata.json, vocab.yaml
"""

import json
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset


class Score2HuBERTDataset(Dataset):
    """Dataset for Score2HuBERT training/validation.

    Supports a single data directory or a list of directories (for mixed
    zh+ja datasets). Entries are filtered by split (train/val/test).

    Args:
        data_dirs: path or list of paths to preprocessed data directories.
        split: one of 'train', 'val', 'test'.
        max_frames: truncate utterances longer than this (0 = no limit).
    """

    def __init__(
        self,
        data_dirs: Union[str, Path, List[Union[str, Path]]],
        split: str = "train",
        max_frames: int = 1200,
        entry_filter: callable = None,
    ):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        data_dirs = [Path(d) for d in data_dirs]

        self.max_frames = max_frames
        self.entries: list[dict] = []

        for data_dir in data_dirs:
            meta_path = data_dir / "metadata.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            for entry in metadata:
                if entry["split"] != split:
                    continue
                if entry_filter is not None and not entry_filter(entry):
                    continue
                entry["_data_dir"] = str(data_dir)
                self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        data_dir = Path(entry["_data_dir"])
        utt_id = entry["id"]

        # Load tensors
        hubert = torch.load(data_dir / "hubert" / f"{utt_id}.pt", weights_only=True)
        f0 = torch.load(data_dir / "f0" / f"{utt_id}.pt", weights_only=True)
        phonemes = torch.load(data_dir / "phonemes" / f"{utt_id}.pt", weights_only=True)
        pitch = torch.load(data_dir / "pitch" / f"{utt_id}.pt", weights_only=True)
        duration = torch.load(data_dir / "duration" / f"{utt_id}.pt", weights_only=True)

        n_frames = hubert.shape[0]
        n_phones = phonemes.shape[0]

        # Fix duration sum to match frame count (off-by-one from preprocessing)
        dur_sum = duration.sum().item()
        if dur_sum != n_frames:
            diff = n_frames - dur_sum
            # Adjust the longest phoneme's duration
            longest_idx = duration.argmax()
            duration[longest_idx] += diff

        # Truncate to max_frames if needed
        if self.max_frames > 0 and n_frames > self.max_frames:
            hubert = hubert[: self.max_frames]
            f0 = f0[: self.max_frames]
            n_frames = self.max_frames

            # Trim phonemes to fit: walk durations until we hit the frame limit
            cumsum = torch.cumsum(duration, dim=0)
            # Number of phones that fit entirely within max_frames
            keep = (cumsum <= self.max_frames).sum().item()
            if keep < n_phones:
                # Keep one extra phone with truncated duration
                keep = min(keep + 1, n_phones)
                phonemes = phonemes[:keep]
                pitch = pitch[:keep]
                duration = duration[:keep]
                # Last phone gets remaining frames
                duration[-1] = self.max_frames - (cumsum[keep - 2].item() if keep >= 2 else 0)
                n_phones = keep
            # Re-fix duration sum after truncation
            duration[-1] += self.max_frames - duration.sum().item()

        return {
            "phonemes": phonemes.long(),       # [T_phone]
            "pitch": pitch.long(),             # [T_phone]
            "duration": duration.long(),       # [T_phone]
            "hubert": hubert.float(),          # [T_frame, 768]
            "f0": f0.float(),                  # [T_frame]
            "n_frames": n_frames,
            "n_phones": n_phones,
            "language": entry["language"],
        }
