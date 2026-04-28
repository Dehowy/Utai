"""
Batch collation for Score2HuBERT training.

Handles variable-length sequences by padding to batch maximum.
Creates masks for both phone-level and frame-level sequences.
"""

import torch
from typing import List, Dict


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of samples into a padded batch.

    Returns:
        Dictionary with padded tensors and masks:
        - phonemes: [B, T_phone_max] (padded with 0)
        - pitch: [B, T_phone_max]
        - duration: [B, T_phone_max]
        - hubert: [B, T_frame_max, 768]
        - f0: [B, T_frame_max]
        - singing_ratio: [B]
        - lang_id: [B]
        - phone_mask: [B, T_phone_max] (True = valid)
        - frame_mask: [B, T_frame_max] (True = valid)
        - energy: [B, T_frame_max] (if available)
        - breathiness: [B, T_frame_max] (if available)
    """
    B = len(batch)

    # Find max lengths
    max_phones = max(s["n_phones"] for s in batch)
    max_frames = max(s["n_frames"] for s in batch)

    # Initialize padded tensors
    n_tech = batch[0]["tech"].size(-1) if "tech" in batch[0] else 6
    phonemes = torch.zeros(B, max_phones, dtype=torch.long)
    pitch = torch.zeros(B, max_phones, dtype=torch.long)
    duration = torch.zeros(B, max_phones, dtype=torch.long)
    tech = torch.zeros(B, max_phones, n_tech, dtype=torch.float32)
    hubert = torch.zeros(B, max_frames, batch[0]["hubert"].size(-1))
    f0 = torch.zeros(B, max_frames)
    singing_ratio = torch.zeros(B)
    lang_id = torch.zeros(B, dtype=torch.long)
    phone_mask = torch.zeros(B, max_phones, dtype=torch.bool)
    frame_mask = torch.zeros(B, max_frames, dtype=torch.bool)

    # Discrete HuBERT tokens (Phase 5)
    has_tokens = "hubert_tokens" in batch[0]
    if has_tokens:
        hubert_tokens = torch.full((B, max_frames), -100, dtype=torch.long)  # CE ignore_index

    # Optional features
    has_energy = "energy" in batch[0]
    has_breathiness = "breathiness" in batch[0]
    if has_energy:
        energy = torch.zeros(B, max_frames)
    if has_breathiness:
        breathiness = torch.zeros(B, max_frames)

    # Fill in
    ids = []
    for i, sample in enumerate(batch):
        np = sample["n_phones"]
        nf = sample["n_frames"]

        phonemes[i, :np] = sample["phonemes"][:np]
        pitch[i, :np] = sample["pitch"][:np]
        duration[i, :np] = sample["duration"][:np]
        if "tech" in sample:
            tech[i, :np] = sample["tech"][:np]
        hubert[i, :nf] = sample["hubert"][:nf]
        f0[i, :nf] = sample["f0"][:nf]
        singing_ratio[i] = sample["singing_ratio"]
        lang_id[i] = sample["lang_id"]
        phone_mask[i, :np] = True
        frame_mask[i, :nf] = True

        if has_tokens and "hubert_tokens" in sample:
            t_len = min(nf, sample["hubert_tokens"].size(0))
            hubert_tokens[i, :t_len] = sample["hubert_tokens"][:t_len]
        if has_energy and "energy" in sample:
            energy[i, :nf] = sample["energy"][:nf]
        if has_breathiness and "breathiness" in sample:
            breathiness[i, :nf] = sample["breathiness"][:nf]

        ids.append(sample["id"])

    result = {
        "ids": ids,
        "phonemes": phonemes,
        "pitch": pitch,
        "duration": duration,
        "tech": tech,
        "hubert": hubert,
        "f0": f0,
        "singing_ratio": singing_ratio,
        "lang_id": lang_id,
        "phone_mask": phone_mask,
        "frame_mask": frame_mask,
    }

    if has_tokens:
        result["hubert_tokens"] = hubert_tokens
    if has_energy:
        result["energy"] = energy
    if has_breathiness:
        result["breathiness"] = breathiness

    return result
