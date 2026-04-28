"""Collate function for Score2HuBERT v2 DataLoader.

Pads variable-length sequences to batch max and builds masks.
Masks follow the convention: True = padding position (ignored by model).
"""

import torch


def collate_fn(batch: list[dict]) -> dict:
    """Collate a list of dataset items into a padded batch.

    Returns:
        dict with keys:
            phonemes:   [B, T_phone] long
            pitch:      [B, T_phone] long
            duration:   [B, T_phone] long
            hubert:     [B, T_frame, 768] float
            f0:         [B, T_frame] float
            phone_mask: [B, T_phone] bool (True = pad)
            frame_mask: [B, T_frame] bool (True = pad)
            n_frames:   [B] long
            n_phones:   [B] long
            languages:  list[str] of length B
    """
    B = len(batch)
    max_phones = max(item["n_phones"] for item in batch)
    max_frames = max(item["n_frames"] for item in batch)
    hubert_dim = batch[0]["hubert"].shape[-1]

    # Pre-allocate padded tensors
    phonemes = torch.zeros(B, max_phones, dtype=torch.long)
    pitch = torch.zeros(B, max_phones, dtype=torch.long)
    duration = torch.zeros(B, max_phones, dtype=torch.long)
    hubert = torch.zeros(B, max_frames, hubert_dim, dtype=torch.float32)
    f0 = torch.zeros(B, max_frames, dtype=torch.float32)

    n_frames_list = []
    n_phones_list = []
    languages = []

    for i, item in enumerate(batch):
        np = item["n_phones"]
        nf = item["n_frames"]

        phonemes[i, :np] = item["phonemes"]
        pitch[i, :np] = item["pitch"]
        duration[i, :np] = item["duration"]
        hubert[i, :nf] = item["hubert"]
        f0[i, :nf] = item["f0"]

        n_frames_list.append(nf)
        n_phones_list.append(np)
        languages.append(item["language"])

    n_frames_t = torch.tensor(n_frames_list, dtype=torch.long)
    n_phones_t = torch.tensor(n_phones_list, dtype=torch.long)

    # Masks: True where padded
    phone_mask = torch.arange(max_phones).unsqueeze(0) >= n_phones_t.unsqueeze(1)
    frame_mask = torch.arange(max_frames).unsqueeze(0) >= n_frames_t.unsqueeze(1)

    return {
        "phonemes": phonemes,
        "pitch": pitch,
        "duration": duration,
        "hubert": hubert,
        "f0": f0,
        "phone_mask": phone_mask,
        "frame_mask": frame_mask,
        "n_frames": n_frames_t,
        "n_phones": n_phones_t,
        "languages": languages,
    }
