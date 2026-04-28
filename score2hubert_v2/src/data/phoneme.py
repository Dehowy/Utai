"""IPA phoneme vocabulary for Score2HuBERT v2.

Loads a vocab.yaml with phone2id mapping. No G2P — that lives in preprocessing.
"""

from pathlib import Path
from typing import List, Union

import yaml


# Fixed special token IDs
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
SP_ID = 3
AP_ID = 4
BR_ID = 5
MASK_ID = 6


class PhonemeVocab:
    """IPA phoneme vocabulary.

    Args:
        vocab_path: path to vocab.yaml containing phone2id mapping.
    """

    def __init__(self, vocab_path: Union[str, Path]):
        vocab_path = Path(vocab_path)
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.phone2id: dict[str, int] = data["phone2id"]
        self.id2phone: dict[int, str] = {v: k for k, v in self.phone2id.items()}

        # Language mapping (optional, may not be present in all vocabs)
        self.lang2id: dict[str, int] = data.get("lang2id", {})
        self.id2lang: dict[int, str] = {v: k for k, v in self.lang2id.items()}

    @property
    def n_phones(self) -> int:
        """Total vocabulary size (including special tokens and unused slots)."""
        return max(self.phone2id.values()) + 1

    def encode(self, phones: List[str]) -> List[int]:
        """Convert phone strings to IDs. Unknown phones map to PAD_ID."""
        return [self.phone2id.get(p, PAD_ID) for p in phones]

    def decode(self, ids: List[int]) -> List[str]:
        """Convert IDs back to phone strings. Unknown IDs map to '<pad>'."""
        return [self.id2phone.get(i, "<pad>") for i in ids]

    def __len__(self) -> int:
        return self.n_phones

    def __repr__(self) -> str:
        return f"PhonemeVocab(n_phones={self.n_phones}, langs={list(self.lang2id.keys())})"
