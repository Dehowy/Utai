"""Populate tech/*.pt tensors for existing gtsinger_{zh,ja,en}/ dirs WITHOUT
re-running HuBERT/F0 extraction.

Reads:
  - Our dataset metadata.json (to enumerate utterances and find orig_item_name)
  - GTSinger source processed/processed/<Lang>/metadata.json (to read the 6 tech flag arrays per utterance)
  - Existing phonemes/*.pt (to know the decomposed phoneme count for alignment sanity)

Writes:
  - data/processed/gtsinger_<lang>/tech/<utt_id>.pt : [T_phone, 6] float32

Speech entries: tech is all zeros (no technique annotation in speech).
Singing entries: tech expanded in parallel with IPA decomposition to match T_phone.

After running, also re-merge the multi_lang_4/tech/ subdir (delete + merge)
so the hardlinks point to the newly-created tech files. This script DOES NOT
re-merge automatically — run scripts/preprocess.py --merge after.

Usage: venv/Scripts/python.exe _add_tech_flags_to_existing_gtsinger.py
"""

import sys
import json
import unicodedata
from pathlib import Path
from tqdm import tqdm

import torch

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
sys.stdout.reconfigure(encoding="utf-8")

from src.preprocessing.prepare_dataset import (
    GTSINGER_TECH_KEYS,
    N_TECHNIQUES,
    _expand_tech_via_gtsinger_g2p,
)

GT_ROOT = PROJECT / "GTSinger"
LANG_NAME_MAP = {"zh": "Chinese", "ja": "Japanese", "en": "English"}


def build_source_index(lang_name: str) -> dict:
    """Index GTSinger source metadata by item_name for O(1) lookup."""
    src_meta_path = GT_ROOT / "processed" / "processed" / lang_name / "metadata.json"
    with open(src_meta_path, encoding="utf-8") as f:
        src_meta = json.load(f)
    return {e["item_name"]: e for e in src_meta}


def process_language(lang_code: str, lang_name: str):
    proc = PROJECT / f"data/processed/gtsinger_{lang_code}"
    tech_dir = proc / "tech"
    tech_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {lang_code} ({lang_name}) ===")
    print(f"  output: {tech_dir}")

    src_index = build_source_index(lang_name)
    print(f"  source metadata: {len(src_index)} entries indexed")

    with open(proc / "metadata.json", encoding="utf-8") as f:
        our_meta = json.load(f)
    print(f"  our metadata:    {len(our_meta)} entries")

    n_sing = 0
    n_sing_missing_src = 0
    n_sing_length_mismatch = 0
    n_speech = 0
    n_failed = 0

    for entry in tqdm(our_meta, desc=f"{lang_code} tech"):
        utt_id = entry["id"]
        phoneme_path = proc / "phonemes" / f"{utt_id}.pt"
        if not phoneme_path.exists():
            n_failed += 1
            continue
        T_phone = torch.load(phoneme_path, weights_only=True).shape[0]

        if entry.get("singing_ratio", 1.0) == 1.0:
            # Singing: expand real tech flags from source metadata
            orig_name = entry.get("orig_item_name", "")
            src_entry = src_index.get(orig_name)
            if src_entry is None:
                n_sing_missing_src += 1
                # Fallback: save zeros so downstream code always finds a file
                torch.save(
                    torch.zeros(T_phone, N_TECHNIQUES, dtype=torch.float32),
                    tech_dir / f"{utt_id}.pt",
                )
                continue

            raw_phs = src_entry["ph"]
            raw_tech_matrix = []
            for i in range(len(raw_phs)):
                row = []
                for key in GTSINGER_TECH_KEYS:
                    arr = src_entry.get(key, [])
                    row.append(int(arr[i]) if i < len(arr) else 0)
                raw_tech_matrix.append(row)

            tech_expanded = _expand_tech_via_gtsinger_g2p(
                raw_phs, raw_tech_matrix, language=lang_code
            )

            # Align to our stored phoneme length (decomposition might have been
            # truncated by min_len sanity check during original prep).
            if len(tech_expanded) != T_phone:
                n_sing_length_mismatch += 1
                if len(tech_expanded) > T_phone:
                    tech_expanded = tech_expanded[:T_phone]
                else:
                    # Pad with zeros if shorter (unusual but possible)
                    while len(tech_expanded) < T_phone:
                        tech_expanded.append([0] * N_TECHNIQUES)

            torch.save(
                torch.tensor(tech_expanded, dtype=torch.float32),
                tech_dir / f"{utt_id}.pt",
            )
            n_sing += 1
        else:
            # Speech: no technique annotation → zeros
            torch.save(
                torch.zeros(T_phone, N_TECHNIQUES, dtype=torch.float32),
                tech_dir / f"{utt_id}.pt",
            )
            n_speech += 1

    print(f"  singing: {n_sing} OK, {n_sing_missing_src} fallback-zero (orig_name not in src),")
    print(f"           {n_sing_length_mismatch} length-adjusted,")
    print(f"  speech:  {n_speech} (all-zero)")
    print(f"  failed:  {n_failed}")


def main():
    for lang_code, lang_name in LANG_NAME_MAP.items():
        process_language(lang_code, lang_name)

    print()
    print("DONE. Next steps:")
    print("  1. Verify by spot-checking a few tech/*.pt files (non-zero for GTSinger singing w/ techniques)")
    print("  2. Re-merge multi_lang_4/tech via:")
    print("     rm -rf data/processed/multi_lang_4/tech")
    print("     venv/Scripts/python.exe scripts/preprocess.py --merge \\")
    print("         --out-dir data/processed/multi_lang_4 \\")
    print("         --sources data/processed/ace_opencpop data/processed/gtsinger_zh \\")
    print("                   data/processed/gtsinger_ja data/processed/gtsinger_en")


if __name__ == "__main__":
    main()
