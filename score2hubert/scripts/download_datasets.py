#!/usr/bin/env python3
"""
Dataset Download Script for Score2HuBERT.

Downloads and organizes the training datasets:
1. ACE-Opencpop (Chinese singing, ~130 hours, 30 singers — RECOMMENDED pre-training)
2. Opencpop (Chinese singing, ~5.2 hours, 1 singer — smaller alternative)
3. Kiritan (Japanese singing, ~57 min)
4. Ofuton-P (Japanese singing, ~46 min, male)
5. GTSinger (multi-lingual, Japanese subset, ~80h total with speech pairs)

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --dataset ace_opencpop
    python scripts/download_datasets.py --dataset kiritan
"""

import sys
import os
import argparse
import subprocess
import zipfile
import tarfile
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


# ── Dataset URLs and metadata ──

DATASETS = {
    "ace_opencpop": {
        "description": "ACE-Opencpop: Scaled Chinese singing corpus (~130 hours, 30 singers, 106k segments)",
        "url": "https://huggingface.co/datasets/espnet/ace-opencpop-segments",
        "huggingface": "espnet/ace-opencpop-segments",
        "paper": "https://arxiv.org/abs/2401.17619",
        "instructions": """
ACE-Opencpop (RECOMMENDED for pre-training):
  HuggingFace: https://huggingface.co/datasets/espnet/ace-opencpop-segments
  Paper: Shi et al., "Singing Voice Data Scaling-up", INTERSPEECH 2024

  ~130h from 30 singers, 48kHz/16-bit, CC-BY-NC-4.0
  Full phoneme+note timing, MIDI values, lyrics, tempo annotations.
  Created by synthesizing Opencpop songs through ACE Studio with manual tuning.

  Auto-download via HuggingFace datasets library:
    python -c "from datasets import load_dataset; load_dataset('espnet/ace-opencpop-segments')"

  Or place in: data/raw/ace_opencpop/
""",
        "auto_download": True,
    },

    "opencpop": {
        "description": "Opencpop: Chinese singing voice corpus (~5.2 hours, 1 singer, 3756 utterances)",
        "url": "https://wenet-e2e.github.io/opencpop/",
        "github": "https://github.com/wenet-e2e/opencpop",
        "paper": "https://arxiv.org/abs/2201.07429",
        "instructions": """
Opencpop (smaller alternative to ACE-Opencpop):
  Official: https://wenet-e2e.github.io/opencpop/download/
  GitHub: https://github.com/wenet-e2e/opencpop
  License: CC BY-NC-ND 4.0

  Requires Google Form submission — links emailed automatically.
  100 songs, 3756 utterances, ~5.2h, 44.1kHz, 1 professional female singer.
  Includes: wavs/, segments/transcriptions.txt, midis/, textgrids/

  After downloading, extract to: data/raw/opencpop/
  Expected structure:
    opencpop/
    ├── wavs/
    ├── segments/
    │   ├── wavs/              # segmented utterance audio
    │   └── transcriptions.txt # phoneme|note|duration annotations
    ├── midis/
    └── textgrids/
""",
        "auto_download": False,
    },

    "kiritan": {
        "description": "Tohoku Kiritan Singing Database (~57 min, 50 songs, Japanese)",
        "url": "https://zunko.jp/kiridev/login.php",
        "label_repo": "https://github.com/mmorise/kiritan_singing",
        "instructions": """
Kiritan (東北きりたん歌唱データベース):
  Audio download: https://zunko.jp/kiridev/login.php (requires Twitter/X login)
  Labels (MIDI + phonemes): https://github.com/mmorise/kiritan_singing
  Alt labels (DNN-SVS): https://github.com/r9y9/kiritan_singing

  50 songs, ~57 min. Non-commercial research only.
  Note: Song 08 has no MIDI for first half (too difficult to transcribe).

  After downloading, extract to: data/raw/kiritan/
  Expected structure:
    kiritan/
    ├── wav/                # WAV audio files
    ├── mono_label/         # Monophone HTS labels (.lab)
    ├── midi_label/         # MIDI transcriptions
    └── musicxml/           # MusicXML scores
""",
        "auto_download": False,
    },

    "ofuton": {
        "description": "Ofuton-P Singing Database (~46.5 min, 46 songs, Japanese male)",
        "url": "https://sites.google.com/view/oftn-utagoedb",
        "nnsvs_recipe": "https://github.com/taroushirani/nnsvs_ofuton_p_utagoe_db",
        "instructions": """
Ofuton-P (おふとんP歌声データベース):
  Official: https://sites.google.com/view/oftn-utagoedb
  NNSVS recipe: https://github.com/taroushirani/nnsvs_ofuton_p_utagoe_db

  46 songs, ~46.5 min, 96kHz/24-bit, male voice (children's songs).
  Non-commercial without permission. Attribution: "DB制作:おふとんP"
  Includes: WAV, LAB labels, MIDI, MusicXML, UST files.

  After downloading, extract to: data/raw/ofuton/
  Expected structure:
    ofuton/
    ├── wav/
    ├── lab/                # Phoneme labels (100ns precision)
    ├── mid/                # MIDI files
    └── musicxml/
""",
        "auto_download": False,
    },

    "gtsinger": {
        "description": "GTSinger: Multi-lingual Singing Corpus (80.6h singing + 16.2h speech, NeurIPS 2024)",
        "url": "https://github.com/AaronZ345/GTSinger",
        "huggingface": "GTSinger/GTSinger",
        "google_drive": "https://drive.google.com/drive/folders/1xcdvCxNAEEfJElt7sEP-xT8dMKxn1_Lz",
        "instructions": """
GTSinger (NeurIPS 2024 Spotlight):
  GitHub: https://github.com/AaronZ345/GTSinger
  HuggingFace: https://huggingface.co/datasets/GTSinger/GTSinger
  Google Drive: https://drive.google.com/drive/folders/1xcdvCxNAEEfJElt7sEP-xT8dMKxn1_Lz

  ~29 GB total, 80.6h singing + 16.2h paired speech, 48kHz/24-bit
  9 languages (ja, zh, en, ko, ru, es, fr, de, it), 20 singers
  CC BY-NC-SA 4.0 (accept terms on HuggingFace before download)

  Annotations: TextGrid phoneme alignments, technique labels (mixed voice,
  falsetto, breathy, pharyngeal, vibrato, glissando), MusicXML scores,
  metadata.json + phone_set.json per language.

  Download Japanese subset. Extract to: data/raw/gtsinger/
""",
        "auto_download": True,
    },
}


def download_ace_opencpop(output_dir: Path):
    """Download ACE-Opencpop (~130h, 30 singers) from HuggingFace."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "ace_opencpop"

    # Try HuggingFace datasets library first (handles streaming/caching)
    try:
        from datasets import load_dataset
        logger.info("Downloading ACE-Opencpop via HuggingFace datasets library...")
        logger.info("This is ~43.6 GB — may take a while.")
        ds = load_dataset("espnet/ace-opencpop-segments", cache_dir=str(target))
        logger.info(f"ACE-Opencpop downloaded: {len(ds['train'])} train, {len(ds['test'])} test")
        return True
    except ImportError:
        logger.warning("'datasets' library not installed. pip install datasets")
    except Exception as e:
        logger.warning(f"datasets download failed: {e}")

    # Fallback: huggingface_hub snapshot
    try:
        from huggingface_hub import snapshot_download
        logger.info("Downloading ACE-Opencpop via huggingface_hub...")
        snapshot_download(
            "espnet/ace-opencpop-segments",
            repo_type="dataset",
            local_dir=str(target),
        )
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_gtsinger(output_dir: Path):
    """Download GTSinger (~29 GB) from HuggingFace."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "gtsinger"

    try:
        from huggingface_hub import snapshot_download
        logger.info("Downloading GTSinger from HuggingFace...")
        logger.info("Full dataset is ~29 GB. Filtering to Japanese subset.")
        snapshot_download(
            "GTSinger/GTSinger",
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["*Japanese*", "*ja*", "*.json", "*.wav", "*.TextGrid"],
        )
        logger.info(f"GTSinger downloaded to {target}")
        return True
    except Exception as e:
        logger.warning(f"HuggingFace download failed: {e}")

    # Fallback: git clone
    try:
        logger.info("Trying git lfs clone (requires git-lfs)...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://huggingface.co/datasets/GTSinger/GTSinger",
             str(target)],
            check=True,
        )
        return True
    except Exception as e:
        logger.error(f"Git clone failed: {e}")
        return False


def download_kiritan_labels(output_dir: Path):
    """Download Kiritan label data (MIDI + phoneme labels) from GitHub."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "kiritan" / "labels"

    try:
        logger.info("Downloading Kiritan labels from GitHub...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/mmorise/kiritan_singing.git",
             str(target)],
            check=True,
        )
        logger.info(f"Kiritan labels downloaded to {target}")
        logger.info("NOTE: Audio files must be downloaded separately from https://zunko.jp/kiridev/login.php")
        return True
    except Exception as e:
        logger.error(f"Git clone failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Score2HuBERT training datasets")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()),
                        help="Specific dataset to download")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--info", action="store_true", help="Print dataset info and instructions")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.info or (not args.dataset and not args.all):
        print("\n" + "=" * 60)
        print("Score2HuBERT Training Datasets")
        print("=" * 60)
        for name, info in DATASETS.items():
            print(f"\n{'─' * 40}")
            print(f"  {name}: {info['description']}")
            print(f"  URL: {info['url']}")
            print(info['instructions'])
        return

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())

    for name in datasets:
        info = DATASETS[name]
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Dataset: {name}")
        logger.info(f"Description: {info['description']}")

        target_dir = output_dir / name
        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"Already exists: {target_dir}, skipping")
            continue

        if name == "ace_opencpop":
            success = download_ace_opencpop(output_dir)
            if not success:
                print(info["instructions"])
        elif name == "gtsinger":
            success = download_gtsinger(output_dir)
            if not success:
                print(info["instructions"])
        elif name == "kiritan":
            download_kiritan_labels(output_dir)
            print(info["instructions"])
        else:
            print(f"\n{name} requires manual download:")
            print(info["instructions"])

    print("\n" + "=" * 60)
    print("After downloading, run preprocessing:")
    print(f"  python scripts/preprocess.py --dataset <name> --raw-dir data/raw/<name>")
    print("=" * 60)


if __name__ == "__main__":
    main()
