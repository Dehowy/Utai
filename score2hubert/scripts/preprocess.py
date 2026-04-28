#!/usr/bin/env python3
"""
Dataset Preprocessing Entry Point.

Usage:
    # Process Opencpop (Chinese pre-training data)
    python scripts/preprocess.py --dataset opencpop --raw-dir data/raw/opencpop --out-dir data/processed/opencpop

    # Process Kiritan (Japanese)
    python scripts/preprocess.py --dataset kiritan --raw-dir data/raw/kiritan --out-dir data/processed/kiritan

    # Process GTSinger Japanese subset
    python scripts/preprocess.py --dataset gtsinger --raw-dir data/raw/gtsinger --out-dir data/processed/gtsinger --language ja

    # Merge all processed datasets
    python scripts/preprocess.py --merge --out-dir data/processed/merged \
        --sources data/processed/opencpop data/processed/kiritan data/processed/gtsinger

    # Compute RVC compensation (optional)
    python scripts/preprocess.py --rvc-compensation --hubert-dir data/processed/merged/hubert \
        --f0-dir data/processed/merged/f0 --rvc-model RVC/RVC20240604Nvidia/assets/weights/Iroha_e175_s6300.pth
"""

import sys
import argparse
import json
import shutil
import logging
from pathlib import Path
from omegaconf import OmegaConf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.data.phoneme import PhonemeVocab
from src.preprocessing.extract_hubert import HuBERTExtractor, compute_hubert_statistics
from src.preprocessing.extract_f0 import F0Extractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for Score2HuBERT")
    parser.add_argument("--config", type=str, default="configs/base.yaml")

    # Dataset selection
    parser.add_argument("--dataset", type=str,
                        choices=["ace_opencpop", "opencpop", "kiritan", "gtsinger", "ofuton"],
                        help="Dataset to process")
    parser.add_argument("--raw-dir", type=str, help="Raw dataset directory")
    parser.add_argument("--out-dir", type=str, help="Output processed directory")
    parser.add_argument("--language", type=str, default="ja")

    # Merge mode
    parser.add_argument("--merge", action="store_true", help="Merge multiple processed datasets")
    parser.add_argument("--sources", nargs="*", help="Source directories to merge")

    # RVC compensation
    parser.add_argument("--rvc-compensation", action="store_true")
    parser.add_argument("--hubert-dir", type=str)
    parser.add_argument("--f0-dir", type=str)
    parser.add_argument("--rvc-model", type=str)
    parser.add_argument("--alpha", type=float, default=0.35)

    # Feature extraction only
    parser.add_argument("--extract-only", type=str, choices=["hubert", "f0", "both"],
                        help="Only extract specific features")
    parser.add_argument("--audio-dir", type=str, help="Audio directory for extraction")

    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    # Initialize vocab
    vocab = PhonemeVocab()

    # Resolve HuBERT model path
    hubert_model = config["paths"]["hubert_model"]
    if not Path(hubert_model).is_absolute():
        hubert_model = str(project_root / hubert_model)

    rmvpe_model = config["paths"]["rmvpe_model"]
    if not Path(rmvpe_model).is_absolute():
        rmvpe_model = str(project_root / rmvpe_model)

    if args.merge:
        # Merge processed datasets
        merge_datasets(args.sources, args.out_dir)
        return

    if args.rvc_compensation:
        # Compute RVC compensation
        from src.preprocessing.rvc_compensation import RVCCompensationAnalyzer
        analyzer = RVCCompensationAnalyzer(
            rvc_model_path=args.rvc_model,
            hubert_model_path=hubert_model,
        )
        stats = analyzer.compute_compensation(
            hubert_dir=Path(args.hubert_dir),
            f0_dir=Path(args.f0_dir),
            output_dir=Path(args.out_dir or "data/processed/rvc_compensation"),
            alpha=args.alpha,
        )
        logger.info(f"Compensation stats: mean|Δ|={stats['per_dim_mean_abs']:.6f}")
        return

    if args.extract_only:
        # Feature extraction only
        if args.extract_only in ("hubert", "both"):
            extractor = HuBERTExtractor(hubert_model)
            extractor.process_directory(args.audio_dir, args.out_dir + "/hubert")
        if args.extract_only in ("f0", "both"):
            f0_ext = F0Extractor(method="rmvpe", rmvpe_model_path=rmvpe_model)
            f0_ext.process_directory(args.audio_dir, args.out_dir + "/f0")
        return

    # Full dataset processing
    if not args.dataset or not args.raw_dir:
        parser.error("--dataset and --raw-dir are required")

    # Initialize extractors
    hubert_extractor = HuBERTExtractor(hubert_model)
    f0_extractor = F0Extractor(method="rmvpe", rmvpe_model_path=rmvpe_model)

    from src.preprocessing.prepare_dataset import (
        prepare_ace_opencpop, prepare_opencpop, prepare_kiritan,
        prepare_ofuton, prepare_gtsinger, assign_splits
    )

    out_dir = args.out_dir or f"data/processed/{args.dataset}"

    if args.dataset == "ace_opencpop":
        metadata = prepare_ace_opencpop(
            args.raw_dir or "data/raw", out_dir, vocab, hubert_extractor, f0_extractor
        )
    elif args.dataset == "opencpop":
        metadata = prepare_opencpop(
            args.raw_dir, out_dir, vocab, hubert_extractor, f0_extractor
        )
    elif args.dataset == "kiritan":
        metadata = prepare_kiritan(
            args.raw_dir, out_dir, vocab, hubert_extractor, f0_extractor
        )
    elif args.dataset == "ofuton":
        metadata = prepare_ofuton(
            args.raw_dir, out_dir, vocab, hubert_extractor, f0_extractor
        )
    elif args.dataset == "gtsinger":
        metadata = prepare_gtsinger(
            args.raw_dir, out_dir, vocab, args.language, hubert_extractor, f0_extractor
        )

    # Assign splits and save metadata
    metadata = assign_splits(metadata,
                             val_ratio=config["data"]["val_ratio"],
                             test_ratio=config["data"]["test_ratio"])

    with open(Path(out_dir) / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Processed {len(metadata)} samples → {out_dir}")

    # Save vocab
    vocab.save(Path(out_dir) / "vocab.yaml")

    # Compute HuBERT statistics
    stats = compute_hubert_statistics(Path(out_dir) / "hubert")
    torch.save(stats, Path(out_dir) / "hubert_stats.pt")


def merge_datasets(sources: list, output_dir: str):
    """Merge multiple processed dataset directories into one."""
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []

    for src_dir in sources:
        src_dir = Path(src_dir)
        meta_path = src_dir / "metadata.json"
        if not meta_path.exists():
            logger.warning(f"No metadata.json in {src_dir}, skipping")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        dataset_name = src_dir.name
        logger.info(f"Merging {dataset_name}: {len(metadata)} samples")

        # Link feature files with dataset prefix.
        # Use hardlinks (os.link) to avoid duplicating disk usage and avoid the
        # 30+ minute copy overhead for large datasets like ace_opencpop (~90k files).
        # NTFS supports hardlinks natively without admin privileges. Falls back
        # to copy on cross-volume merges or systems where link is unavailable.
        # NOTE: when updating a source's files in-place (e.g., re-extracting F0),
        # atomic rename breaks the hardlink → re-run merge with dst already deleted
        # for the affected subdir to re-establish the link. See _reextract_f0_gtsinger.py.
        for subdir in ["hubert", "f0", "phonemes", "pitch", "duration", "energy", "breathiness", "tech"]:
            src_feat = src_dir / subdir
            if not src_feat.exists():
                continue
            dst_feat = output_dir / subdir
            dst_feat.mkdir(parents=True, exist_ok=True)

            for pt_file in src_feat.rglob("*.pt"):
                rel = pt_file.relative_to(src_feat)
                new_name = f"{dataset_name}_{rel}"
                dst = dst_feat / new_name
                if dst.exists():
                    continue
                try:
                    import os
                    os.link(pt_file, dst)
                except (OSError, NotImplementedError):
                    shutil.copy2(pt_file, dst)

        # Update metadata IDs with dataset prefix
        for m in metadata:
            m["id"] = f"{dataset_name}_{m['id']}"
            all_metadata.append(m)

    # Re-assign splits on merged data
    from src.preprocessing.prepare_dataset import assign_splits
    all_metadata = assign_splits(all_metadata)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Merged {len(all_metadata)} total samples → {output_dir}")


if __name__ == "__main__":
    main()
