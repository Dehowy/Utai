#!/usr/bin/env python3
"""Train K-means codebook on HuBERT features and quantize to discrete tokens.

Usage:
    python scripts/quantize_hubert.py --processed-dir data/processed/multi_lang_4 \
        --n-codes 1024 --sample-frames 3000000

Outputs:
    <processed_dir>/kmeans_<n_codes>.pt   — codebook centroids [n_codes, 768]
    <processed_dir>/hubert_tokens/<id>.pt — per-utterance token indices [T] int16
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def sample_frames(hubert_dir: Path, metadata: list, n_frames: int, seed: int = 42):
    """Randomly sample n_frames from the dataset for K-means training."""
    rng = np.random.RandomState(seed)

    # Compute total available frames
    total = sum(m["n_frames"] for m in metadata)
    ratio = min(n_frames / total, 1.0)
    logger.info(f"Sampling {n_frames:,} frames from {total:,} total ({ratio:.1%})")

    frames = []
    collected = 0
    indices = list(range(len(metadata)))
    rng.shuffle(indices)

    for idx in indices:
        if collected >= n_frames:
            break
        m = metadata[idx]
        path = hubert_dir / f"{m['id']}.pt"
        if not path.exists():
            continue
        feat = torch.load(path, weights_only=True).float().numpy()  # [T, 768]
        # Sub-sample within utterance if we only need a fraction
        n = feat.shape[0]
        if ratio < 0.5:
            keep = max(1, int(n * ratio * 2))  # oversample per-file to compensate skipping
            sel = rng.choice(n, size=min(keep, n), replace=False)
            feat = feat[sel]
        frames.append(feat)
        collected += feat.shape[0]

    all_frames = np.concatenate(frames, axis=0)
    if all_frames.shape[0] > n_frames:
        sel = rng.choice(all_frames.shape[0], size=n_frames, replace=False)
        all_frames = all_frames[sel]

    logger.info(f"Collected {all_frames.shape[0]:,} frames for K-means")
    return all_frames


def train_kmeans(frames: np.ndarray, n_codes: int, seed: int = 42) -> np.ndarray:
    """Train MiniBatchKMeans and return centroids [n_codes, 768]."""
    logger.info(f"Training MiniBatchKMeans(n_clusters={n_codes}) on {frames.shape[0]:,} frames...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_codes,
        batch_size=10000,
        max_iter=300,
        random_state=seed,
        verbose=0,
        init="k-means++",
    )
    kmeans.fit(frames)
    logger.info(f"K-means converged. Inertia: {kmeans.inertia_:.2f}")
    return kmeans.cluster_centers_  # [n_codes, 768]


def quantize_all(hubert_dir: Path, out_dir: Path, centroids: np.ndarray,
                 metadata: list) -> dict:
    """Quantize all utterances and save token files. Returns quality stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    centroids_t = torch.from_numpy(centroids).float()  # [K, 768]

    l1_errors = []
    cos_sims = []
    token_counts = np.zeros(centroids.shape[0], dtype=np.int64)
    n_done = 0

    for m in metadata:
        path = hubert_dir / f"{m['id']}.pt"
        if not path.exists():
            continue

        feat = torch.load(path, weights_only=True).float()  # [T, 768]

        # Nearest centroid assignment (batched L2)
        # dists[i,j] = ||feat[i] - centroids[j]||^2
        dists = torch.cdist(feat.unsqueeze(0), centroids_t.unsqueeze(0)).squeeze(0)  # [T, K]
        tokens = dists.argmin(dim=1)  # [T]

        # Save as int16
        torch.save(tokens.to(torch.int16), out_dir / f"{m['id']}.pt")

        # Quality stats
        quantized = centroids_t[tokens]  # [T, 768]
        l1 = (feat - quantized).abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(feat, quantized, dim=-1).mean().item()
        l1_errors.append(l1)
        cos_sims.append(cos)

        for t in tokens.numpy():
            token_counts[t] += 1

        n_done += 1
        if n_done % 10000 == 0:
            logger.info(f"  quantized {n_done:,}/{len(metadata):,}")

    stats = {
        "n_utterances": n_done,
        "mean_l1": float(np.mean(l1_errors)),
        "mean_cos": float(np.mean(cos_sims)),
        "min_cos": float(np.min(cos_sims)),
        "max_l1": float(np.max(l1_errors)),
        "dead_tokens": int((token_counts == 0).sum()),
        "min_token_count": int(token_counts.min()),
        "max_token_count": int(token_counts.max()),
        "median_token_count": int(np.median(token_counts)),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Quantize HuBERT features to discrete tokens")
    parser.add_argument("--processed-dir", type=str, default="data/processed/multi_lang_4")
    parser.add_argument("--n-codes", type=int, default=1024)
    parser.add_argument("--sample-frames", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    proc = Path(args.processed_dir)
    hubert_dir = proc / "hubert"
    meta_path = proc / "metadata.json"

    logger.info(f"Loading metadata from {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info(f"Total entries: {len(metadata):,}")

    # Step 1: Sample frames for K-means
    frames = sample_frames(hubert_dir, metadata, args.sample_frames, args.seed)

    # Step 2: Train K-means
    centroids = train_kmeans(frames, args.n_codes, args.seed)

    # Step 3: Save codebook
    codebook_path = proc / f"kmeans_{args.n_codes}.pt"
    torch.save(torch.from_numpy(centroids).float(), codebook_path)
    logger.info(f"Saved codebook: {codebook_path} [{args.n_codes}, {centroids.shape[1]}]")

    # Step 4: Quantize all utterances
    token_dir = proc / "hubert_tokens"
    logger.info(f"Quantizing {len(metadata):,} utterances → {token_dir}")
    stats = quantize_all(hubert_dir, token_dir, centroids, metadata)

    # Step 5: Report
    logger.info("=" * 50)
    logger.info("Quantization quality report:")
    logger.info(f"  Utterances quantized: {stats['n_utterances']:,}")
    logger.info(f"  Mean L1 error:        {stats['mean_l1']:.4f}")
    logger.info(f"  Mean cosine sim:      {stats['mean_cos']:.4f}")
    logger.info(f"  Worst L1:             {stats['max_l1']:.4f}")
    logger.info(f"  Worst cosine:         {stats['min_cos']:.4f}")
    logger.info(f"  Dead tokens (unused): {stats['dead_tokens']}/{args.n_codes}")
    logger.info(f"  Token count range:    {stats['min_token_count']}-{stats['max_token_count']} "
                f"(median {stats['median_token_count']})")
    logger.info("=" * 50)

    # Gate check
    if stats["mean_cos"] < 0.9:
        logger.warning(f"⚠ Mean cosine {stats['mean_cos']:.4f} < 0.9 — codebook may be too small. "
                       f"Consider increasing --n-codes to 2048.")
    if stats["dead_tokens"] > args.n_codes * 0.1:
        logger.warning(f"⚠ {stats['dead_tokens']} dead tokens ({stats['dead_tokens']/args.n_codes:.0%}) "
                       f"— codebook may be too large.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
