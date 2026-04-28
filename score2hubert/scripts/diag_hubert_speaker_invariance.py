#!/usr/bin/env python3
"""
Diagnostic: how speaker-invariant are HuBERT features in ACE-Opencpop?

Hypothesis (from user, 2026-04-09): HuBERT was designed to remove timbre and
keep content, so the same lyric sung by different singers should produce very
similar HuBERT features. If true, our score-to-HuBERT model can be trained on
mixed-speaker data without speaker conditioning, and "averaging across speakers"
isn't actually a multi-modal problem because there's nothing to average.

Test setup:
- ACE-Opencpop has 3285 segments sung by all 30 singers (verified)
- Same segment_id across singers has identical n_frames and n_phones, so HuBERT
  features are already temporally aligned (frame-by-frame comparable)
- Pick K random multi-singer segments
- For each segment, compute pairwise cosine similarity across all singer pairs
  at every frame, then aggregate

Baselines for context:
- Cross-singer same-segment cos sim (the hypothesis test) — measured here
- Cross-segment same-singer cos sim (random baseline — what unrelated content
  looks like in HuBERT space) — also measured for comparison
- Same-segment same-singer cos sim — would be 1.0, skipped

Interpretation:
- If cross-singer same-segment cos >> cross-segment same-singer cos, hypothesis
  is confirmed: HuBERT IS largely speaker-invariant for content.
- If they're similar, HuBERT preserves a lot of speaker info and the user's
  hypothesis is wrong.

Usage:
    venv/Scripts/python.exe scripts/diag_hubert_speaker_invariance.py
"""

import sys
import json
import random
import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    proc = Path("data/processed/ace_opencpop")
    hubert_dir = proc / "hubert"

    with open(proc / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Group by segment_id (after #)
    segment_to_samples = defaultdict(list)
    for m in meta:
        seg = m["id"].split("#")[1]
        segment_to_samples[seg].append(m)

    # Use only fully-covered segments (all 30 singers, identical n_frames)
    fully_covered = [
        (seg, samples) for seg, samples in segment_to_samples.items()
        if len(samples) == 30 and len(set(s["n_frames"] for s in samples)) == 1
    ]
    logger.info(f"Total fully-covered segments (30 singers, aligned): {len(fully_covered)}")

    # Sample K segments for the test
    K = 100
    rng = random.Random(42)
    rng.shuffle(fully_covered)
    test_segments = fully_covered[:K]
    logger.info(f"Sampling {len(test_segments)} segments for verification")

    # ---- Test 1: cross-singer, same-segment ----
    # For each segment, load all 30 singers' HuBERT, compute pairwise cos at every
    # frame, then average over frames and pairs
    cross_singer_cos = []  # one entry per segment

    for seg_idx, (seg, samples) in enumerate(test_segments):
        # Load all 30 singers' HuBERT for this segment
        hubert_stack = []
        for s in samples:
            h = torch.load(hubert_dir / f"{s['id']}.pt", weights_only=True).float()
            hubert_stack.append(h)
        # Stack to [N=30, T, 768]
        h = torch.stack(hubert_stack)  # [30, T, 768]
        N, T, D = h.shape

        # Compute pairwise cos sim at every frame
        # Normalize first
        h_norm = F.normalize(h, dim=-1)  # [30, T, 768]
        # Pairwise cos: [30, 30, T] = h_norm @ h_norm.transpose(0, 1) for each frame
        # Equivalent: einsum
        pairwise = torch.einsum("itd,jtd->ijt", h_norm, h_norm)  # [30, 30, T]

        # Take upper triangle (excluding diagonal, since diag is self-sim = 1)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        pair_cos = pairwise[mask]  # [N*(N-1)/2, T] = [435, T]
        # Average over pairs and frames for this segment
        seg_mean = pair_cos.mean().item()
        cross_singer_cos.append(seg_mean)

        if (seg_idx + 1) % 20 == 0:
            logger.info(f"  processed {seg_idx + 1}/{len(test_segments)} segments")

    # ---- Test 2: random baseline — cross-segment same-singer ----
    # For comparison, what does cos sim look like for unrelated content?
    # Pick random pairs from the same singer but different segments,
    # truncate to common length, compute cos at every frame
    cross_segment_cos = []

    # Build per-singer index
    singer_to_samples = defaultdict(list)
    for m in meta:
        singer_to_samples[m["singer"]].append(m)

    n_random = 500
    for _ in range(n_random):
        singer = rng.choice(list(singer_to_samples.keys()))
        if len(singer_to_samples[singer]) < 2:
            continue
        s1, s2 = rng.sample(singer_to_samples[singer], 2)
        h1 = torch.load(hubert_dir / f"{s1['id']}.pt", weights_only=True).float()
        h2 = torch.load(hubert_dir / f"{s2['id']}.pt", weights_only=True).float()
        T = min(h1.size(0), h2.size(0))
        if T < 10:
            continue
        h1 = h1[:T]
        h2 = h2[:T]
        cos = F.cosine_similarity(h1, h2, dim=-1).mean().item()
        cross_segment_cos.append(cos)

    # ---- Test 2.5: SINGLE SINGER vs ALL-30-SINGER MEAN ----
    # This is the theoretical cos ceiling for a deterministic regression model:
    # if you perfectly learn the "average voice for this content", how close can
    # you get to any specific singer's actual features?
    singer_vs_mean_cos = []
    for seg, samples in test_segments[:50]:
        hubert_stack = []
        for s in samples:
            h = torch.load(hubert_dir / f"{s['id']}.pt", weights_only=True).float()
            hubert_stack.append(h)
        h = torch.stack(hubert_stack)  # [30, T, 768]
        mean_h = h.mean(dim=0)  # [T, 768], the "averaged singer"
        # Cos between each singer and the mean, frame-by-frame
        for i in range(h.size(0)):
            cos = F.cosine_similarity(h[i], mean_h, dim=-1).mean().item()
            singer_vs_mean_cos.append(cos)

    # ---- Test 3: cross-singer cross-segment baseline ----
    # Most general baseline: random different singers, random different segments
    cross_all_cos = []
    all_samples = list(meta)
    for _ in range(n_random):
        s1, s2 = rng.sample(all_samples, 2)
        if s1["singer"] == s2["singer"]:
            continue
        if s1["id"].split("#")[1] == s2["id"].split("#")[1]:
            continue
        h1 = torch.load(hubert_dir / f"{s1['id']}.pt", weights_only=True).float()
        h2 = torch.load(hubert_dir / f"{s2['id']}.pt", weights_only=True).float()
        T = min(h1.size(0), h2.size(0))
        if T < 10:
            continue
        h1 = h1[:T]
        h2 = h2[:T]
        cos = F.cosine_similarity(h1, h2, dim=-1).mean().item()
        cross_all_cos.append(cos)

    # ---- Report ----
    def stats(xs, label):
        xs_t = torch.tensor(xs)
        mean = xs_t.mean().item()
        std = xs_t.std().item()
        p10 = xs_t.quantile(0.10).item()
        p50 = xs_t.quantile(0.50).item()
        p90 = xs_t.quantile(0.90).item()
        logger.info(f"  {label:<45} N={len(xs):4d}  "
                    f"mean={mean:.4f}  std={std:.4f}  "
                    f"p10={p10:.4f}  p50={p50:.4f}  p90={p90:.4f}")
        return mean, std

    logger.info("")
    logger.info("=" * 90)
    logger.info("HuBERT pairwise cosine similarity statistics")
    logger.info("=" * 90)
    cs_mean, cs_std = stats(cross_singer_cos, "CROSS-SINGER same-segment (the hypothesis)")
    sm_mean, sm_std = stats(singer_vs_mean_cos, "single SINGER vs all-30-MEAN (det ceiling)")
    cseg_mean, cseg_std = stats(cross_segment_cos, "cross-SEGMENT same-singer (baseline 1)")
    call_mean, call_std = stats(cross_all_cos, "cross-singer cross-segment (baseline 2)")
    logger.info("")

    # Interpretation
    logger.info("=" * 90)
    logger.info("Interpretation")
    logger.info("=" * 90)
    delta = cs_mean - cseg_mean
    logger.info(f"  Cross-singer same-segment cos: {cs_mean:.4f}")
    logger.info(f"  Same-singer different-segment cos: {cseg_mean:.4f}")
    logger.info(f"  Delta (singer-invariance signal): {delta:+.4f}")
    logger.info("")
    if cs_mean > 0.85:
        logger.info("  → HuBERT is HIGHLY speaker-invariant (>0.85). User's hypothesis CONFIRMED.")
        logger.info("  → Mixing speakers in training does NOT cause averaging blur.")
    elif cs_mean > 0.70:
        logger.info("  → HuBERT is largely speaker-invariant (0.70-0.85). Hypothesis mostly correct.")
        logger.info("  → Mixing speakers is OK but some residual variance exists.")
    else:
        logger.info("  → HuBERT retains significant speaker variation (<0.70). Hypothesis NOT confirmed.")
        logger.info("  → Speaker conditioning may help reduce blur.")
    logger.info("")
    if delta > 0.20:
        logger.info(f"  → Strong signal: same lyric across singers is much more similar than")
        logger.info(f"     same singer across lyrics ({delta:+.3f} above baseline)")
        logger.info(f"  → HuBERT clearly encodes content > speaker.")


if __name__ == "__main__":
    main()
