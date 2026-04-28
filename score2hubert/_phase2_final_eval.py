"""Phase 2 final audio eval — multilingual + speech + experimental rap-in-song.

Generates LONG test samples across multiple categories so you can listen carefully
in a fine-tuning context. Uses Phase 2's `checkpoints/finetune_multi_det/best.pt`.

Categories (each → 5 long samples × 4 ablations = 20 wav files):
  1. zh_singing   — Chinese singing, 12+ sec  (gtsinger_zh test split)
  2. ja_singing   — Japanese singing            (gtsinger_ja test split)
  3. en_singing   — English singing             (gtsinger_en test split)
  4. zh_speech    — Chinese spoken lyrics       (gtsinger_zh, singing_ratio=0)
  5. ja_speech    — Japanese spoken lyrics      (gtsinger_ja, singing_ratio=0)
  6. en_speech    — English spoken lyrics       (gtsinger_en, singing_ratio=0)

Plus EXPERIMENTAL category (1 sample × 1 variant only):
  7. rap_in_song  — long zh singing utterance with the MIDDLE 1/3 of phonemes
                    flagged as singing_ratio=0 (rap), keeping the start and
                    end as singing_ratio=1. Tests transition from sung →
                    rapped → sung mid-utterance, which is the user's actual
                    use case (pop songs with rap interludes).

Output structure:
  test_outputs_phase2_final/
    ├── zh_singing/sample{1..5}_{gt_ceiling,pred_full,h_pred_f0gt,hgt_f0pred}.wav
    ├── ja_singing/...
    ├── en_singing/...
    ├── zh_speech/...
    ├── ja_speech/...
    ├── en_speech/...
    └── rap_in_song/sample{1..5}_pred_full.wav  (single variant)

Usage:
    venv/Scripts/python.exe _phase2_final_eval.py [--rvc-model PATH]
"""

import sys
import json
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.model.score2hubert import Score2HuBERT
from eval_audio import load_model, load_rvc, rvc_decode, save_audio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CKPT_PATH = PROJECT / "checkpoints/finetune_discrete/best.pt"
DEFAULT_RVC = PROJECT / "RVC/RVC20240604Nvidia/assets/weights/yachiyo_runami_e350_s10500.pth"
OUT_ROOT = PROJECT / "test_outputs_phase5_pitchcorrect"
SEED = 42
N_SAMPLES = 8
N_RAP_SAMPLES = 5

# Each category specifies how to find candidate samples.
# (singing categories: longer minimum, speech: shorter because spoken segments are typically shorter)
CATEGORIES = {
    "zh_singing": dict(
        source="data/processed/gtsinger_zh",
        sing_ratio=1.0,
        min_frames=400,   # zh max is 731 (14.6s), so lower threshold to get more candidates
        max_frames=2000,
    ),
    "ja_singing": dict(
        source="data/processed/gtsinger_ja",
        sing_ratio=1.0,
        min_frames=400,   # ja has samples up to 1411 (28.2s)
        max_frames=2000,
    ),
    "en_singing": dict(
        source="data/processed/gtsinger_en",
        sing_ratio=1.0,
        min_frames=400,   # en has samples up to 1006 (20.1s)
        max_frames=2000,
    ),
    "zh_speech": dict(
        source="data/processed/gtsinger_zh",
        sing_ratio=0.0,
        min_frames=150,   # speech is shorter
        max_frames=2000,
    ),
    "ja_speech": dict(
        source="data/processed/gtsinger_ja",
        sing_ratio=0.0,
        min_frames=150,
        max_frames=2000,
    ),
    "en_speech": dict(
        source="data/processed/gtsinger_en",
        sing_ratio=0.0,
        min_frames=150,
        max_frames=2000,
    ),
    # --- Phase 3 technique-targeted categories (the real A/B test) ---
    "breathy_group_zh": dict(
        source="data/processed/gtsinger_zh",
        sing_ratio=1.0,
        min_frames=300,
        max_frames=2000,
        group_filter="Breathy_Group",
    ),
    "pharyngeal_group_zh": dict(
        source="data/processed/gtsinger_zh",
        sing_ratio=1.0,
        min_frames=300,
        max_frames=2000,
        group_filter="Pharyngeal_Group",
    ),
    "control_group_zh": dict(
        source="data/processed/gtsinger_zh",
        sing_ratio=1.0,
        min_frames=300,
        max_frames=2000,
        group_filter="Control_Group",
    ),
    "ace_opencpop": dict(
        source="data/processed/ace_opencpop",
        sing_ratio=1.0,
        min_frames=200,      # ace max is 397 (7.9s), median 210 (4.2s)
        max_frames=2000,
    ),
}


def pick_samples(processed_dir: Path, sing_ratio: float, min_frames: int, max_frames: int,
                 n: int, rng: random.Random, group_filter: str = None):
    """Filter test split, pick half longest + half random for balanced eval.

    Longest samples let the user listen carefully for sustained issues;
    random samples prevent selection bias from only evaluating long utterances.
    Sorted by descending length in output for convenient listening order.

    group_filter: if set, only include entries whose orig_item_name contains
                  '#<group_filter>#' (e.g., 'Breathy_Group').
    """
    with open(processed_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    candidates = [
        m for m in meta
        if m.get("split") == "test"
        and m.get("singing_ratio", 1.0) == sing_ratio
        and min_frames <= m.get("n_frames", 0) <= max_frames
        and (group_filter is None
             or f"#{group_filter}#" in m.get("orig_item_name", ""))
    ]
    if not candidates:
        logger.warning(f"  no candidates in {processed_dir.name} "
                       f"(ratio={sing_ratio}, frames {min_frames}-{max_frames})")
        return []
    candidates.sort(key=lambda m: -m["n_frames"])
    n_long = n // 2
    n_rand = n - n_long
    # Deduplicate by song content: for multi-singer datasets (ace_opencpop),
    # different singers singing the same song produce near-identical output
    # after RVC voice conversion. Use the segment hash from the ID to dedup.
    seen_songs = set()
    deduped = []
    for m in candidates:
        # Extract song identifier: strip singer prefix to find content key.
        # ace_opencpop: "ace_acesinger_10#2013000484" → segment "2013000484"
        # gtsinger: "gtsinger_zh_ZH-Alto-1_breathy_xxx_0006" → hash "xxx_0006"
        sid = m["id"]
        if "#" in sid:
            song_key = sid.split("#", 1)[1]  # ace format
        else:
            parts = sid.rsplit("_", 2)
            song_key = "_".join(parts[-2:]) if len(parts) >= 3 else sid
        if song_key not in seen_songs:
            seen_songs.add(song_key)
            deduped.append(m)
    candidates = deduped if len(deduped) >= n else candidates
    longest = candidates[:n_long]
    remaining = candidates[n_long:]
    rng.shuffle(remaining)
    picks = longest + remaining[:n_rand]
    picks.sort(key=lambda m: -m["n_frames"])
    return picks


def load_sample(processed_dir: Path, sample_id: str):
    """Load all per-utterance tensors for one sample."""
    sample = {
        "phonemes": torch.load(processed_dir / "phonemes" / f"{sample_id}.pt", weights_only=True).long(),
        "pitch":    torch.load(processed_dir / "pitch"    / f"{sample_id}.pt", weights_only=True).long(),
        "duration": torch.load(processed_dir / "duration" / f"{sample_id}.pt", weights_only=True).long(),
        "hubert":   torch.load(processed_dir / "hubert"   / f"{sample_id}.pt", weights_only=True).float(),
        "f0":       torch.load(processed_dir / "f0"       / f"{sample_id}.pt", weights_only=True).float(),
    }
    # Phase 3: load GT technique flags if present (gtsinger samples have them,
    # ace_opencpop and other legacy data don't → fall back to zeros).
    tech_path = processed_dir / "tech" / f"{sample_id}.pt"
    if tech_path.exists():
        sample["tech"] = torch.load(tech_path, weights_only=True).float()
    else:
        sample["tech"] = torch.zeros(sample["phonemes"].size(0), 6, dtype=torch.float32)
    return sample


@torch.no_grad()
def synthesize(model, sample, singing_ratio_value, lang_id_value, device, tech_override=None):
    """Run model.synthesize() with given singing_ratio.

    singing_ratio_value can be:
      - float (e.g., 1.0 or 0.0): expanded to [1] tensor (uniform across utterance)
      - 1D tensor of shape [T_phone]: per-phoneme override (used for rap-in-song)
    tech_override: optional [T_phone, 6] tensor. If None, uses sample["tech"].
    """
    phonemes_b = sample["phonemes"].unsqueeze(0).to(device)
    pitch_b    = sample["pitch"].unsqueeze(0).to(device)
    duration_b = sample["duration"].unsqueeze(0).to(device)

    if isinstance(singing_ratio_value, torch.Tensor):
        sr_b = singing_ratio_value.unsqueeze(0).to(device).float()  # [1, T_phone]
    else:
        sr_b = torch.tensor([singing_ratio_value], dtype=torch.float32).to(device)  # [1]
    lang_b = torch.tensor([lang_id_value], dtype=torch.long).to(device)

    tech_src = tech_override if tech_override is not None else sample.get("tech")
    if tech_src is not None:
        tech_b = tech_src.unsqueeze(0).to(device).float()  # [1, T_phone, 6]
    else:
        tech_b = None

    # n_steps=4 for hybrid refiner ODE; temperature=0.0 because the residual
    # distribution has std~0.18 but OT-CFM noise is std=1.0 (5.7x mismatch).
    # At temp=0 the refiner acts as a deterministic residual corrector (l1
    # improves 0.1251→0.1198, a 4.2% gain). For det mode these are ignored.
    output = model.synthesize(
        phonemes=phonemes_b,
        pitch=pitch_b,
        duration=duration_b,
        singing_ratio=sr_b,
        lang_id=lang_b,
        n_steps=4,
        solver="euler",
        cfg_scale=1.0,
        temperature=1.0,
        tech=tech_b,
    )
    return output["hubert"][0].float(), output["f0"][0].float()


def truncate_common(*tensors):
    T = min(t.size(0) for t in tensors)
    return tuple(t[:T] for t in tensors)


def eval_one_category(name, cfg, model, net_g, sid, sr_audio, device, rng):
    """Process one (non-rap) category: 5 samples × 4 ablations."""
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    src = PROJECT / cfg["source"]

    picks = pick_samples(src, cfg["sing_ratio"], cfg["min_frames"], cfg["max_frames"],
                         N_SAMPLES, rng, group_filter=cfg.get("group_filter"))
    if not picks:
        return

    logger.info(f"=== {name}: {len(picks)} samples "
                f"(frames {picks[-1]['n_frames']}-{picks[0]['n_frames']}, "
                f"≈{picks[-1]['n_frames']/50:.1f}-{picks[0]['n_frames']/50:.1f}s) ===")

    for i, m in enumerate(picks, start=1):
        sid_str = m["id"]
        n_frames = m["n_frames"]
        logger.info(f"  [{i}/{len(picks)}] {sid_str} ({n_frames} frames ≈ {n_frames/50:.1f}s)")

        sample = load_sample(src, sid_str)
        pred_h, pred_f = synthesize(model, sample,
                                     m.get("singing_ratio", 1.0),
                                     m.get("lang_id", 0),
                                     device)
        pred_h, pred_f, gt_h, gt_f = truncate_common(pred_h, pred_f,
                                                      sample["hubert"], sample["f0"])

        variants = {
            "gt_ceiling":  (gt_h,  gt_f),
            "pred_full":   (pred_h, pred_f),
            "h_pred_f0gt": (pred_h, gt_f),
            "hgt_f0pred":  (gt_h,  pred_f),
        }
        for v_name, (h, f) in variants.items():
            audio = rvc_decode(net_g, sid, h, f, device)
            save_audio(audio, sr_audio, out_dir / f"sample{i}_{v_name}.wav")
            logger.info(f"      saved {v_name} ({len(audio)/sr_audio:.2f}s)")


def eval_rap_in_song(model, net_g, sid, sr_audio, device, rng):
    """Experimental: take long zh singing samples and flip middle 1/3 to rap mode.

    Constructs a per-phoneme singing_ratio tensor:
      [1.0] * first_third + [0.0] * middle_third + [1.0] * last_third

    Effect: starts sung, transitions to rap (speech-like F0 from rap_head)
    in the middle, returns to singing at the end. Same lyrics throughout.
    """
    out_dir = OUT_ROOT / "rap_in_song"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = PROJECT / "data/processed/gtsinger_zh"

    picks = pick_samples(src, 1.0, 600, 1200, N_RAP_SAMPLES, rng)
    if not picks:
        logger.warning("rap_in_song: no candidate samples found")
        return

    logger.info(f"=== rap_in_song: {len(picks)} long zh samples, mid-1/3 → rap ===")

    for i, m in enumerate(picks, start=1):
        sid_str = m["id"]
        n_frames = m["n_frames"]
        logger.info(f"  [{i}/{len(picks)}] {sid_str} ({n_frames} frames ≈ {n_frames/50:.1f}s)")

        sample = load_sample(src, sid_str)
        T_phone = sample["phonemes"].size(0)

        # Build per-phoneme singing_ratio: 1 for first/last third, 0 for middle third
        singing_ratio_per_phone = torch.ones(T_phone, dtype=torch.float32)
        third = T_phone // 3
        singing_ratio_per_phone[third:2*third] = 0.0

        # Synthesize with per-phoneme singing_ratio
        pred_h, pred_f = synthesize(model, sample,
                                     singing_ratio_per_phone,  # tensor, not float
                                     m.get("lang_id", 0),
                                     device)
        pred_h, pred_f = truncate_common(pred_h, pred_f)

        # Save pred_full only — gt_ceiling makes no sense (no GT for the rap section)
        audio = rvc_decode(net_g, sid, pred_h, pred_f, device)
        save_audio(audio, sr_audio, out_dir / f"sample{i}_pred_full.wav")
        logger.info(f"      saved pred_full ({len(audio)/sr_audio:.2f}s)")

        # ALSO save: full singing (no rap transition) for direct A/B comparison
        pred_h_pure, pred_f_pure = synthesize(model, sample, 1.0, m.get("lang_id", 0), device)
        pred_h_pure, pred_f_pure = truncate_common(pred_h_pure, pred_f_pure)
        audio_pure = rvc_decode(net_g, sid, pred_h_pure, pred_f_pure, device)
        save_audio(audio_pure, sr_audio, out_dir / f"sample{i}_pure_singing_for_comparison.wav")
        logger.info(f"      saved pure_singing comparison ({len(audio_pure)/sr_audio:.2f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rvc-model", type=str, default=str(DEFAULT_RVC))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    if not CKPT_PATH.exists():
        logger.error(f"Checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {CKPT_PATH}")
    model, _ = load_model(str(CKPT_PATH), args.device)

    logger.info(f"Loading RVC from {args.rvc_model}")
    net_g, sid, sr_audio = load_rvc(args.rvc_model, args.device)

    # Standard 6 categories
    for name, cfg in CATEGORIES.items():
        eval_one_category(name, cfg, model, net_g, sid, sr_audio, args.device, rng)

    # Experimental rap-in-song
    eval_rap_in_song(model, net_g, sid, sr_audio, args.device, rng)

    logger.info(f"Done. All outputs in: {OUT_ROOT}")


if __name__ == "__main__":
    main()
