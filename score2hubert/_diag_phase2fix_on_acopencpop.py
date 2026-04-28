"""Diagnostic: Phase 2 fix model (finetune_multi_det_fix/best.pt) on ACE-Opencpop test samples.

Purpose: verify the theory that the "fuzzy articulation" observed on GTSinger samples
is due to data-distribution difficulty (human expressive singing), NOT model regression.

If the Phase 2 fix model produces CLEAR audio on ACE-Opencpop samples (comparable to
Plan C quality), it proves:
  - The det_head hasn't degraded across fine-tuning
  - The fuzzy articulation on GTSinger is inherent to human-singer data variance
  - Fixing GTSinger quality needs the hybrid arch (det + flow refiner), not more training

Output: test_outputs_phase2fix_acopencpop_diag/sample{N}_{variant}.wav
"""

import sys
import json
import random
import logging
from pathlib import Path

import torch

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from eval_audio import load_model, load_rvc, rvc_decode, save_audio

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CKPT = PROJECT / "checkpoints/finetune_multi_det_fix/best.pt"
RVC = PROJECT / "RVC/RVC20240604Nvidia/assets/weights/yachiyo_runami_e350_s10500.pth"
PROCESSED = PROJECT / "data/processed/ace_opencpop"
OUT = PROJECT / "test_outputs_phase2fix_acopencpop_diag"
N_SAMPLES = 5
MIN_FRAMES = 350  # ACE-Opencpop segments cap ~397 frames (~8 sec)
MAX_FRAMES = 500
SEED = 42


def pick_samples(processed_dir: Path, n: int, rng: random.Random):
    with open(processed_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    candidates = [
        m for m in meta
        if m.get("split") == "test"
        and MIN_FRAMES <= m.get("n_frames", 0) <= MAX_FRAMES
    ]
    rng.shuffle(candidates)
    picks = candidates[:n]
    picks.sort(key=lambda m: -m["n_frames"])  # longest first
    return picks


def load_sample(processed_dir: Path, sample_id: str):
    sample = {
        "phonemes": torch.load(processed_dir / "phonemes" / f"{sample_id}.pt", weights_only=True).long(),
        "pitch":    torch.load(processed_dir / "pitch"    / f"{sample_id}.pt", weights_only=True).long(),
        "duration": torch.load(processed_dir / "duration" / f"{sample_id}.pt", weights_only=True).long(),
        "hubert":   torch.load(processed_dir / "hubert"   / f"{sample_id}.pt", weights_only=True).float(),
        "f0":       torch.load(processed_dir / "f0"       / f"{sample_id}.pt", weights_only=True).float(),
    }
    tech_path = processed_dir / "tech" / f"{sample_id}.pt"
    if tech_path.exists():
        sample["tech"] = torch.load(tech_path, weights_only=True).float()
    else:
        sample["tech"] = torch.zeros(sample["phonemes"].size(0), 6, dtype=torch.float32)
    return sample


@torch.no_grad()
def synthesize(model, sample, device):
    phonemes_b = sample["phonemes"].unsqueeze(0).to(device)
    pitch_b    = sample["pitch"].unsqueeze(0).to(device)
    duration_b = sample["duration"].unsqueeze(0).to(device)
    sr_b = torch.tensor([1.0], dtype=torch.float32).to(device)
    lang_b = torch.tensor([1], dtype=torch.long).to(device)  # zh = 1 in vocab
    tech_b = sample["tech"].unsqueeze(0).to(device).float()  # [1, T_phone, 6]
    output = model.synthesize(
        phonemes=phonemes_b, pitch=pitch_b, duration=duration_b,
        singing_ratio=sr_b, lang_id=lang_b,
        n_steps=1, solver="euler", cfg_scale=1.0, temperature=1.0,
        tech=tech_b,
    )
    return output["hubert"][0].float(), output["f0"][0].float()


def main():
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    logger.info(f"Loading Phase 2 fix model from {CKPT}")
    model, _ = load_model(str(CKPT), device)
    logger.info(f"Loading RVC from {RVC}")
    net_g, sid, sr_audio = load_rvc(str(RVC), device)

    picks = pick_samples(PROCESSED, N_SAMPLES, rng)
    logger.info(f"Picked {len(picks)} ace_opencpop test samples: "
                f"frames {picks[-1]['n_frames']}-{picks[0]['n_frames']} "
                f"(≈{picks[-1]['n_frames']/50:.1f}-{picks[0]['n_frames']/50:.1f}s)")

    for i, m in enumerate(picks, start=1):
        sid_str = m["id"]
        n_frames = m["n_frames"]
        logger.info(f"[{i}/{len(picks)}] {sid_str} ({n_frames} frames ≈ {n_frames/50:.1f}s)")

        sample = load_sample(PROCESSED, sid_str)
        pred_h, pred_f = synthesize(model, sample, device)

        # Align to common length
        T = min(pred_h.size(0), pred_f.size(0), sample["hubert"].size(0), sample["f0"].size(0))
        pred_h = pred_h[:T]
        pred_f = pred_f[:T]
        gt_h = sample["hubert"][:T]
        gt_f = sample["f0"][:T]

        variants = {
            "gt_ceiling":  (gt_h,  gt_f),
            "pred_full":   (pred_h, pred_f),
            "h_pred_f0gt": (pred_h, gt_f),
            "hgt_f0pred":  (gt_h,  pred_f),
        }
        for v_name, (h, f) in variants.items():
            audio = rvc_decode(net_g, sid, h, f, device)
            save_audio(audio, sr_audio, OUT / f"sample{i}_{v_name}.wav")
            logger.info(f"  saved {v_name} ({len(audio)/sr_audio:.2f}s)")

    logger.info(f"Done. Outputs in: {OUT}")


if __name__ == "__main__":
    main()
