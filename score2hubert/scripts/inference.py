#!/usr/bin/env python3
"""
Score2HuBERT Inference Script.

Usage:
    # Basic synthesis (Score2HuBERT only, outputs HuBERT features)
    python scripts/inference.py --model checkpoints/best.pt \
        --phonemes "ja_k ja_o ja_n ja_n ja_i ja_ch ja_i ja_w ja_a" \
        --pitches "60 60 62 62 64 64 64 62 62" \
        --durations "30 30 30 30 30 30 30 30 60"

    # Full pipeline with RVC decoder
    python scripts/inference.py --model checkpoints/best.pt \
        --rvc-model RVC/RVC20240604Nvidia/assets/weights/Iroha_e175_s6300.pth \
        --phonemes "..." --pitches "..." --durations "..." \
        --output output.wav

    # From USTX file (parse OpenUtau project)
    python scripts/inference.py --model checkpoints/best.pt \
        --rvc-model RVC/RVC20240604Nvidia/assets/weights/<model>.pth \
        --ustx project.ustx --output output.wav

    # Rap mode
    python scripts/inference.py --model checkpoints/best.pt \
        --singing-ratio 0.0 --phonemes "..." --durations "..."
"""

import sys
import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Score2HuBERT Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--rvc-model", type=str, default=None, help="Path to RVC model for audio output")
    parser.add_argument("--compensation", type=str, default=None, help="RVC compensation stats file")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio path")

    # Input options (manual)
    parser.add_argument("--phonemes", type=str, help="Space-separated phoneme sequence")
    parser.add_argument("--pitches", type=str, help="Space-separated MIDI pitches")
    parser.add_argument("--durations", type=str, help="Space-separated frame durations")
    parser.add_argument("--language", type=str, default="ja")
    parser.add_argument("--singing-ratio", type=float, default=1.0)

    # Input options (file)
    parser.add_argument("--ustx", type=str, help="USTX project file")
    parser.add_argument("--json-input", type=str, help="JSON input file")

    # Generation parameters
    parser.add_argument("--steps", type=int, default=10, help="Flow matching ODE steps")
    parser.add_argument("--solver", type=str, default="midpoint", choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--cfg-scale", type=float, default=2.0, help="Guidance scale")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pitch-shift", type=int, default=0, help="RVC pitch shift (semitones)")

    args = parser.parse_args()

    from src.inference.synthesizer import Score2HuBERTSynthesizer
    from src.data.phoneme import PhonemeVocab

    vocab = PhonemeVocab()

    print(f"Loading model from {args.model}...")
    synth = Score2HuBERTSynthesizer(
        model_path=args.model,
        rvc_model_path=args.rvc_model,
        vocab=vocab,
        compensation_stats=args.compensation,
    )

    # Parse input
    if args.json_input:
        with open(args.json_input, "r") as f:
            data = json.load(f)
        phonemes = data["phonemes"]
        pitches = data["pitches"]
        durations = data["durations"]
        language = data.get("language", args.language)
        singing_ratio = data.get("singing_ratio", args.singing_ratio)
    elif args.phonemes:
        phonemes = args.phonemes.split()
        pitches = [int(p) for p in args.pitches.split()]
        durations = [int(d) for d in args.durations.split()]
        language = args.language
        singing_ratio = args.singing_ratio
    else:
        parser.error("Provide --phonemes or --json-input or --ustx")

    print(f"Synthesizing: {len(phonemes)} phonemes, {sum(durations)} frames "
          f"({sum(durations)/100:.2f}s)")
    print(f"  Steps={args.steps}, Solver={args.solver}, CFG={args.cfg_scale}, "
          f"Singing ratio={singing_ratio}")

    result = synth.synthesize(
        phonemes=phonemes,
        pitches=pitches,
        durations=durations,
        language=language,
        singing_ratio=singing_ratio,
        n_steps=args.steps,
        solver=args.solver,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        pitch_shift=args.pitch_shift,
    )

    if result["audio"] is not None:
        synth.save_audio(result["audio"], result["sr"], args.output)
        print(f"Audio saved: {args.output}")
    else:
        # Save HuBERT features for later use
        import torch
        feat_path = args.output.replace(".wav", "_hubert.pt")
        f0_path = args.output.replace(".wav", "_f0.pt")
        torch.save(torch.from_numpy(result["hubert"]), feat_path)
        torch.save(torch.from_numpy(result["f0"]), f0_path)
        print(f"Features saved: {feat_path}, {f0_path}")
        print("(No RVC model specified — audio not generated)")


if __name__ == "__main__":
    main()
