"""Smoke test for the rewritten prepare_gtsinger.

Runs preprocessing on a small subset (singer JA-Tenor-1, max 20 singing + 20 speech entries),
writes to data/processed/_smoke_gtsinger/, and prints diagnostic info on outputs.

Usage: venv/Scripts/python.exe _smoke_gtsinger.py
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from src.data.phoneme import PhonemeVocab
from src.preprocessing.prepare_dataset import prepare_gtsinger
from src.preprocessing.extract_hubert import HuBERTExtractor
from src.preprocessing.extract_f0 import F0Extractor

PROJECT_ROOT = Path(__file__).parent
GT_ROOT = PROJECT_ROOT / "GTSinger"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "_smoke_gtsinger"

print(f"Project root : {PROJECT_ROOT}")
print(f"GTSinger root: {GT_ROOT}")
print(f"Output dir   : {OUT_DIR}")
print()

# Initialize models
print("Loading HuBERT and F0 extractors (warm-up)...")
hubert_model = PROJECT_ROOT / "RVC/RVC20240604Nvidia/assets/hubert/hubert_base.pt"
rmvpe_model  = PROJECT_ROOT / "RVC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt"

hubert_extractor = HuBERTExtractor(str(hubert_model))
f0_extractor = F0Extractor(method="rmvpe", rmvpe_model_path=str(rmvpe_model))
vocab = PhonemeVocab()
print(f"Vocab size: {vocab.size}")
print(f"Token 73/74: {vocab.id2phone[73]} / {vocab.id2phone[74]}")
print()

# Wipe previous smoke output
import shutil
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

# Run preprocessing on JA-Tenor-1 with cap of 20 singing + 20 speech
print("=" * 60)
print("Running prepare_gtsinger smoke test...")
print("  language=ja, singer_filter=JA-Tenor-1, max_entries=20")
print("=" * 60)
metadata = prepare_gtsinger(
    raw_dir=str(GT_ROOT),
    output_dir=str(OUT_DIR),
    vocab=vocab,
    language="ja",
    hubert_extractor=hubert_extractor,
    f0_extractor=f0_extractor,
    include_speech=True,
    singer_filter="JA-Tenor-1",
    max_entries=20,
)

print()
print("=" * 60)
print(f"Result: {len(metadata)} entries in metadata")
print("=" * 60)

# Aggregate stats
singing_count = sum(1 for m in metadata if m["singing_ratio"] == 1.0)
speech_count = sum(1 for m in metadata if m["singing_ratio"] == 0.0)
print(f"  singing entries: {singing_count}")
print(f"  speech entries : {speech_count}")
print()

# Check files on disk
for subdir in ["hubert", "f0", "phonemes", "pitch", "duration"]:
    files = list((OUT_DIR / subdir).glob("*.pt"))
    print(f"  {subdir}/: {len(files)} .pt files")
print()

# Inspect first singing and first speech entry
print("=" * 60)
print("First singing entry:")
print("=" * 60)
sing_meta = next((m for m in metadata if m["singing_ratio"] == 1.0), None)
if sing_meta:
    for k, v in sing_meta.items():
        print(f"  {k}: {v}")
    utt_id = sing_meta["id"]
    phs = torch.load(OUT_DIR / "phonemes" / f"{utt_id}.pt")
    pitches = torch.load(OUT_DIR / "pitch" / f"{utt_id}.pt")
    durs = torch.load(OUT_DIR / "duration" / f"{utt_id}.pt")
    hubert = torch.load(OUT_DIR / "hubert" / f"{utt_id}.pt")
    f0 = torch.load(OUT_DIR / "f0" / f"{utt_id}.pt")
    print(f"  phonemes shape: {phs.shape}, decoded: {[vocab.id2phone[i.item()] for i in phs[:10]]}...")
    print(f"  pitches shape : {pitches.shape}, first 10: {pitches[:10].tolist()}")
    print(f"  durations frames sum: {durs.sum().item()}")
    print(f"  hubert shape  : {hubert.shape}")
    print(f"  f0 shape      : {f0.shape}, mean: {f0[f0>50].mean().item():.1f} Hz (voiced)")

print()
print("=" * 60)
print("First speech entry:")
print("=" * 60)
speech_meta = next((m for m in metadata if m["singing_ratio"] == 0.0), None)
if speech_meta:
    for k, v in speech_meta.items():
        print(f"  {k}: {v}")
    utt_id = speech_meta["id"]
    phs = torch.load(OUT_DIR / "phonemes" / f"{utt_id}.pt")
    pitches = torch.load(OUT_DIR / "pitch" / f"{utt_id}.pt")
    durs = torch.load(OUT_DIR / "duration" / f"{utt_id}.pt")
    hubert = torch.load(OUT_DIR / "hubert" / f"{utt_id}.pt")
    f0 = torch.load(OUT_DIR / "f0" / f"{utt_id}.pt")
    print(f"  phonemes shape: {phs.shape}, decoded: {[vocab.id2phone[i.item()] for i in phs[:10]]}...")
    print(f"  pitches shape : {pitches.shape}, first 10: {pitches[:10].tolist()} (should all be 0)")
    print(f"  durations frames sum: {durs.sum().item()}")
    print(f"  hubert shape  : {hubert.shape}")
    voiced_f0 = f0[f0 > 50]
    if len(voiced_f0) > 0:
        print(f"  f0 shape      : {f0.shape}, mean voiced: {voiced_f0.mean().item():.1f} Hz")
    else:
        print(f"  f0 shape      : {f0.shape}, NO VOICED FRAMES (suspicious for speech)")

# Phoneme distribution sanity check (across all entries)
print()
print("=" * 60)
print("Phoneme distribution sanity check (across all entries):")
print("=" * 60)
all_phs = []
for m in metadata:
    phs = torch.load(OUT_DIR / "phonemes" / f"{m['id']}.pt")
    all_phs.extend(phs.tolist())

counter = Counter(vocab.id2phone[i] for i in all_phs)
print(f"  Total phoneme tokens: {len(all_phs)}")
print(f"  Unique tokens used: {len(counter)}")
print(f"  Top 15: {counter.most_common(15)}")

sp_count = counter.get("SP", 0)
sp_ratio = sp_count / len(all_phs) if all_phs else 0
print(f"  SP token ratio: {sp_ratio:.1%} ({sp_count}/{len(all_phs)})")
if sp_ratio > 0.4:
    print("  ⚠️  WARNING: high SP ratio suggests G2P fallback is hitting unexpected phonemes")
elif sp_ratio < 0.05:
    print("  Note: low SP ratio is OK (some Japanese has minimal silence markers)")
else:
    print("  Healthy SP ratio.")
