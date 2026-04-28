"""Full GTSinger preprocessing for Phase 2 multilingual training.

Processes Chinese / Japanese / English subsets (singing + paired speech) into
data/processed/gtsinger_{zh,ja,en}/. Each language is processed independently
so a failure in one doesn't lose the others.

Usage: venv/Scripts/python.exe _preprocess_gtsinger_full.py [zh|ja|en|all]
"""

import sys
import json
import time
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from src.data.phoneme import PhonemeVocab
from src.preprocessing.prepare_dataset import prepare_gtsinger, assign_splits
from src.preprocessing.extract_hubert import HuBERTExtractor, compute_hubert_statistics
from src.preprocessing.extract_f0 import F0Extractor

PROJECT_ROOT = Path(__file__).parent
GT_ROOT = PROJECT_ROOT / "GTSinger"
HUBERT_MODEL = PROJECT_ROOT / "RVC/RVC20240604Nvidia/assets/hubert/hubert_base.pt"
RMVPE_MODEL = PROJECT_ROOT / "RVC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt"

LANGUAGES = ["zh", "ja", "en"]

# Allow processing a subset via CLI
if len(sys.argv) > 1 and sys.argv[1] != "all":
    LANGUAGES = [sys.argv[1]]

VAL_RATIO = 0.05
TEST_RATIO = 0.05


def process_one_language(lang: str, hubert_extractor, f0_extractor, vocab):
    out_dir = PROJECT_ROOT / "data" / "processed" / f"gtsinger_{lang}"
    print()
    print("=" * 70)
    print(f"Language: {lang}  →  {out_dir}")
    print("=" * 70)

    # Wipe previous output for this language to ensure clean run
    if out_dir.exists():
        print(f"Removing existing {out_dir}...")
        shutil.rmtree(out_dir)

    t0 = time.time()
    metadata = prepare_gtsinger(
        raw_dir=str(GT_ROOT),
        output_dir=str(out_dir),
        vocab=vocab,
        language=lang,
        hubert_extractor=hubert_extractor,
        f0_extractor=f0_extractor,
        include_speech=True,
    )
    t1 = time.time()
    print(f"\nprepare_gtsinger took {t1 - t0:.0f}s, returned {len(metadata)} entries")

    if not metadata:
        print(f"ERROR: no entries produced for {lang}!")
        return None

    # Split breakdown
    n_sing = sum(1 for m in metadata if m["singing_ratio"] == 1.0)
    n_speech = sum(1 for m in metadata if m["singing_ratio"] == 0.0)
    print(f"  singing: {n_sing}, speech: {n_speech}")

    # Assign splits
    metadata = assign_splits(metadata, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)
    n_train = sum(1 for m in metadata if m.get("split") == "train")
    n_val = sum(1 for m in metadata if m.get("split") == "val")
    n_test = sum(1 for m in metadata if m.get("split") == "test")
    print(f"  splits: train={n_train}, val={n_val}, test={n_test}")

    # Save metadata.json
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save vocab
    vocab.save(str(out_dir / "vocab.yaml"))

    # Compute HuBERT stats
    print("Computing HuBERT statistics...")
    stats = compute_hubert_statistics(out_dir / "hubert")
    torch.save(stats, out_dir / "hubert_stats.pt")
    print(f"  hubert mean: {stats['mean'].mean().item():.4f}, std: {stats['std'].mean().item():.4f}")

    print(f"Done with {lang}. Total time: {time.time() - t0:.0f}s")
    return metadata


def main():
    print("Loading models...")
    hubert_extractor = HuBERTExtractor(str(HUBERT_MODEL))
    f0_extractor = F0Extractor(method="rmvpe", rmvpe_model_path=str(RMVPE_MODEL))
    vocab = PhonemeVocab()
    print(f"Vocab size: {vocab.size}, θ at id {vocab.phone2id['θ']}, ð at id {vocab.phone2id['ð']}")

    summary = {}
    for lang in LANGUAGES:
        try:
            meta = process_one_language(lang, hubert_extractor, f0_extractor, vocab)
            if meta:
                summary[lang] = {
                    "total": len(meta),
                    "singing": sum(1 for m in meta if m["singing_ratio"] == 1.0),
                    "speech": sum(1 for m in meta if m["singing_ratio"] == 0.0),
                }
        except Exception as e:
            print(f"FAILED for {lang}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for lang, s in summary.items():
        print(f"  {lang}: total={s['total']:>6}  singing={s['singing']:>6}  speech={s['speech']:>6}")
    grand_total = sum(s["total"] for s in summary.values())
    print(f"  GRAND TOTAL: {grand_total}")


if __name__ == "__main__":
    main()
