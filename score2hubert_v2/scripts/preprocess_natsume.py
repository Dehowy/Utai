"""Score2HuBERT v2 — Natsume (夏目悠李) Singing DB preprocessing.

Source: https://ksdcm1ng.wixsite.com/njksofficial/enunu-nnsvs
GitHub: Natsume_Singing_DB_0713 (label-only repo)
Author: 夏目悠李 (Natsume Yuli) — male voice
License: 利用規約 https://ksdcm1ng.wixsite.com/njksofficial/規約-rules

Structure (modeled after 東北きりたんDB):
  Natsume_Singing_DB_0713/
    wav/{N}.wav          # 52 songs
    mono_label/{N}.lab   # float-seconds OpenJTalk phonemes (Kiritan format)

Reuses preprocess_kiritan_itako.py with lab_fmt='sec' and a wav-dir/lab-dir layout.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import argparse
import json
import logging
import shutil

from tqdm import tqdm

from src.data.phoneme_v1 import PhonemeVocab
from src.preprocessing.extract_hubert import HuBERTExtractor
from preprocess_kiritan_itako import process_song, FRAME_RATE  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("preprocess_natsume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "Natsume_Singing_DB_0713"))
    ap.add_argument("--out-root", default=str(REPO_ROOT / "data" / "processed"))
    ap.add_argument("--hubert-model",
                    default=str(REPO_ROOT / "RVC/RVC20240604Nvidia/assets/hubert/hubert_base.pt"))
    ap.add_argument("--rmvpe-model",
                    default=str(REPO_ROOT / "RVC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt"))
    ap.add_argument("--vocab",
                    default=str(REPO_ROOT / "data/processed/ace_opencpop/vocab.yaml"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    vocab = PhonemeVocab(args.vocab)

    logger.info("Loading HuBERT...")
    hubert_ext = HuBERTExtractor(args.hubert_model, layer=12, device=args.device, use_fp16=args.fp16)
    hubert_model = hubert_ext.model

    logger.info("Loading RMVPE...")
    sys.path.insert(0, str(REPO_ROOT / "RVC" / "RVC20240604Nvidia"))
    sys.path.insert(0, str(REPO_ROOT / "RVC" / "RVC20240604Nvidia" / "infer"))
    from infer.lib.rmvpe import RMVPE
    rmvpe = RMVPE(args.rmvpe_model, is_half=args.fp16, device=args.device)

    out_dir = Path(args.out_root) / "natsume"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = {n: (out_dir / n) for n in ("hubert", "f0", "phonemes", "pitch", "duration")}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.vocab, out_dir / "vocab.yaml")

    root = Path(args.root)
    wav_dir = root / "wav"
    lab_dir = root / "mono_label"

    wav_files = sorted(wav_dir.glob("*.wav"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    logger.info(f"[natsume] {len(wav_files)} songs in {wav_dir}")

    all_meta = []
    pbar = tqdm(wav_files, desc="natsume", unit="song")
    for wav_path in pbar:
        song_id = wav_path.stem
        lab_path = lab_dir / f"{song_id}.lab"
        if not lab_path.exists():
            continue
        try:
            meta = process_song(
                wav_path=wav_path,
                lab_path=lab_path,
                lab_fmt="sec",  # Kiritan-style float seconds
                song_id=song_id,
                hubert_model=hubert_model,
                rmvpe=rmvpe,
                vocab=vocab,
                device=args.device,
                use_fp16=args.fp16,
                out_dirs=out_dirs,
                dataset="natsume",
                singer="natsume_yuli",
                id_prefix="natsume",
            )
            all_meta.extend(meta)
        except Exception as e:
            logger.warning(f"failed song {song_id}: {e}")
        pbar.set_postfix(segs=len(all_meta))

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=0)

    total_frames = sum(e["n_frames"] for e in all_meta)
    logger.info(f"[natsume] {len(all_meta)} segs, {total_frames} frames, "
                f"{total_frames/FRAME_RATE/3600:.2f}h")


if __name__ == "__main__":
    main()
