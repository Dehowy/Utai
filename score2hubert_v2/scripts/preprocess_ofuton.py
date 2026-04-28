"""Score2HuBERT v2 — OFUTON_P_UTAGOE_DB preprocessing.

Source: same author family as ONIKU (おふとんP歌声DB)
Note: 使用时需在文档/credit 中标注作者 おふとんP

Identical structure to ONIKU:
    OFUTON_P_UTAGOE_DB/OFUTON_P_UTAGOE_DB/{song_name}/{song_name}.{wav,lab,mid,musicxml,ust}
    HTK 100ns labels, OpenJTalk phoneme system

Specs from Note.txt:
- Male voice
- 46 songs, 49 melody patterns, 57 audio files
- ~46:30 of voiced phoneme content (excluding pau/br/cl)
- 96kHz/24bit (some converted from 44.1kHz)
- Pitch-corrected and noise-removed (same caveat as ONIKU)
- BPM ~100
- Spell singing for CVVC coverage
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
logger = logging.getLogger("preprocess_ofuton")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "OFUTON_P_UTAGOE_DB" / "OFUTON_P_UTAGOE_DB"))
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

    out_dir = Path(args.out_root) / "ofuton"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = {n: (out_dir / n) for n in ("hubert", "f0", "phonemes", "pitch", "duration")}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.vocab, out_dir / "vocab.yaml")

    root = Path(args.root)
    song_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    logger.info(f"[ofuton] {len(song_dirs)} song dirs in {root}")

    all_meta = []
    pbar = tqdm(song_dirs, desc="ofuton", unit="song")
    for song_dir in pbar:
        song_name = song_dir.name
        wav_path = song_dir / f"{song_name}.wav"
        lab_path = song_dir / f"{song_name}.lab"
        if not wav_path.exists() or not lab_path.exists():
            continue
        try:
            meta = process_song(
                wav_path=wav_path,
                lab_path=lab_path,
                lab_fmt="htk",  # 100ns units (verified per Note.txt)
                song_id=song_name,
                hubert_model=hubert_model,
                rmvpe=rmvpe,
                vocab=vocab,
                device=args.device,
                use_fp16=args.fp16,
                out_dirs=out_dirs,
                dataset="ofuton",
                singer="ofuton_p",  # おふとんP
                id_prefix="ofuton",
            )
            all_meta.extend(meta)
        except Exception as e:
            logger.warning(f"failed {song_name}: {e}")
        pbar.set_postfix(segs=len(all_meta))

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=0)

    total_frames = sum(e["n_frames"] for e in all_meta)
    logger.info(f"[ofuton] {len(all_meta)} segs, {total_frames} frames, "
                f"{total_frames/FRAME_RATE/3600:.2f}h")


if __name__ == "__main__":
    main()
