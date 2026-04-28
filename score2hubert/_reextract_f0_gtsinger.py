"""Re-extract F0 for all GTSinger samples and overwrite stored f0/*.pt with
correctly aligned F0 (resampled from RMVPE's 100fps to HuBERT's 50fps via
torch interpolate, NOT truncated).

Fixes a bug in the original prepare_gtsinger that did `f0[:n_frames]` instead
of resampling. The truncation kept only the first half of the F0 contour
(stored as if it were the full duration), causing pitch to play at 2x speed
during decoding (= off-key).

Only re-extracts F0 — keeps hubert/phonemes/pitch/duration .pt files unchanged.
multi_lang_4 hardlinks to gtsinger_*/f0/*.pt files, so this script automatically
fixes the merged dataset too via shared inodes.

Usage: venv/Scripts/python.exe _reextract_f0_gtsinger.py
"""

import sys
import json
import unicodedata
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
sys.stdout.reconfigure(encoding="utf-8")

from src.preprocessing.extract_f0 import F0Extractor

GT_ROOT = PROJECT / "GTSinger"
RMVPE = PROJECT / "RVC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt"

LANG_NAME_MAP = {"zh": "Chinese", "ja": "Japanese", "en": "English"}


def fix_mojibake(name: str) -> str:
    if all(ord(c) < 128 for c in name):
        return name
    try:
        return unicodedata.normalize("NFC", name.encode("cp437").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def build_index(lang_root: Path) -> dict:
    """Walk lang_root and return {NFC-normalized_metadata_wav_fn: actual_disk_path}."""
    index = {}
    for p in lang_root.rglob("*.wav"):
        if "__MACOSX" in p.parts:
            continue
        rel = p.relative_to(lang_root)
        fixed_parts = [fix_mojibake(part) for part in rel.parts]
        index["/".join(fixed_parts)] = p
    return index


def reconstruct_singing_wav_fn(orig_item_name: str) -> str:
    """orig_item_name is e.g. 'Chinese#ZH-Alto-1#Pharyngeal#你就不要想起我#Control_Group#0008'
    Returns 'Chinese/ZH-Alto-1/Pharyngeal/你就不要想起我/Control_Group/0008.wav'"""
    parts = orig_item_name.split("#")
    if len(parts) < 6:
        return ""
    return f"{parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}/{parts[4]}/{parts[5]}.wav"


def find_speech_wav(orig_path: str) -> Path:
    """orig_path is the disk-form rel path to the json file (with mojibake intact).
    Returns the corresponding wav path on disk."""
    if not orig_path:
        return None
    # orig_path is something like 'Japanese\\Japanese\\JA-Tenor-1\\...\\Paired_Speech_Group\\0000.json'
    p = GT_ROOT / orig_path
    return p.with_suffix(".wav")


def reextract_one_lang(lang_code: str, lang_name: str, f0_ext: F0Extractor):
    proc = PROJECT / f"data/processed/gtsinger_{lang_code}"
    print(f"\n=== {lang_code} ({lang_name}) ===")

    lang_root = GT_ROOT / lang_name
    path_index = build_index(lang_root)
    print(f"  Indexed {len(path_index)} wav files")

    with open(proc / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"  {len(meta)} metadata entries")

    n_done = 0
    n_failed_no_wav = 0
    n_failed_no_hubert = 0
    n_failed_extract = 0

    for entry in tqdm(meta, desc=f"{lang_code} f0 reextract"):
        utt_id = entry["id"]

        # Resolve wav path
        if entry.get("singing_ratio", 1.0) == 1.0:
            wav_fn = reconstruct_singing_wav_fn(entry.get("orig_item_name", ""))
            if not wav_fn:
                n_failed_no_wav += 1
                continue
            wav_path = path_index.get(unicodedata.normalize("NFC", wav_fn))
        else:
            wav_path = find_speech_wav(entry.get("orig_path", ""))

        if wav_path is None or not wav_path.exists():
            n_failed_no_wav += 1
            continue

        # Get target frame count from existing hubert .pt
        hubert_path = proc / "hubert" / f"{utt_id}.pt"
        if not hubert_path.exists():
            n_failed_no_hubert += 1
            continue
        try:
            target_n_frames = torch.load(hubert_path, weights_only=True).shape[0]
        except Exception:
            n_failed_no_hubert += 1
            continue

        # Re-extract F0 from wav and resample to target_n_frames
        try:
            f0_raw = f0_ext.extract(wav_path)
            f0_resampled = F.interpolate(
                f0_raw.view(1, 1, -1).float(),
                size=target_n_frames, mode="linear", align_corners=False,
            ).view(-1)
            # Atomic-ish overwrite (write to temp then replace)
            f0_path = proc / "f0" / f"{utt_id}.pt"
            tmp_path = f0_path.with_suffix(".pt.tmp")
            torch.save(f0_resampled, tmp_path)
            tmp_path.replace(f0_path)
            n_done += 1
        except Exception as e:
            tqdm.write(f"  failed {utt_id}: {e}")
            n_failed_extract += 1

    print(f"  Done: {n_done} success, "
          f"{n_failed_no_wav} no_wav, "
          f"{n_failed_no_hubert} no_hubert, "
          f"{n_failed_extract} extract_fail")
    return n_done, n_failed_no_wav + n_failed_no_hubert + n_failed_extract


def main():
    if not RMVPE.exists():
        print(f"RMVPE model not found: {RMVPE}")
        sys.exit(1)

    print("Loading RMVPE...")
    f0_ext = F0Extractor(method="rmvpe", rmvpe_model_path=str(RMVPE))

    total_done = 0
    total_failed = 0
    for lang_code, lang_name in LANG_NAME_MAP.items():
        done, failed = reextract_one_lang(lang_code, lang_name, f0_ext)
        total_done += done
        total_failed += failed

    print(f"\nGRAND TOTAL: {total_done} re-extracted, {total_failed} failed")
    if total_failed:
        print("WARNING: some samples failed. Check the log for details.")


if __name__ == "__main__":
    main()
