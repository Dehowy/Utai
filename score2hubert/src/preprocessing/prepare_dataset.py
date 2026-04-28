"""
Full dataset preparation pipeline for Score2HuBERT.

Handles multiple dataset formats:
1. Opencpop (Chinese singing, pre-training)
2. Kiritan (Japanese singing)
3. Ofuton-P (Japanese singing)
4. GTSinger (multi-lingual singing)

Pipeline for each dataset:
1. Parse annotations (lyrics, phonemes, note pitch, note duration)
2. Convert to unified format (phoneme IDs, MIDI, frame counts)
3. Extract audio features (HuBERT, F0, energy, breathiness)
4. Optionally apply RVC compensation
5. Generate train/val/test splits
6. Save metadata.json + per-sample .pt files
"""

import torch
import numpy as np
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re
from tqdm import tqdm
import logging

from ..data.phoneme import (
    PhonemeVocab, opencpop_to_phonemes, kiritan_to_phonemes,
    convert_phonemes_to_ipa, convert_gtsinger_phonemes_to_ipa,
)
import hashlib
import unicodedata

logger = logging.getLogger(__name__)

FRAME_RATE = 50  # fps — HuBERT-Base actual rate (16kHz / 320 internal hop stride)
# NOTE: HuBERT-Base conv feature extractor strides are 5,2,2,2,2,2,2 = total 320
# At 16kHz: 16000/320 = 50fps. F0 is extracted at 100fps and downsampled to match.


def prepare_opencpop(
    raw_dir: str,
    output_dir: str,
    vocab: PhonemeVocab,
    hubert_extractor=None,
    f0_extractor=None,
):
    """
    Prepare Opencpop dataset.

    Expected raw_dir structure:
        opencpop/
        ├── wavs/           # 3756 WAV files
        ├── segments/
        │   └── transcriptions.txt  # phoneme-level annotations
        └── textgrids/      # Praat TextGrid alignments (optional)

    Transcriptions format per line:
        uttid|phonemes|note_seq|note_dur_seq|is_slur
        2001000001|SP d ong1 f eng1 SP|rest C4 C4 D4 D4 rest|0.08 0.41 ...
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    # Create output directories
    for subdir in ["hubert", "f0", "phonemes", "pitch", "duration", "energy"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Parse transcriptions
    trans_file = raw_dir / "segments" / "transcriptions.txt"
    if not trans_file.exists():
        trans_file = raw_dir / "transcriptions.txt"

    metadata = []

    with open(trans_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing Opencpop"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("|")
        if len(parts) < 5:
            continue

        utt_id = parts[0]
        phoneme_str = parts[1]
        note_str = parts[2]
        dur_str = parts[3]

        # Find audio file
        wav_path = raw_dir / "wavs" / f"{utt_id}.wav"
        if not wav_path.exists():
            continue

        # Parse phonemes
        raw_phones = phoneme_str.strip().split()
        phoneme_ids = vocab.encode(raw_phones, language="zh")

        # Parse notes (e.g., "rest C4 C4 D4")
        notes = note_str.strip().split()
        midi_pitches = []
        for n in notes:
            if n.lower() in ("rest", "0"):
                midi_pitches.append(0)
            else:
                midi_pitches.append(_note_name_to_midi(n))

        # Parse durations (seconds → frames)
        durs_sec = [float(d) for d in dur_str.strip().split()]
        durs_frames = [max(1, round(d * FRAME_RATE)) for d in durs_sec]

        # Ensure all arrays are same length
        min_len = min(len(phoneme_ids), len(midi_pitches), len(durs_frames))
        phoneme_ids = phoneme_ids[:min_len]
        midi_pitches = midi_pitches[:min_len]
        durs_frames = durs_frames[:min_len]

        n_frames = sum(durs_frames)

        # Extract audio features
        if hubert_extractor:
            try:
                hubert = hubert_extractor.extract(wav_path)
                # Align frame count
                if hubert.size(0) > n_frames:
                    hubert = hubert[:n_frames]
                elif hubert.size(0) < n_frames:
                    # Adjust last phoneme duration
                    diff = n_frames - hubert.size(0)
                    durs_frames[-1] = max(1, durs_frames[-1] - diff)
                    n_frames = sum(durs_frames)
                    hubert = hubert[:n_frames]

                torch.save(hubert, output_dir / "hubert" / f"{utt_id}.pt")
            except Exception as e:
                logger.warning(f"HuBERT extraction failed for {utt_id}: {e}")
                continue

        if f0_extractor:
            try:
                f0 = f0_extractor.extract(wav_path)
                f0 = f0[:n_frames]
                torch.save(f0, output_dir / "f0" / f"{utt_id}.pt")
            except Exception as e:
                logger.warning(f"F0 extraction failed for {utt_id}: {e}")

        # Save phoneme-level features
        torch.save(torch.tensor(phoneme_ids, dtype=torch.long),
                    output_dir / "phonemes" / f"{utt_id}.pt")
        torch.save(torch.tensor(midi_pitches, dtype=torch.long),
                    output_dir / "pitch" / f"{utt_id}.pt")
        torch.save(torch.tensor(durs_frames, dtype=torch.long),
                    output_dir / "duration" / f"{utt_id}.pt")

        metadata.append({
            "id": utt_id,
            "n_phones": min_len,
            "n_frames": n_frames,
            "language": "zh",
            "lang_id": vocab.get_language_id("zh"),
            "singing_ratio": 1.0,
            "dataset": "opencpop",
        })

    return metadata


def prepare_kiritan(
    raw_dir: str,
    output_dir: str,
    vocab: PhonemeVocab,
    hubert_extractor=None,
    f0_extractor=None,
):
    """
    Prepare Kiritan (東北きりたん) singing database.

    Expected raw_dir structure:
        kiritan/
        ├── wav/            # WAV audio files
        ├── mono_label/     # monophone HTS labels
        ├── musicxml/       # MusicXML scores (optional)
        └── lab/            # full-context HTS labels (optional)

    HTS mono_label format per line:
        start_time end_time phoneme
        (times in 100ns units)
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    for subdir in ["hubert", "f0", "phonemes", "pitch", "duration", "energy"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    metadata = []

    # Find label files
    label_dir = raw_dir / "mono_label"
    if not label_dir.exists():
        label_dir = raw_dir / "lab"
    if not label_dir.exists():
        logger.error(f"No label directory found in {raw_dir}")
        return []

    label_files = sorted(label_dir.glob("*.lab"))
    wav_dir = raw_dir / "wav"

    for lab_path in tqdm(label_files, desc="Processing Kiritan"):
        utt_id = lab_path.stem
        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            continue

        # Parse HTS labels
        phonemes = []
        durations_sec = []

        with open(lab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                start = int(parts[0]) / 1e7  # 100ns → seconds
                end = int(parts[1]) / 1e7
                phone = parts[2]

                phonemes.append(phone)
                durations_sec.append(end - start)

        if not phonemes:
            continue

        # Convert OpenJTalk phonemes → IPA segments (with duration split)
        ipa_phones, ipa_durs_sec = convert_phonemes_to_ipa(
            phonemes, durations_sec, language="ja"
        )
        phoneme_ids = vocab.encode(ipa_phones, language="ja")

        # MIDI pitches — placeholder, will estimate from F0 below
        midi_pitches = [60] * len(phoneme_ids)

        # Durations to frames
        durs_frames = [max(1, round(d * FRAME_RATE)) for d in ipa_durs_sec]

        # Extract HuBERT first to get actual frame count
        n_frames = sum(durs_frames)
        actual_n_frames = n_frames  # fallback
        if hubert_extractor:
            try:
                hubert = hubert_extractor.extract(wav_path)
                actual_n_frames = hubert.shape[0]
                torch.save(hubert, output_dir / "hubert" / f"{utt_id}.pt")
            except Exception as e:
                logger.warning(f"HuBERT extraction failed for {utt_id}: {e}")
                continue

        # Extract F0 then resample to match HuBERT frame count
        f0 = None
        if f0_extractor:
            try:
                f0_raw = f0_extractor.extract(wav_path)
                f0 = torch.nn.functional.interpolate(
                    f0_raw.unsqueeze(0).unsqueeze(0),
                    size=actual_n_frames, mode="linear",
                    align_corners=False,
                ).squeeze()
                torch.save(f0, output_dir / "f0" / f"{utt_id}.pt")
            except Exception as e:
                logger.warning(f"F0 extraction failed for {utt_id}: {e}")
                f0 = None

        # Recompute durations to sum to actual_n_frames
        n_frames = actual_n_frames
        total_dur_sec = sum(ipa_durs_sec)
        if total_dur_sec > 0:
            durs_frames = [max(1, round(d / total_dur_sec * n_frames))
                           for d in ipa_durs_sec]
            diff = n_frames - sum(durs_frames)
            durs_frames[-1] = max(1, durs_frames[-1] + diff)

        min_len = min(len(phoneme_ids), len(midi_pitches), len(durs_frames))
        phoneme_ids = phoneme_ids[:min_len]
        midi_pitches = midi_pitches[:min_len]
        durs_frames = durs_frames[:min_len]
        n_frames = sum(durs_frames)

        # Estimate MIDI pitch per IPA segment from resampled F0
        if f0 is not None:
            midi_pitches = _estimate_midi_from_f0(f0, durs_frames)

        torch.save(torch.tensor(phoneme_ids, dtype=torch.long),
                    output_dir / "phonemes" / f"{utt_id}.pt")
        torch.save(torch.tensor(midi_pitches, dtype=torch.long),
                    output_dir / "pitch" / f"{utt_id}.pt")
        torch.save(torch.tensor(durs_frames, dtype=torch.long),
                    output_dir / "duration" / f"{utt_id}.pt")

        metadata.append({
            "id": utt_id,
            "n_phones": min_len,
            "n_frames": n_frames,
            "language": "ja",
            "lang_id": vocab.get_language_id("ja"),
            "singing_ratio": 1.0,
            "dataset": "kiritan",
        })

    return metadata


def prepare_gtsinger(
    raw_dir: str,
    output_dir: str,
    vocab: PhonemeVocab,
    language: str = "zh",
    hubert_extractor=None,
    f0_extractor=None,
    include_speech: bool = True,
    singer_filter: Optional[str] = None,
    max_entries: Optional[int] = None,
):
    """
    Prepare GTSinger dataset (Phase 2 multilingual fine-tune).

    GTSinger ships with rich per-language `processed/processed/<lang>/metadata.json`
    files that contain phoneme lists, durations, MIDI pitches, technique flags,
    and pointers to both singing wav (`wav_fn`) and paired speech wav (`speech_fn`).
    We bypass TextGrid parsing entirely and use these directly for singing.

    For SPEECH samples, the per-language metadata only has `speech_fn` paths
    without alignments. We parse the per-utterance `Paired_Speech_Group/*.json`
    files (one per speech wav, word-level entries with phoneme start/end times)
    to recover speech alignments.

    Three GTSinger gotchas this function handles transparently:
      1. **Mojibake from Mac→Windows zip extraction**: filenames are stored as
         cp437-encoded utf-8 bytes. `_build_gtsinger_path_index` walks once and
         decodes via cp437→utf-8 + NFC, building a {fixed_wav_fn: actual_path}
         dict for resolution.
      2. **Per-language phoneme schemes (NOT IPA)**: Chinese=pypinyin,
         English=ARPABET, Japanese=MFA-IPA. All routed through
         `convert_gtsinger_phonemes_to_ipa` which handles each.
      3. **Double-nested layout**: zip extracted as e.g. `GTSinger/Chinese/Chinese/...`
         instead of `GTSinger/Chinese/...`. Handled by walking under `raw_dir/<lang_name>/`.

    Args:
        raw_dir: GTSinger root containing `Chinese/`, `Japanese/`, `English/`,
                 and `processed/processed/<lang>/metadata.json`.
        output_dir: Where to write `data/processed/gtsinger_<lang>/...`.
        vocab: PhonemeVocab (must include θ, ð for English support).
        language: 'zh' / 'ja' / 'en' (short code) or 'Chinese' / 'Japanese' / 'English'.
        hubert_extractor: HuBERTExtractor instance.
        f0_extractor: F0Extractor instance.
        include_speech: If True, also process Paired_Speech_Group entries with
                        singing_ratio=0.0 (for rap/spoken-segment training).
        singer_filter: If set (e.g. "JA-Tenor-1"), process only that singer's
                       data. Used for smoke tests.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    # Normalize language to both short code (for our vocab) and full name (for paths)
    LANG_NAME_MAP = {"zh": "Chinese", "ja": "Japanese", "en": "English", "ko": "Korean"}
    LANG_CODE_MAP = {v: k for k, v in LANG_NAME_MAP.items()}
    if language in LANG_NAME_MAP:
        lang_code, lang_name = language, LANG_NAME_MAP[language]
    elif language in LANG_CODE_MAP:
        lang_code, lang_name = LANG_CODE_MAP[language], language
    else:
        raise ValueError(f"Unsupported GTSinger language: {language}")

    for subdir in ["hubert", "f0", "phonemes", "pitch", "duration", "tech"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Build mojibake-aware path index for SINGING wavs (resolved via metadata.json)
    path_index = _build_gtsinger_path_index(raw_dir, lang_name)
    logger.info(f"GTSinger {lang_name}: indexed {len(path_index)} wav files (mojibake-aware)")
    if not path_index:
        logger.error(f"GTSinger {lang_name}: no wav files found under {raw_dir / lang_name}")
        return []

    metadata = []

    # ── Process SINGING from per-language metadata.json ──
    sing_meta_path = raw_dir / "processed" / "processed" / lang_name / "metadata.json"
    if sing_meta_path.exists():
        with open(sing_meta_path, encoding="utf-8") as f:
            sing_entries = json.load(f)
        if singer_filter:
            sing_entries = [e for e in sing_entries if e.get("singer") == singer_filter]
            logger.info(f"Singer filter '{singer_filter}': {len(sing_entries)} singing entries")
        if max_entries is not None:
            sing_entries = sing_entries[:max_entries]
            logger.info(f"max_entries cap: trimmed to {len(sing_entries)} singing entries")
        sing_meta = _process_gtsinger_singing(
            sing_entries, path_index, output_dir, vocab, lang_code,
            hubert_extractor, f0_extractor,
        )
        metadata.extend(sing_meta)
        logger.info(f"GTSinger {lang_name}: kept {len(sing_meta)} / {len(sing_entries)} singing entries")
    else:
        logger.error(f"GTSinger {lang_name}: missing {sing_meta_path}")

    # ── Process SPEECH from per-utt Paired_Speech_Group/*.json ──
    if include_speech:
        speech_jsons = _find_gtsinger_speech_jsons(raw_dir, lang_name, singer_filter)
        if max_entries is not None:
            speech_jsons = speech_jsons[:max_entries]
        speech_meta = _process_gtsinger_speech(
            speech_jsons, raw_dir, output_dir, vocab, lang_code,
            hubert_extractor, f0_extractor,
        )
        metadata.extend(speech_meta)
        logger.info(f"GTSinger {lang_name}: kept {len(speech_meta)} / {len(speech_jsons)} speech entries")

    return metadata


# ──────────────────────────────────────────────────────────────────────────────
# GTSinger helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fix_mojibake(name: str) -> str:
    """Fix cp437→utf-8 mojibake (Mac zip extracted on Windows) and normalize to NFC.

    Pure ASCII names pass through unchanged. For non-ASCII names, attempt to
    re-encode as cp437 (the legacy DOS codepage Windows misinterpreted the bytes
    as) and decode as utf-8 (the actual byte intent). Then normalize to NFC to
    match metadata.json (which is also NFC).

    Verified 100% recovery rate on GTSinger Chinese/Japanese/English subsets.
    """
    if all(ord(c) < 128 for c in name):
        return name
    try:
        return unicodedata.normalize("NFC", name.encode("cp437").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def _build_gtsinger_path_index(raw_dir: Path, lang_name: str) -> Dict[str, Path]:
    """Walk raw_dir/<lang_name>/ and return {metadata_wav_fn: actual_disk_path}.

    The metadata.json `wav_fn` field is e.g. 'Chinese/ZH-Alto-1/Breathy/不再见/Breathy_Group/0000.wav'.
    On disk, due to GTSinger's double-nested zip layout AND mojibake, the actual
    path is 'raw_dir/Chinese/Chinese/ZH-Alto-1/Breathy/<mojibake song>/.../0000.wav'.

    Resolution: walk under raw_dir/<lang_name>/, fix each path component via
    `_fix_mojibake`, then drop the duplicate outer <lang_name> from the rel path
    to get the metadata-style key.
    """
    lang_root = raw_dir / lang_name
    if not lang_root.exists():
        return {}

    index = {}
    for p in lang_root.rglob("*.wav"):
        if "__MACOSX" in p.parts:
            continue
        rel = p.relative_to(lang_root)
        # rel.parts starts with the inner <lang_name>: ('Chinese', 'ZH-Alto-1', ...)
        # which IS the metadata.json wav_fn format after mojibake fixing.
        fixed_parts = [_fix_mojibake(part) for part in rel.parts]
        metadata_form = "/".join(fixed_parts)
        index[metadata_form] = p
    return index


def _find_gtsinger_speech_jsons(
    raw_dir: Path,
    lang_name: str,
    singer_filter: Optional[str] = None,
) -> List[Path]:
    """Find all Paired_Speech_Group/*.json files for a language.

    Path structure under raw_dir/<lang_name>/:
        <inner_lang>/<singer>/<technique>/<song>/Paired_Speech_Group/{NNNN}.json
    """
    lang_root = raw_dir / lang_name
    if not lang_root.exists():
        return []

    jsons = []
    for p in lang_root.rglob("Paired_Speech_Group/*.json"):
        if "__MACOSX" in p.parts:
            continue
        if singer_filter:
            try:
                rel = p.relative_to(lang_root)
                # rel = <inner_lang>/<singer>/<technique>/<song>/Paired_Speech_Group/file.json
                if len(rel.parts) >= 2 and rel.parts[1] != singer_filter:
                    continue
            except ValueError:
                pass
        jsons.append(p)
    return jsons


def _gtsinger_utt_id(item_name: str, suffix: str = "") -> str:
    """Generate a deterministic ASCII-safe utterance ID from a GTSinger item_name.

    GTSinger item_name format: '<Lang>#<Singer>#<Technique>#<Song>#<Group>#<Idx>'
    Example: 'Chinese#ZH-Alto-1#Breathy#不再见#Breathy_Group#0000'

    Output: 'gtsinger_zh_ZH-Alto-1_breathy_<md5_12>_0000[_speech]'

    md5 of the full item_name ensures uniqueness even if the song name has
    non-ASCII chars; the ASCII-safe parts give human readability.
    """
    parts = item_name.split("#")
    if len(parts) >= 6:
        lang, singer, technique = parts[0], parts[1], parts[2]
        idx = parts[5]
        lang_code = {"Chinese": "zh", "Japanese": "ja", "English": "en", "Korean": "ko"}.get(
            lang, lang.lower()[:2]
        )
        h = hashlib.md5(item_name.encode("utf-8")).hexdigest()[:12]
        # Sanitize singer/technique to ASCII-safe form
        singer_safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in singer)
        tech_safe = technique.lower().replace("_group", "")
        utt_id = f"gtsinger_{lang_code}_{singer_safe}_{tech_safe}_{h}_{idx}"
    else:
        h = hashlib.md5(item_name.encode("utf-8")).hexdigest()[:16]
        utt_id = f"gtsinger_unknown_{h}"
    if suffix:
        utt_id = f"{utt_id}_{suffix}"
    return utt_id


def _expand_pitches_via_gtsinger_g2p(
    raw_phs: List[str],
    raw_pitches: List[int],
    language: str,
) -> List[int]:
    """Re-run G2P per-phoneme to determine the decomposition factor, then
    duplicate the parent pitch across each output IPA segment.

    Necessary because Pinyin finals (iang→j a ŋ) and English diphthongs
    (AY→a ɪ) decompose into multiple segments — each must inherit the parent
    note's MIDI pitch.
    """
    expanded = []
    for ph, pitch in zip(raw_phs, raw_pitches):
        sub_ipa, _ = convert_gtsinger_phonemes_to_ipa([ph], [1.0], language=language)
        expanded.extend([int(pitch)] * len(sub_ipa))
    return expanded


# Phase 3: fixed technique ordering. Must match encoder's tech_proj input dim.
GTSINGER_TECH_KEYS = [
    "mix_tech",        # index 0: mixed voice
    "falsetto_tech",   # index 1: falsetto
    "breathy_tech",    # index 2: breathy
    "pharyngeal_tech", # index 3: pharyngeal
    "vibrato_tech",    # index 4: vibrato
    "glissando_tech",  # index 5: glissando
]
N_TECHNIQUES = len(GTSINGER_TECH_KEYS)


def _expand_tech_via_gtsinger_g2p(
    raw_phs: List[str],
    tech_matrix: List[List[int]],  # [N_raw, 6] per-raw-phoneme tech flags
    language: str,
) -> List[List[int]]:
    """Expand per-phoneme technique flags in parallel with IPA decomposition.

    Each decomposed IPA segment inherits the parent raw phoneme's tech flags,
    mirroring how _expand_pitches_via_gtsinger_g2p works for MIDI pitches.
    """
    expanded: List[List[int]] = []
    for ph, flags in zip(raw_phs, tech_matrix):
        sub_ipa, _ = convert_gtsinger_phonemes_to_ipa([ph], [1.0], language=language)
        for _ in sub_ipa:
            expanded.append([int(f) for f in flags])
    return expanded


def _process_gtsinger_singing(
    entries: list,
    path_index: Dict[str, Path],
    output_dir: Path,
    vocab: PhonemeVocab,
    language: str,
    hubert_extractor,
    f0_extractor,
) -> list:
    """Process GTSinger singing entries from per-language metadata.json.

    Each entry has flat ph/ph_durs/ep_pitches lists already aligned to the
    singing wav. We:
      1. Resolve wav path via mojibake-aware index
      2. Convert phonemes (raw → IPA → vocab IDs)
      3. Expand MIDI pitches in parallel with phoneme decomposition
      4. Convert seconds → frames at FRAME_RATE
      5. Extract HuBERT and F0 from wav
      6. Save .pt tensors keyed by deterministic utt_id
    """
    metadata = []
    lang_id = vocab.get_language_id(language)

    n_skipped_no_wav = 0
    n_skipped_too_short = 0
    n_skipped_too_long = 0
    n_skipped_hubert_fail = 0

    for entry in tqdm(entries, desc=f"GTSinger {language} singing"):
        wav_fn = unicodedata.normalize("NFC", entry["wav_fn"])
        wav_path = path_index.get(wav_fn)
        if wav_path is None or not wav_path.exists():
            n_skipped_no_wav += 1
            continue

        utt_id = _gtsinger_utt_id(entry["item_name"])

        # Phoneme conversion
        raw_phs = entry["ph"]
        raw_durs = entry["ph_durs"]
        ipa_phs, ipa_durs = convert_gtsinger_phonemes_to_ipa(
            raw_phs, raw_durs, language=language
        )
        if not ipa_phs:
            continue
        phoneme_ids = vocab.encode(ipa_phs)

        # Duration conversion (sec → frames)
        durs_frames = [max(1, round(d * FRAME_RATE)) for d in ipa_durs]

        # MIDI pitch expansion (parallel to phoneme decomposition)
        midi_pitches = _expand_pitches_via_gtsinger_g2p(
            raw_phs, entry["ep_pitches"], language=language
        )

        # Technique flag expansion (Phase 3): stack 6 per-phoneme binary arrays
        # into a [N_raw, 6] matrix, then expand parallel to phoneme decomposition.
        raw_tech_matrix = []
        for i in range(len(raw_phs)):
            row = []
            for key in GTSINGER_TECH_KEYS:
                arr = entry.get(key, [])
                row.append(int(arr[i]) if i < len(arr) else 0)
            raw_tech_matrix.append(row)
        tech_expanded = _expand_tech_via_gtsinger_g2p(
            raw_phs, raw_tech_matrix, language=language
        )  # list of list-of-6-ints

        # Length sanity (decomposition should preserve count exactly)
        min_len = min(len(phoneme_ids), len(midi_pitches), len(durs_frames), len(tech_expanded))
        phoneme_ids = phoneme_ids[:min_len]
        midi_pitches = midi_pitches[:min_len]
        durs_frames = durs_frames[:min_len]
        tech_expanded = tech_expanded[:min_len]
        n_frames = sum(durs_frames)

        if n_frames < 30:
            n_skipped_too_short += 1
            continue
        if n_frames > 1500:
            n_skipped_too_long += 1
            continue

        # Feature extraction.
        # CRITICAL: F0 from RMVPE comes out at 100fps but HuBERT is at 50fps.
        # Must INTERPOLATE F0 to HuBERT's frame count, NOT truncate. Otherwise
        # we keep only the first half of the F0 contour, causing gt_ceiling to
        # play pitch at 2x speed (off-key). See _process_gtsinger_singing fix
        # comment for the bug history.
        if hubert_extractor:
            try:
                hubert = hubert_extractor.extract(wav_path)[:n_frames]
                if len(hubert) < 10:
                    n_skipped_too_short += 1
                    continue
                torch.save(hubert, output_dir / "hubert" / f"{utt_id}.pt")
            except Exception as e:
                logger.warning(f"HuBERT failed for {utt_id}: {e}")
                n_skipped_hubert_fail += 1
                continue

        if f0_extractor:
            try:
                f0_raw = f0_extractor.extract(wav_path)  # ~100fps
                # Resample F0 to match HuBERT frame count (n_frames at 50fps)
                f0 = torch.nn.functional.interpolate(
                    f0_raw.view(1, 1, -1).float(),
                    size=n_frames, mode="linear", align_corners=False,
                ).view(-1)
                torch.save(f0, output_dir / "f0" / f"{utt_id}.pt")
            except Exception as e:
                logger.warning(f"F0 failed for {utt_id}: {e}")

        torch.save(torch.tensor(phoneme_ids, dtype=torch.long),
                   output_dir / "phonemes" / f"{utt_id}.pt")
        torch.save(torch.tensor(midi_pitches, dtype=torch.long),
                   output_dir / "pitch" / f"{utt_id}.pt")
        torch.save(torch.tensor(durs_frames, dtype=torch.long),
                   output_dir / "duration" / f"{utt_id}.pt")
        # Phase 3: per-phoneme technique flags [T_phone, 6] as float (0.0 / 1.0)
        torch.save(torch.tensor(tech_expanded, dtype=torch.float32),
                   output_dir / "tech" / f"{utt_id}.pt")

        metadata.append({
            "id": utt_id,
            "n_phones": len(phoneme_ids),
            "n_frames": n_frames,
            "language": language,
            "lang_id": lang_id,
            "singing_ratio": 1.0,
            "dataset": "gtsinger",
            "singer": entry.get("singer", ""),
            "orig_item_name": entry.get("item_name", ""),
        })

    if n_skipped_no_wav or n_skipped_too_short or n_skipped_too_long or n_skipped_hubert_fail:
        logger.info(
            f"GTSinger singing skip stats: no_wav={n_skipped_no_wav}, "
            f"too_short={n_skipped_too_short}, too_long={n_skipped_too_long}, "
            f"hubert_fail={n_skipped_hubert_fail}"
        )
    return metadata


def _process_gtsinger_speech(
    json_paths: List[Path],
    raw_dir: Path,
    output_dir: Path,
    vocab: PhonemeVocab,
    language: str,
    hubert_extractor,
    f0_extractor,
) -> list:
    """Process GTSinger paired speech from per-utterance Paired_Speech_Group/*.json.

    Each json is a list of word entries with phoneme-level start/end times. We
    flatten across words and produce one sample per speech wav with:
      - singing_ratio = 0.0 (rap/speech mode → uses rap_head in F0 predictor)
      - midi_pitches  = 0   (no melodic notes for spoken text)
      - F0 extracted from speech wav (natural speaking F0, not melodic)

    Plan C's rap_head was never trained (ACE-Opencpop is all singing). Phase 2
    is the first time it sees gradient, so speech audio quality will lag
    singing in early training steps.
    """
    metadata = []
    lang_id = vocab.get_language_id(language)

    n_skipped = 0

    for json_path in tqdm(json_paths, desc=f"GTSinger {language} speech"):
        # Speech wav is at the same path with .wav extension (mojibake intact, OK)
        wav_path = json_path.with_suffix(".wav")
        if not wav_path.exists():
            n_skipped += 1
            continue

        try:
            with open(json_path, encoding="utf-8") as f:
                word_entries = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {json_path}: {e}")
            n_skipped += 1
            continue

        if not isinstance(word_entries, list) or not word_entries:
            n_skipped += 1
            continue

        # Flatten phonemes/durations across word entries
        raw_phs: List[str] = []
        raw_durs: List[float] = []
        for w in word_entries:
            ph_list = w.get("ph", [])
            ph_starts = w.get("ph_start", [])
            ph_ends = w.get("ph_end", [])
            for ph, s, e in zip(ph_list, ph_starts, ph_ends):
                raw_phs.append(ph)
                raw_durs.append(max(0.0, float(e) - float(s)))

        if not raw_phs:
            n_skipped += 1
            continue

        ipa_phs, ipa_durs = convert_gtsinger_phonemes_to_ipa(
            raw_phs, raw_durs, language=language
        )
        if not ipa_phs:
            continue
        phoneme_ids = vocab.encode(ipa_phs)
        durs_frames = [max(1, round(d * FRAME_RATE)) for d in ipa_durs]
        midi_pitches = [0] * len(phoneme_ids)  # speech has no melody

        n_frames = sum(durs_frames)
        if n_frames < 30 or n_frames > 1500:
            n_skipped += 1
            continue

        # Reconstruct an item_name from the path for utt_id generation
        try:
            rel = json_path.relative_to(raw_dir)
            fixed = [_fix_mojibake(p) for p in rel.parts]
            # Drop duplicate outer lang dir if present
            if len(fixed) >= 2 and fixed[0] == fixed[1]:
                fixed = fixed[1:]
            item_name = "#".join(fixed).replace(".json", "")
        except (ValueError, IndexError):
            item_name = str(json_path)
        utt_id = _gtsinger_utt_id(item_name, suffix="speech")

        # Same F0 alignment fix as singing path: RMVPE outputs 100fps but HuBERT
        # is 50fps. Must INTERPOLATE F0 to HuBERT length, NOT truncate.
        if hubert_extractor:
            try:
                hubert = hubert_extractor.extract(wav_path)[:n_frames]
                if len(hubert) < 10:
                    n_skipped += 1
                    continue
                torch.save(hubert, output_dir / "hubert" / f"{utt_id}.pt")
            except Exception as e:
                logger.warning(f"HuBERT failed for speech {utt_id}: {e}")
                n_skipped += 1
                continue

        if f0_extractor:
            try:
                f0_raw = f0_extractor.extract(wav_path)  # ~100fps
                f0 = torch.nn.functional.interpolate(
                    f0_raw.view(1, 1, -1).float(),
                    size=n_frames, mode="linear", align_corners=False,
                ).view(-1)
                torch.save(f0, output_dir / "f0" / f"{utt_id}.pt")
            except Exception as e:
                logger.warning(f"F0 failed for speech {utt_id}: {e}")

        torch.save(torch.tensor(phoneme_ids, dtype=torch.long),
                   output_dir / "phonemes" / f"{utt_id}.pt")
        torch.save(torch.tensor(midi_pitches, dtype=torch.long),
                   output_dir / "pitch" / f"{utt_id}.pt")
        torch.save(torch.tensor(durs_frames, dtype=torch.long),
                   output_dir / "duration" / f"{utt_id}.pt")
        # Phase 3: speech has no technique annotation → all-zero [T_phone, 6]
        torch.save(torch.zeros(len(phoneme_ids), N_TECHNIQUES, dtype=torch.float32),
                   output_dir / "tech" / f"{utt_id}.pt")

        metadata.append({
            "id": utt_id,
            "n_phones": len(phoneme_ids),
            "n_frames": n_frames,
            "language": language,
            "lang_id": lang_id,
            "singing_ratio": 0.0,
            "dataset": "gtsinger",
            "orig_path": str(rel) if "rel" in dir() else str(json_path),
        })

    if n_skipped:
        logger.info(f"GTSinger speech skipped {n_skipped} entries")
    return metadata


def prepare_ofuton(
    raw_dir: str,
    output_dir: str,
    vocab: PhonemeVocab,
    hubert_extractor=None,
    f0_extractor=None,
):
    """
    Prepare Ofuton-P (おふとんP) singing database.

    Expected raw_dir structure:
        ofuton/
        ├── wav/            # WAV audio files (96kHz/24bit)
        ├── lab/            # HTS monophone labels (100ns precision)
        ├── mid/            # MIDI files (optional)
        └── musicxml/       # MusicXML scores (optional)

    HTS label format is identical to Kiritan:
        start_time end_time phoneme
        (times in 100ns units)
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    for subdir in ["hubert", "f0", "phonemes", "pitch", "duration", "energy"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    metadata = []

    # Find label files
    label_dir = raw_dir / "lab"
    if not label_dir.exists():
        label_dir = raw_dir / "mono_label"
    if not label_dir.exists():
        logger.error(f"No label directory found in {raw_dir}")
        return []

    label_files = sorted(label_dir.glob("*.lab"))
    wav_dir = raw_dir / "wav"

    for lab_path in tqdm(label_files, desc="Processing Ofuton-P"):
        utt_id = lab_path.stem
        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            continue

        # Parse HTS labels (same format as Kiritan)
        phonemes = []
        durations_sec = []

        with open(lab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                start = int(parts[0]) / 1e7  # 100ns → seconds
                end = int(parts[1]) / 1e7
                phone = parts[2]

                phonemes.append(phone)
                durations_sec.append(end - start)

        if not phonemes:
            continue

        # Convert OpenJTalk phonemes → IPA segments (with duration split)
        ipa_phones, ipa_durs_sec = convert_phonemes_to_ipa(
            phonemes, durations_sec, language="ja"
        )
        phoneme_ids = vocab.encode(ipa_phones, language="ja")
        midi_pitches = [60] * len(phoneme_ids)
        durs_frames = [max(1, round(d * FRAME_RATE)) for d in ipa_durs_sec]

        # Extract HuBERT first to get actual frame count
        n_frames = sum(durs_frames)
        actual_n_frames = n_frames
        if hubert_extractor:
            try:
                hubert = hubert_extractor.extract(wav_path)
                actual_n_frames = hubert.shape[0]
                torch.save(hubert, output_dir / "hubert" / f"{utt_id}.pt")
            except Exception as e:
                logger.warning(f"HuBERT extraction failed for {utt_id}: {e}")
                continue

        f0 = None
        if f0_extractor:
            try:
                f0_raw = f0_extractor.extract(wav_path)
                f0 = torch.nn.functional.interpolate(
                    f0_raw.unsqueeze(0).unsqueeze(0),
                    size=actual_n_frames, mode="linear",
                    align_corners=False,
                ).squeeze()
                torch.save(f0, output_dir / "f0" / f"{utt_id}.pt")
            except Exception as e:
                logger.warning(f"F0 extraction failed for {utt_id}: {e}")
                f0 = None

        # Recompute durations to sum to actual_n_frames
        n_frames = actual_n_frames
        total_dur_sec = sum(ipa_durs_sec)
        if total_dur_sec > 0:
            durs_frames = [max(1, round(d / total_dur_sec * n_frames))
                           for d in ipa_durs_sec]
            diff = n_frames - sum(durs_frames)
            durs_frames[-1] = max(1, durs_frames[-1] + diff)

        min_len = min(len(phoneme_ids), len(midi_pitches), len(durs_frames))
        phoneme_ids = phoneme_ids[:min_len]
        midi_pitches = midi_pitches[:min_len]
        durs_frames = durs_frames[:min_len]
        n_frames = sum(durs_frames)

        if f0 is not None:
            midi_pitches = _estimate_midi_from_f0(f0, durs_frames)

        torch.save(torch.tensor(phoneme_ids, dtype=torch.long),
                    output_dir / "phonemes" / f"{utt_id}.pt")
        torch.save(torch.tensor(midi_pitches, dtype=torch.long),
                    output_dir / "pitch" / f"{utt_id}.pt")
        torch.save(torch.tensor(durs_frames, dtype=torch.long),
                    output_dir / "duration" / f"{utt_id}.pt")

        metadata.append({
            "id": utt_id,
            "n_phones": min_len,
            "n_frames": n_frames,
            "language": "ja",
            "lang_id": vocab.get_language_id("ja"),
            "singing_ratio": 1.0,
            "dataset": "ofuton",
        })

    return metadata


def prepare_ace_opencpop(
    raw_dir: str,
    output_dir: str,
    vocab: PhonemeVocab,
    hubert_extractor=None,
    f0_extractor=None,
    max_samples: Optional[int] = None,
):
    """
    Prepare ACE-Opencpop dataset (~130h, 30 singers, HuggingFace format).

    Uses Arrow format access to bypass torchcodec audio decoder issues on Windows.
    Audio is decoded directly from raw bytes via soundfile.

    ACE-Opencpop fields (Arrow schema):
    - audio: struct{bytes, path}
    - phn: list<string>             # phoneme sequence
    - phn_start_time: list<double>  # phoneme start times (seconds)
    - phn_end_time: list<double>    # phoneme end times (seconds)
    - note_midi: list<double>       # MIDI values per note
    - note_phns: list<string>       # phonemes per note (for MIDI expansion)
    - singer: int64
    - segment_id: string
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    for subdir in ["hubert", "f0", "phonemes", "pitch", "duration", "energy"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    metadata = []

    # Load via HuggingFace datasets
    try:
        from datasets import load_dataset
        logger.info("Loading ACE-Opencpop from HuggingFace cache...")

        ds_path = raw_dir / "ace_opencpop"
        ds = load_dataset("espnet/ace-opencpop-segments",
                          cache_dir=str(ds_path) if ds_path.exists() else str(raw_dir))

        train_ds = ds.get("train", ds)
        logger.info(f"Loaded {len(train_ds)} samples")

        if max_samples:
            train_ds = train_ds.select(range(min(max_samples, len(train_ds))))

        # Use Arrow format to bypass torchcodec audio decoder
        train_ds = train_ds.with_format("arrow")

    except Exception as e:
        logger.error(f"Failed to load ACE-Opencpop: {e}")
        logger.info("Run: python scripts/download_datasets.py --dataset ace_opencpop")
        return []

    import io
    import soundfile as sf
    import tempfile

    n_total = len(train_ds)
    batch_size = 100  # process in batches for progress tracking

    for batch_start in tqdm(range(0, n_total, batch_size), desc="Processing ACE-Opencpop",
                            total=(n_total + batch_size - 1) // batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch = train_ds[batch_start:batch_end]

        for row_idx in range(batch.num_rows):
            idx = batch_start + row_idx

            try:
                # Extract fields from Arrow row
                seg_id = batch.column("segment_id")[row_idx].as_py()
                utt_id = f"ace_{seg_id}"

                # Skip if already processed
                if (output_dir / "phonemes" / f"{utt_id}.pt").exists():
                    continue

                # Parse phonemes (list<string> in Arrow) — these are pinyin
                raw_phones = batch.column("phn")[row_idx].as_py()
                if not raw_phones:
                    continue

                # Parse MIDI notes (list<double>)
                note_midis_raw = batch.column("note_midi")[row_idx].as_py()
                if not note_midis_raw:
                    continue
                note_midis = [int(m) if m and m != 0 else 0 for m in note_midis_raw]

                # Parse phoneme timings (list<double>)
                phn_starts = batch.column("phn_start_time")[row_idx].as_py()
                phn_ends = batch.column("phn_end_time")[row_idx].as_py()

                if not phn_starts or len(phn_starts) != len(raw_phones):
                    continue

                durs_sec_raw = [end - start for start, end in zip(phn_starts, phn_ends)]

                # Expand note MIDI to phoneme level (still in pinyin space)
                note_phns_list = batch.column("note_phns")[row_idx].as_py()
                if note_phns_list:
                    pinyin_midi = _expand_note_midi_to_phoneme_list(
                        note_midis, note_phns_list, len(raw_phones))
                else:
                    pinyin_midi = note_midis[:len(raw_phones)]
                    while len(pinyin_midi) < len(raw_phones):
                        pinyin_midi.append(pinyin_midi[-1] if pinyin_midi else 60)

                # ── Convert pinyin → IPA (decomposition + duration split) ──
                ipa_phones, ipa_durs_sec = convert_phonemes_to_ipa(
                    raw_phones, durs_sec_raw, language="zh"
                )

                # Expand MIDI to match the new IPA segment count
                # Each pinyin phoneme's MIDI propagates to all its IPA segments
                ipa_midi: List[int] = []
                for i, ph in enumerate(raw_phones):
                    n_segs = len(_get_ipa_segments_for_pinyin(ph))
                    midi_val = pinyin_midi[i] if i < len(pinyin_midi) else 0
                    ipa_midi.extend([midi_val] * n_segs)

                # Truncate to actual IPA length (in case of mismatch)
                min_len = min(len(ipa_phones), len(ipa_midi), len(ipa_durs_sec))
                ipa_phones = ipa_phones[:min_len]
                ipa_midi = ipa_midi[:min_len]
                ipa_durs_sec = ipa_durs_sec[:min_len]

                phoneme_ids = vocab.encode(ipa_phones, language="zh")
                midi_pitches = ipa_midi
                durs_frames = [max(1, round(d * FRAME_RATE)) for d in ipa_durs_sec]
                durs_sec = ipa_durs_sec

                min_len = min(len(phoneme_ids), len(midi_pitches), len(durs_frames))
                phoneme_ids = phoneme_ids[:min_len]
                midi_pitches = midi_pitches[:min_len]
                durs_frames = durs_frames[:min_len]
                n_frames = sum(durs_frames)

                # Decode audio from Arrow bytes
                audio_struct = batch.column("audio")[row_idx].as_py()
                audio_bytes = audio_struct.get("bytes")
                if audio_bytes:
                    audio_array, audio_sr = sf.read(io.BytesIO(audio_bytes))
                    if audio_array.ndim > 1:
                        audio_array = audio_array.mean(axis=1)

                    # Save to temp file for feature extraction
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name
                        sf.write(tmp_path, audio_array, audio_sr)

                    # Extract HuBERT first to get actual frame count
                    actual_n_frames = n_frames  # fallback
                    if hubert_extractor:
                        try:
                            hubert = hubert_extractor.extract(tmp_path)
                            actual_n_frames = hubert.shape[0]
                            torch.save(hubert, output_dir / "hubert" / f"{utt_id}.pt")
                        except Exception as e:
                            logger.warning(f"HuBERT failed for {utt_id}: {e}")
                            Path(tmp_path).unlink(missing_ok=True)
                            continue

                    # Extract F0 then resample to match HuBERT frame count
                    if f0_extractor:
                        try:
                            f0_raw = f0_extractor.extract(tmp_path)
                            # Resample F0 to HuBERT frame count
                            f0 = torch.nn.functional.interpolate(
                                f0_raw.unsqueeze(0).unsqueeze(0),
                                size=actual_n_frames, mode="linear",
                                align_corners=False,
                            ).squeeze()
                            torch.save(f0, output_dir / "f0" / f"{utt_id}.pt")
                        except Exception as e:
                            logger.warning(f"F0 failed for {utt_id}: {e}")

                    Path(tmp_path).unlink(missing_ok=True)

                    # Recompute durations to sum to actual_n_frames
                    n_frames = actual_n_frames
                    total_dur_sec = sum(durs_sec)
                    if total_dur_sec > 0:
                        durs_frames = [max(1, round(d / total_dur_sec * n_frames))
                                       for d in durs_sec]
                        # Adjust last to ensure exact sum
                        diff = n_frames - sum(durs_frames)
                        durs_frames[-1] = max(1, durs_frames[-1] + diff)
                    min_len = min(len(phoneme_ids), len(midi_pitches), len(durs_frames))
                    phoneme_ids = phoneme_ids[:min_len]
                    midi_pitches = midi_pitches[:min_len]
                    durs_frames = durs_frames[:min_len]
                    n_frames = sum(durs_frames)

                # Save phoneme-level features
                torch.save(torch.tensor(phoneme_ids, dtype=torch.long),
                            output_dir / "phonemes" / f"{utt_id}.pt")
                torch.save(torch.tensor(midi_pitches, dtype=torch.long),
                            output_dir / "pitch" / f"{utt_id}.pt")
                torch.save(torch.tensor(durs_frames, dtype=torch.long),
                            output_dir / "duration" / f"{utt_id}.pt")

                singer = batch.column("singer")[row_idx].as_py()
                metadata.append({
                    "id": utt_id,
                    "n_phones": min_len,
                    "n_frames": n_frames,
                    "language": "zh",
                    "lang_id": vocab.get_language_id("zh"),
                    "singing_ratio": 1.0,
                    "dataset": "ace_opencpop",
                    "singer": str(singer),
                })

            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {e}")
                continue

    return metadata


def _get_ipa_segments_for_pinyin(pinyin_phone: str) -> list:
    """Return how many IPA segments a pinyin phoneme decomposes into.

    Used to expand per-pinyin MIDI/feature lists to match the IPA segment count.
    """
    from ..data.phoneme import PINYIN_INITIAL_TO_IPA, PINYIN_FINAL_TO_IPA
    ph = pinyin_phone.rstrip("0123456789")
    if ph in ("SP", "AP", "br", "sil", "pau", "sp", ""):
        return ["SP"]
    if ph in PINYIN_INITIAL_TO_IPA:
        return PINYIN_INITIAL_TO_IPA[ph]
    if ph in PINYIN_FINAL_TO_IPA:
        return PINYIN_FINAL_TO_IPA[ph]
    return ["SP"]


def _get_ipa_segments_for_openjtalk(ja_phone: str) -> list:
    """Return how many IPA segments a Japanese phoneme decomposes into."""
    from ..data.phoneme import OPENJTALK_TO_IPA
    ph = ja_phone.strip()
    if ph in ("SP", "AP", "br", "sil", "pau", "sp", ""):
        return ["SP"]
    if ph in OPENJTALK_TO_IPA:
        return OPENJTALK_TO_IPA[ph]
    return ["SP"]


def _expand_note_midi_to_phoneme_list(note_midis: list, note_phns_list: list,
                                      n_phones: int) -> list:
    """Expand note-level MIDI to phoneme-level. note_phns_list is list<string> from Arrow.

    Each entry in note_phns_list is a string like "b_a1" (phonemes joined by _) for one note.
    """
    midi_expanded = []
    for i, phns_str in enumerate(note_phns_list):
        # Each note's phonemes are joined by _ (e.g., "g_an" = 2 phonemes)
        n_phns = len(phns_str.split("_")) if phns_str else 1
        midi = note_midis[i] if i < len(note_midis) else 0
        for _ in range(n_phns):
            midi_expanded.append(int(midi))

    while len(midi_expanded) < n_phones:
        midi_expanded.append(0)
    return midi_expanded[:n_phones]


def _expand_note_midi_to_phoneme(note_midis: list, note_phns_str: str,
                                  n_phones: int) -> list:
    """Expand note-level MIDI to phoneme-level based on note_phns mapping."""
    # note_phns format varies; common pattern: "b a1 | k e1 | ..."
    # Each group separated by | corresponds to one note
    if "|" in note_phns_str:
        groups = note_phns_str.split("|")
    else:
        groups = note_phns_str.strip().split()
        # If no | separator, assume one phone per note
        midi_expanded = []
        for i, phone in enumerate(groups):
            midi = note_midis[i] if i < len(note_midis) else 0
            midi_expanded.append(midi)
        while len(midi_expanded) < n_phones:
            midi_expanded.append(0)
        return midi_expanded[:n_phones]

    midi_expanded = []
    for i, group in enumerate(groups):
        phones_in_note = group.strip().split()
        midi = note_midis[i] if i < len(note_midis) else 0
        for _ in phones_in_note:
            midi_expanded.append(midi)

    while len(midi_expanded) < n_phones:
        midi_expanded.append(0)
    return midi_expanded[:n_phones]


def assign_splits(metadata: list, val_ratio: float = 0.05,
                  test_ratio: float = 0.05, seed: int = 42) -> list:
    """Assign train/val/test splits to metadata."""
    import random
    rng = random.Random(seed)
    rng.shuffle(metadata)

    n = len(metadata)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    for i, m in enumerate(metadata):
        if i < n_test:
            m["split"] = "test"
        elif i < n_test + n_val:
            m["split"] = "val"
        else:
            m["split"] = "train"

    return metadata


# ── Utilities ──

NOTE_TO_MIDI = {}
for octave in range(-1, 10):
    for i, name in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]):
        midi = (octave + 1) * 12 + i
        if 0 <= midi <= 127:
            NOTE_TO_MIDI[f"{name}{octave}"] = midi
            # Flat equivalents
            flat_name = {
                "C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"
            }.get(name)
            if flat_name:
                NOTE_TO_MIDI[f"{flat_name}{octave}"] = midi


def _note_name_to_midi(name: str) -> int:
    """Convert note name (e.g., 'C4', 'A#3') to MIDI number."""
    name = name.strip()
    if name.lower() in ("rest", "0", ""):
        return 0
    midi = NOTE_TO_MIDI.get(name)
    if midi is not None:
        return midi
    # Try case-insensitive
    for k, v in NOTE_TO_MIDI.items():
        if k.lower() == name.lower():
            return v
    logger.warning(f"Unknown note name: {name}, defaulting to C4 (60)")
    return 60


def _estimate_midi_from_f0(f0: torch.Tensor, durations: list) -> list:
    """Estimate MIDI note per phoneme from F0 contour."""
    midi_pitches = []
    frame_idx = 0

    for dur in durations:
        if frame_idx + dur > f0.size(0):
            dur = f0.size(0) - frame_idx
        if dur <= 0:
            midi_pitches.append(0)
            continue

        segment_f0 = f0[frame_idx:frame_idx + dur]
        voiced = segment_f0[segment_f0 > 50]

        if len(voiced) > 0:
            median_f0 = voiced.median().item()
            # Hz to MIDI: 69 + 12 * log2(f/440)
            midi = 69 + 12 * np.log2(median_f0 / 440.0)
            midi_pitches.append(int(round(midi)))
        else:
            midi_pitches.append(0)  # unvoiced

        frame_idx += dur

    return midi_pitches
