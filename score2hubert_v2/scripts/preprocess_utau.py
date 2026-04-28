"""
Score2HuBERT v2 — UTAU voice bank preprocessing.

Parses oto.ini files from Japanese UTAU banks and produces training samples with:
    hubert/{id}.pt       torch.float32 [T_frame, 768]   at 50fps
    f0/{id}.pt           torch.float32 [T_frame]        at 50fps
    phonemes/{id}.pt     torch.long    [T_phone]        IPA vocab IDs
    pitch/{id}.pt        torch.long    [T_phone]        MIDI note numbers
    duration/{id}.pt     torch.long    [T_phone]        frames/phoneme (sum == T_frame)
    metadata.json        list of dicts
    vocab.yaml           copied from ace_opencpop

UTAU alias examples handled:
    `- あD4`     initial (silence -> a)           -> [a]        (single vowel)
    `あD4`       bare kana with pitch             -> [a]        (treated as single vowel)
    `あ`         bare kana no pitch (folder pitch)-> [a]
    `- かD4`     initial CV                       -> [k, a]     split by consonant_ms
    `a いD4`     V-V transition                   -> [i]        (only the vowel target)
    `a きD4`     V-CV transition                  -> [k, i]     split by consonant_ms
    `a きゃD4`   V-CjV transition                 -> [k, j, a]
    `- きゃD4`   initial CjV                      -> [k, j, a]
    `_あ`        file-name-only placeholder       -> SKIP (no pitch, not usable)
    `a RD4`      release to silence               -> SKIP (duplicate of bare vowel)
    `a ・D4`     middle dot (utility)             -> SKIP (no IPA mapping)
    `a R吸`      breath in                        -> SKIP

Segment extraction (matches oto.ini convention):
    start_ms  = offset_ms
    if cutoff_ms < 0:  end_ms = start_ms + abs(cutoff_ms)       (length = |cutoff|)
    elif cutoff_ms > 0: end_ms = total_ms - cutoff_ms           (cutoff from end)
    else:               end_ms = total_ms

NOTE: In the UTAU oto.ini spec the cutoff sign IS inverted from typical time math:
    negative cutoff => fixed length from offset
    positive cutoff => trim from END of file
We match the canonical UTAU behaviour here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# Repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "RVC" / "RVC20240604Nvidia"))

from src.preprocessing.extract_hubert import HuBERTExtractor  # noqa: E402
from src.data.phoneme_v1 import PhonemeVocab  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess_utau")


# ─── Constants ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
FRAME_RATE = 50
HOP = SAMPLE_RATE // FRAME_RATE  # 320

MIN_SEG_SEC = 0.2
MAX_SEG_SEC = 5.0

# 8 banks to process (EXCLUDE 関西しのび and 中部つるぎ per spec)
UTAU_BANKS = [
    "ChugokuUsagi",
    "ずんだもん(連続音)",
    "つくよみちゃんUTAU音源",
    "九州そら",
    "北海道めろん",
    "四国めたん",
    "大江戸ちゃんこ(連続音)",
    "沖縄あわも(連続音)",
]

# Kana -> IPA
KANA_TO_IPA: Dict[str, List[str]] = {
    # Basic monographs
    "あ": ["a"], "い": ["i"], "う": ["ɯ"], "え": ["e"], "お": ["o"],
    "か": ["k", "a"], "き": ["k", "i"], "く": ["k", "ɯ"], "け": ["k", "e"], "こ": ["k", "o"],
    "さ": ["s", "a"], "し": ["ɕ", "i"], "す": ["s", "ɯ"], "せ": ["s", "e"], "そ": ["s", "o"],
    "た": ["t", "a"], "ち": ["tɕ", "i"], "つ": ["ts", "ɯ"], "て": ["t", "e"], "と": ["t", "o"],
    "な": ["n", "a"], "に": ["ɲ", "i"], "ぬ": ["n", "ɯ"], "ね": ["n", "e"], "の": ["n", "o"],
    "は": ["h", "a"], "ひ": ["ç", "i"], "ふ": ["ɸ", "ɯ"], "へ": ["h", "e"], "ほ": ["h", "o"],
    "ま": ["m", "a"], "み": ["m", "i"], "む": ["m", "ɯ"], "め": ["m", "e"], "も": ["m", "o"],
    "や": ["j", "a"], "ゆ": ["j", "ɯ"], "よ": ["j", "o"],
    "ら": ["ɾ", "a"], "り": ["ɾ", "i"], "る": ["ɾ", "ɯ"], "れ": ["ɾ", "e"], "ろ": ["ɾ", "o"],
    "わ": ["w", "a"], "を": ["o"], "ん": ["ɴ"],
    "が": ["g", "a"], "ぎ": ["g", "i"], "ぐ": ["g", "ɯ"], "げ": ["g", "e"], "ご": ["g", "o"],
    "ざ": ["z", "a"], "じ": ["dʑ", "i"], "ず": ["z", "ɯ"], "ぜ": ["z", "e"], "ぞ": ["z", "o"],
    "だ": ["d", "a"], "で": ["d", "e"], "ど": ["d", "o"],
    "ば": ["b", "a"], "び": ["b", "i"], "ぶ": ["b", "ɯ"], "べ": ["b", "e"], "ぼ": ["b", "o"],
    "ぱ": ["p", "a"], "ぴ": ["p", "i"], "ぷ": ["p", "ɯ"], "ぺ": ["p", "e"], "ぽ": ["p", "o"],
    # Palatalized (CjV)
    "きゃ": ["k", "j", "a"], "きゅ": ["k", "j", "ɯ"], "きょ": ["k", "j", "o"],
    "しゃ": ["ɕ", "j", "a"], "しゅ": ["ɕ", "j", "ɯ"], "しょ": ["ɕ", "j", "o"],
    "ちゃ": ["tɕ", "j", "a"], "ちゅ": ["tɕ", "j", "ɯ"], "ちょ": ["tɕ", "j", "o"],
    "にゃ": ["ɲ", "j", "a"], "にゅ": ["ɲ", "j", "ɯ"], "にょ": ["ɲ", "j", "o"],
    "ひゃ": ["ç", "j", "a"], "ひゅ": ["ç", "j", "ɯ"], "ひょ": ["ç", "j", "o"],
    "みゃ": ["m", "j", "a"], "みゅ": ["m", "j", "ɯ"], "みょ": ["m", "j", "o"],
    "りゃ": ["ɾ", "j", "a"], "りゅ": ["ɾ", "j", "ɯ"], "りょ": ["ɾ", "j", "o"],
    "ぎゃ": ["g", "j", "a"], "ぎゅ": ["g", "j", "ɯ"], "ぎょ": ["g", "j", "o"],
    "じゃ": ["dʑ", "j", "a"], "じゅ": ["dʑ", "j", "ɯ"], "じょ": ["dʑ", "j", "o"],
    "びゃ": ["b", "j", "a"], "びゅ": ["b", "j", "ɯ"], "びょ": ["b", "j", "o"],
    "ぴゃ": ["p", "j", "a"], "ぴゅ": ["p", "j", "ɯ"], "ぴょ": ["p", "j", "o"],
}

# note-name -> MIDI (C-1 = 0). C4 = 60.
PITCH_NAME_TO_OFFSET = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 12,
}

# Match C4, D#4, Db4, A#5, etc. at end of string.
PITCH_RE = re.compile(r"([CDEFGAB][#b]?)(\d)$")


def note_to_midi(note: str) -> Optional[int]:
    """Convert e.g. 'D4' -> 62, 'C5' -> 72, 'A#3' -> 58."""
    m = PITCH_RE.match(note)
    if not m:
        return None
    name, octv = m.group(1), int(m.group(2))
    off = PITCH_NAME_TO_OFFSET.get(name)
    if off is None:
        return None
    # MIDI: C0=12, so (octv+1)*12 + off
    return (octv + 1) * 12 + off


def parse_pitch_from_folder(folder_name: str) -> Optional[int]:
    """Many banks have pitch in folder name: 'C4', 'ちゃんこE4', 'ずんだもんD4', '多音階02：_D#4（連続音）'."""
    m = PITCH_RE.search(folder_name)
    if m:
        return note_to_midi(folder_name[m.start() : m.end()])
    # Look anywhere for a pitch marker followed by non-alphanumeric
    for m in re.finditer(r"([CDEFGAB][#b]?)(\d)", folder_name):
        midi = note_to_midi(m.group())
        if midi is not None and 24 <= midi <= 96:  # sanity range
            return midi
    return None


def parse_alias_pitch(alias: str) -> Tuple[str, Optional[int]]:
    """Strip the pitch token (if any) from end of alias and return (kana_part, midi).

    Supports plain "D4" or underscored "_D#4" or no pitch.
    """
    s = alias.rstrip()
    # Underscored: "..._D#4"
    m = re.search(r"_([CDEFGAB][#b]?\d)$", s)
    if m:
        midi = note_to_midi(m.group(1))
        return s[: m.start()].rstrip(), midi
    # Plain: "...D4" — but careful to not eat a trailing kana that happens to end with letter+digit
    m = PITCH_RE.search(s)
    if m:
        midi = note_to_midi(m.group())
        if midi is not None:
            return s[: m.start()].rstrip(), midi
    return s, None


# Longest-match kana tokenizer. We need to split e.g. "きゃ" as one unit before "き" alone.
_DIACRITICS = "\u3099\u309A"


def _starts_with_kana(s: str, idx: int, key: str) -> bool:
    return s.startswith(key, idx)


def kana_to_phonemes(kana_str: str) -> Optional[List[str]]:
    """Convert a SHORT kana string (1-3 chars usually) to IPA phonemes.

    The UTAU oto.ini target kana is typically 1 kana (mora), possibly with 小 kana
    forming a palatalized mora (e.g. 'きゃ'). Returns None if not recognizable.
    """
    s = kana_str.strip()
    if not s:
        return None

    # Try a 2-char key first for palatalized kana (CjV forms)
    if len(s) >= 2 and s[:2] in KANA_TO_IPA:
        key = s[:2]
        if len(s) == 2:
            return list(KANA_TO_IPA[key])
        # extra trailing chars not supported
        return None
    if len(s) == 1 and s in KANA_TO_IPA:
        return list(KANA_TO_IPA[s])
    return None


# ─── oto.ini parsing ────────────────────────────────────────────────────────


def parse_oto_line(line: str) -> Optional[Dict]:
    """Parse one oto.ini line.

    Format: `wavfile=alias,offset_ms,consonant_ms,cutoff_ms,preutterance_ms,overlap_ms`
    """
    if "=" not in line:
        return None
    wavfile, rest = line.split("=", 1)
    parts = rest.split(",")
    if len(parts) < 5:
        return None
    try:
        alias = parts[0]
        offset_ms = float(parts[1])
        consonant_ms = float(parts[2])
        cutoff_ms = float(parts[3])
        # preutterance and overlap are unused for segment extraction
        return {
            "wavfile": wavfile.strip(),
            "alias": alias.strip(),
            "offset_ms": offset_ms,
            "consonant_ms": consonant_ms,
            "cutoff_ms": cutoff_ms,
        }
    except ValueError:
        return None


def alias_to_phonemes(alias: str, folder_pitch: Optional[int]) -> Optional[Dict]:
    """Convert an oto.ini alias to {phonemes, pitch, n_consonant_phones}.

    Returns None if unusable.
    """
    raw = alias.strip()
    # Strip pitch from the alias end (may be absent for folder-pitch banks)
    kana_part, alias_pitch = parse_alias_pitch(raw)
    pitch = alias_pitch if alias_pitch is not None else folder_pitch
    if pitch is None:
        return None

    # Reject placeholders and non-pitched isolated file-name aliases
    if kana_part.startswith("_"):
        return None
    # Reject release/breath/special markers
    # Patterns: "a R", "a ・", "R", etc.
    if re.search(r"(^|\s)[R・]", kana_part):
        return None
    # Reject breath-in markers 吸/息
    if "吸" in kana_part or "息" in kana_part:
        return None

    # Split prev/target (V-V or V-CV uses a space separator)
    if " " in kana_part:
        prev, target = kana_part.split(" ", 1)
        prev = prev.strip()
        target = target.strip()
    else:
        prev = ""
        target = kana_part

    # "- あ" : initial (silence -> target)
    if prev == "-":
        ipa = kana_to_phonemes(target)
        if ipa is None:
            return None
        # For initial the first phoneme(s) up to the vowel are consonant; last is vowel
        # Assume n_consonant = len(ipa) - 1
        n_cons = max(0, len(ipa) - 1)
        return {"phonemes": ipa, "pitch": pitch, "n_consonant": n_cons, "kind": "initial"}

    # "" (bare target): treat as single-vowel (or CV depending on mapping)
    if not prev:
        ipa = kana_to_phonemes(target)
        if ipa is None:
            return None
        n_cons = max(0, len(ipa) - 1)
        return {"phonemes": ipa, "pitch": pitch, "n_consonant": n_cons, "kind": "bare"}

    # prev is a phoneme letter like 'a', 'i', 'u', 'e', 'o' (or 'N', 'n', etc.)
    # V-CV / V-V transition — keep only TARGET phonemes
    ipa = kana_to_phonemes(target)
    if ipa is None:
        return None
    n_cons = max(0, len(ipa) - 1)
    return {"phonemes": ipa, "pitch": pitch, "n_consonant": n_cons, "kind": "transition"}


# ─── Audio segment extraction ──────────────────────────────────────────────


def extract_segment(
    audio16k: np.ndarray,
    total_ms: float,
    offset_ms: float,
    cutoff_ms: float,
) -> Optional[np.ndarray]:
    """Return audio slice for the oto.ini entry at 16kHz.

    cutoff_ms convention (UTAU):
        < 0: segment length = |cutoff_ms| (counted FROM offset)
        > 0: end point = total_ms - cutoff_ms (trim from END of file)
        == 0: go to end of file
    """
    start_ms = offset_ms
    if cutoff_ms < 0:
        end_ms = start_ms + abs(cutoff_ms)
    elif cutoff_ms > 0:
        end_ms = total_ms - cutoff_ms
    else:
        end_ms = total_ms

    start_ms = max(0.0, start_ms)
    end_ms = min(total_ms, end_ms)
    dur_ms = end_ms - start_ms
    if dur_ms <= 0:
        return None

    dur_sec = dur_ms / 1000.0
    if dur_sec < MIN_SEG_SEC or dur_sec > MAX_SEG_SEC:
        return None

    s = int(round(start_ms / 1000.0 * SAMPLE_RATE))
    e = int(round(end_ms / 1000.0 * SAMPLE_RATE))
    s = max(0, min(s, len(audio16k)))
    e = max(0, min(e, len(audio16k)))
    if e - s < int(MIN_SEG_SEC * SAMPLE_RATE):
        return None
    return audio16k[s:e].astype(np.float32)


# ─── F0 / HuBERT for a numpy array ──────────────────────────────────────────


@torch.no_grad()
def hubert_from_array(model, audio16k: np.ndarray, device: str, use_fp16: bool) -> torch.Tensor:
    """Run HuBERT on a raw audio array (matches HuBERTExtractor.extract but skips I/O)."""
    audio = torch.from_numpy(audio16k).float().unsqueeze(0).to(device)
    if use_fp16:
        audio = audio.half()
    audio = F.pad(audio, (40, 40), mode="reflect")
    feats = model.extract_features(source=audio, padding_mask=None, output_layer=12)
    feat = feats[0].squeeze(0).float().cpu()
    return feat


def rmvpe_to_frames(rmvpe, audio16k: np.ndarray, n_target: int, device: str) -> torch.Tensor:
    """Run RMVPE (100fps), then resample to n_target (50fps)."""
    # RMVPE expects torch tensor (see RVC source): audio float32, on device is fine
    f0_100 = rmvpe.infer_from_audio(audio16k.astype(np.float32), thred=0.03)
    f0_100 = np.asarray(f0_100, dtype=np.float32)  # [T_100]
    if f0_100.size == 0 or n_target <= 0:
        return torch.zeros(n_target, dtype=torch.float32)

    # Interpolate to n_target using nearest-in-time mapping
    # Source time: i*0.01s; target time: j*0.02s => src_idx = j*2
    src_len = f0_100.shape[0]
    idxs = np.clip((np.arange(n_target) * 2), 0, src_len - 1)
    f0 = f0_100[idxs]
    return torch.from_numpy(f0).float()


# ─── Main per-bank processing ──────────────────────────────────────────────


def collect_oto_entries(bank_dir: Path) -> List[Tuple[Path, Path, Optional[int]]]:
    """Return list of (oto_ini_path, wav_base_dir, folder_pitch_hint).

    Walks the bank recursively to find every dir that has an oto.ini + at least
    one .wav file. Skips breath-only folders. Handles both common layouts:
      - bank/character/pitch/oto.ini  (e.g. ChugokuUsagi)
      - bank/pitch/oto.ini             (e.g. ずんだもん)
    """
    found: List[Tuple[Path, Path, Optional[int]]] = []
    for root, dirs, files in os.walk(bank_dir):
        root_p = Path(root)
        if "oto.ini" not in files:
            continue
        if not any(f.lower().endswith(".wav") for f in files):
            continue
        name = root_p.name
        if any(k in name for k in ["ブレス", "breath", "単独ブレス"]):
            continue
        oto = root_p / "oto.ini"
        folder_pitch = parse_pitch_from_folder(name)
        found.append((oto, root_p, folder_pitch))
    return sorted(found, key=lambda t: str(t[0]))


def sanitize_bank_name(s: str) -> str:
    # Drop parens, non-ASCII -> romaji-ish (use hashing of chars)
    mapping = {
        "ChugokuUsagi": "ChugokuUsagi",
        "ずんだもん(連続音)": "Zundamon",
        "つくよみちゃんUTAU音源": "Tsukuyomi",
        "九州そら": "KyushuSora",
        "北海道めろん": "HokkaidoMelon",
        "四国めたん": "ShikokuMetan",
        "大江戸ちゃんこ(連続音)": "OoedoChanko",
        "沖縄あわも(連続音)": "OkinawaAwamo",
    }
    return mapping.get(s, re.sub(r"[^A-Za-z0-9]", "_", s))


def split_for(utt_id: str) -> str:
    h = int(hashlib.md5(utt_id.encode("utf-8")).hexdigest(), 16)
    return "train" if (h % 10) < 9 else "test"


def distribute_duration(T: int, n_phones: int, n_cons: int, consonant_ms: float) -> List[int]:
    """Distribute T frames across n_phones.

    n_cons: number of leading consonant/glide phonemes.
    consonant_ms: consonant length from oto.ini.
    """
    if n_phones == 1:
        return [T]
    # Frames occupied by consonants combined
    total_cons_frames = max(n_cons, int(round(consonant_ms / 1000.0 * FRAME_RATE)))
    total_cons_frames = max(n_cons, min(total_cons_frames, T - 1))  # leave at least 1 for vowel
    vowel_frames = T - total_cons_frames
    if vowel_frames < 1:
        vowel_frames = 1
        total_cons_frames = T - 1

    durations: List[int] = []
    if n_cons == 0:
        # All frames go to the vowel(s) — only possible if n_phones == 1 (already handled)
        per = T // n_phones
        rem = T - per * n_phones
        for i in range(n_phones):
            durations.append(per + (1 if i < rem else 0))
    else:
        # Split consonant frames across n_cons leading consonant phones
        cons_per = total_cons_frames // n_cons
        cons_rem = total_cons_frames - cons_per * n_cons
        for i in range(n_cons):
            durations.append(cons_per + (1 if i < cons_rem else 0))
        # Remaining phones are vowels (usually 1)
        n_vowels = n_phones - n_cons
        vow_per = vowel_frames // n_vowels
        vow_rem = vowel_frames - vow_per * n_vowels
        for i in range(n_vowels):
            durations.append(vow_per + (1 if i < vow_rem else 0))

    # Guarantee each >=1 and sum exactly T
    for i in range(len(durations)):
        if durations[i] < 1:
            durations[i] = 1
    diff = T - sum(durations)
    # Fix overshoot/undershoot on the longest element
    if diff != 0:
        j = int(np.argmax(durations))
        durations[j] += diff
    # Final safety clamp
    for i in range(len(durations)):
        if durations[i] < 1:
            durations[i] = 1
    diff = T - sum(durations)
    if diff != 0:
        j = int(np.argmax(durations))
        durations[j] += diff

    return durations


def process_one_bank(
    bank: str,
    utau_root: Path,
    out_dir: Path,
    vocab: PhonemeVocab,
    hubert_model,
    rmvpe,
    device: str,
    use_fp16: bool,
) -> List[Dict]:
    """Process one UTAU bank. Returns metadata list."""
    bank_sanitized = sanitize_bank_name(bank)
    bank_dir = utau_root / bank
    if not bank_dir.exists():
        logger.warning(f"Bank dir missing: {bank_dir}")
        return []

    oto_groups = collect_oto_entries(bank_dir)
    logger.info(f"[{bank}] found {len(oto_groups)} oto.ini groups")

    # Flatten list of entries, remembering which oto file they came from
    entry_list: List[Tuple[Path, str, Dict, Optional[int]]] = []
    for oto_path, wav_dir, folder_pitch in oto_groups:
        try:
            with open(oto_path, "r", encoding="shift_jis", errors="ignore") as f:
                raw_lines = f.readlines()
        except OSError as e:
            logger.warning(f"Failed to read {oto_path}: {e}")
            continue
        for line in raw_lines:
            parsed = parse_oto_line(line.strip())
            if parsed is None:
                continue
            entry_list.append((wav_dir, parsed["wavfile"], parsed, folder_pitch))

    logger.info(f"[{bank}] total oto entries: {len(entry_list)}")

    # Group by wav file so we only load each once
    by_wav: Dict[Tuple[Path, str], List[Tuple[Dict, Optional[int]]]] = {}
    for wav_dir, wavfile, parsed, folder_pitch in entry_list:
        key = (wav_dir, wavfile)
        by_wav.setdefault(key, []).append((parsed, folder_pitch))

    meta: List[Dict] = []
    entry_idx = 0
    skipped_alias = 0
    skipped_segment = 0
    errors = 0

    hubert_dir = out_dir / "hubert"
    f0_dir = out_dir / "f0"
    ph_dir = out_dir / "phonemes"
    pitch_dir = out_dir / "pitch"
    dur_dir = out_dir / "duration"
    for d in (hubert_dir, f0_dir, ph_dir, pitch_dir, dur_dir):
        d.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(by_wav.items(), desc=f"[{bank_sanitized}]", unit="wav")
    for (wav_dir, wavfile), group in pbar:
        wav_path = wav_dir / wavfile
        if not wav_path.exists():
            # Try case-sensitive / alt-encoded match
            cand = list(wav_dir.glob(wavfile))
            if not cand:
                continue
            wav_path = cand[0]
        try:
            audio, sr = sf.read(str(wav_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
            else:
                audio = audio.astype(np.float32)
            total_ms = len(audio) / SAMPLE_RATE * 1000.0
        except Exception as e:
            logger.debug(f"load fail {wav_path}: {e}")
            errors += 1
            continue

        for parsed, folder_pitch in group:
            alias_info = alias_to_phonemes(parsed["alias"], folder_pitch)
            if alias_info is None:
                skipped_alias += 1
                continue
            segment = extract_segment(audio, total_ms, parsed["offset_ms"], parsed["cutoff_ms"])
            if segment is None:
                skipped_segment += 1
                continue
            try:
                hubert = hubert_from_array(hubert_model, segment, device=device, use_fp16=use_fp16)
                T = hubert.shape[0]
                if T < 2:
                    skipped_segment += 1
                    continue
                f0 = rmvpe_to_frames(rmvpe, segment, T, device=device)
            except Exception as e:
                logger.debug(f"feature fail {parsed['alias']}: {e}")
                errors += 1
                continue

            ipa_list = alias_info["phonemes"]
            ph_ids = vocab.encode(ipa_list)
            midi = alias_info["pitch"]
            n_cons = alias_info["n_consonant"]
            durations = distribute_duration(T, len(ipa_list), n_cons, parsed["consonant_ms"])
            assert sum(durations) == T, f"dur sum {sum(durations)} != T {T}"

            utt_id = f"utau_{bank_sanitized}_{entry_idx:05d}"
            entry_idx += 1

            torch.save(hubert, hubert_dir / f"{utt_id}.pt")
            torch.save(f0, f0_dir / f"{utt_id}.pt")
            torch.save(torch.tensor(ph_ids, dtype=torch.long), ph_dir / f"{utt_id}.pt")
            torch.save(torch.full((len(ipa_list),), midi, dtype=torch.long), pitch_dir / f"{utt_id}.pt")
            torch.save(torch.tensor(durations, dtype=torch.long), dur_dir / f"{utt_id}.pt")

            meta.append({
                "id": utt_id,
                "n_phones": len(ipa_list),
                "n_frames": T,
                "language": "ja",
                "lang_id": vocab.get_language_id("ja"),
                "singing_ratio": 1.0,
                "dataset": "utau",
                "singer": bank_sanitized,
                "split": split_for(utt_id),
            })
        pbar.set_postfix(entries=entry_idx, skip_alias=skipped_alias, skip_seg=skipped_segment, err=errors)

    logger.info(
        f"[{bank}] produced {entry_idx} entries  "
        f"(skipped alias={skipped_alias}, seg={skipped_segment}, err={errors})"
    )
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utau-root", default=str(REPO_ROOT / "kiritan" / "Utau"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "processed" / "utau"))
    parser.add_argument("--hubert-model", default=str(REPO_ROOT / "RVC/RVC20240604Nvidia/assets/hubert/hubert_base.pt"))
    parser.add_argument("--rmvpe-model", default=str(REPO_ROOT / "RVC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt"))
    parser.add_argument("--vocab", default=str(REPO_ROOT / "data/processed/ace_opencpop/vocab.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--banks", nargs="*", default=None, help="Override bank list")
    args = parser.parse_args()

    utau_root = Path(args.utau_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vocab
    vocab = PhonemeVocab(args.vocab)
    # Copy vocab.yaml to out_dir
    shutil.copy(args.vocab, out_dir / "vocab.yaml")

    # HuBERT
    logger.info("Loading HuBERT...")
    hubert_ext = HuBERTExtractor(args.hubert_model, layer=12, device=args.device, use_fp16=args.fp16)
    hubert_model = hubert_ext.model  # lower-level handle so we can feed arrays

    # RMVPE
    logger.info("Loading RMVPE...")
    sys.path.insert(0, str(REPO_ROOT / "RVC" / "RVC20240604Nvidia"))
    sys.path.insert(0, str(REPO_ROOT / "RVC" / "RVC20240604Nvidia" / "infer"))
    from infer.lib.rmvpe import RMVPE
    rmvpe = RMVPE(args.rmvpe_model, is_half=args.fp16, device=args.device)

    banks = args.banks if args.banks else UTAU_BANKS
    all_meta: List[Dict] = []
    for bank in banks:
        m = process_one_bank(
            bank,
            utau_root=utau_root,
            out_dir=out_dir,
            vocab=vocab,
            hubert_model=hubert_model,
            rmvpe=rmvpe,
            device=args.device,
            use_fp16=args.fp16,
        )
        all_meta.extend(m)

    # Save metadata
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=0)

    total_frames = sum(e["n_frames"] for e in all_meta)
    total_hours = total_frames / FRAME_RATE / 3600
    logger.info(f"Done. total={len(all_meta)}, frames={total_frames}, hours={total_hours:.2f}")


if __name__ == "__main__":
    main()
