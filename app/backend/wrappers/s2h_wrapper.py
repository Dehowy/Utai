"""Score2HuBERT synthesis wrapper.

Replaces the DiffSinger (UtaiRenderer.exe) pipeline with:
  USTX notes → G2P → Score2HuBERT model → RVC decode → post mute → WAV

Model: score2hubert_v2/archive/run9_final/best.pt (54.63M params)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

from app.backend.config import PROJECT_ROOT, RVC_ROOT, get_device

logger = logging.getLogger("utai.synth")

S2H_ROOT = PROJECT_ROOT / "score2hubert_v2"
_DIST_MODEL = PROJECT_ROOT / "synth" / "models" / "Jinguji_Suzumi.pt"
_DEV_MODEL = S2H_ROOT / "archive" / "run9_final" / "best.pt"
S2H_MODEL_PATH = _DIST_MODEL if _DIST_MODEL.exists() else _DEV_MODEL
S2H_VOCAB_PATH = S2H_ROOT / "data" / "processed" / "ace_opencpop" / "vocab.yaml"

RVC_WEIGHTS_DIR = RVC_ROOT / "assets" / "weights"
RVC_INDICES_DIR = RVC_ROOT / "assets" / "indices"

_BUILTIN_VOCAB = {
    "lang2id": {"en": 2, "ja": 0, "ko": 3, "zh": 1},
    "phone2id": {
        "<pad>": 0, "<sos>": 1, "<eos>": 2, "SP": 3, "AP": 4, "br": 5,
        "<mask>": 6, "i": 7, "y": 8, "ɪ": 9, "e": 10, "ɛ": 11,
        "æ": 12, "ɨ": 13, "ə": 14, "ɚ": 15, "ɐ": 16, "a": 17,
        "ɑ": 18, "ɒ": 19, "ɔ": 20, "o": 21, "ɤ": 22, "ʊ": 23,
        "u": 24, "ɯ": 25, "p": 26, "pʰ": 27, "b": 28, "t": 29,
        "tʰ": 30, "d": 31, "k": 32, "kʰ": 33, "g": 34, "ʔ": 35,
        "ts": 36, "tsʰ": 37, "dz": 38, "tɕ": 39, "tɕʰ": 40,
        "dʑ": 41, "ʈʂ": 42, "ʈʂʰ": 43, "ɖʐ": 44, "tʃ": 45,
        "dʒ": 46, "f": 47, "v": 48, "s": 49, "z": 50, "ɕ": 51,
        "ʑ": 52, "ʂ": 53, "ʐ": 54, "ʃ": 55, "ʒ": 56, "ç": 57,
        "ɸ": 58, "x": 59, "ɣ": 60, "h": 61, "m": 62, "n": 63,
        "ɲ": 64, "ŋ": 65, "ɴ": 66, "l": 67, "ɾ": 68, "ɹ": 69,
        "j": 70, "w": 71, "ɥ": 72,
        "<unused_73>": 73, "<unused_74>": 74, "<unused_75>": 75,
        "<unused_76>": 76, "<unused_77>": 77, "<unused_78>": 78,
        "<unused_79>": 79,
    },
}

SP_ID = 3
CONSONANT_FRAMES = 3
SAMPLES_PER_FRAME = 960  # 48kHz / 50fps


# ── G2P helpers ──────────────────────────────────────────────────────────

def _ensure_s2h_path():
    if str(S2H_ROOT) not in sys.path:
        sys.path.insert(0, str(S2H_ROOT))


def _chinese_lyric_to_ipa(lyric: str) -> list[str]:
    """Single Chinese character/syllable → IPA phoneme list."""
    try:
        from pypinyin import lazy_pinyin, Style
    except ImportError:
        logger.error("pypinyin not installed — Chinese G2P disabled. Run: pip install pypinyin")
        return ["a"]

    _ensure_s2h_path()
    from src.data.phoneme_v1 import PINYIN_INITIAL_TO_IPA, PINYIN_FINAL_TO_IPA

    pinyin = lazy_pinyin(lyric, style=Style.NORMAL)
    if not pinyin:
        return ["a"]
    py = pinyin[0].strip().lower()
    if not py:
        return ["a"]

    initial, final = _split_pinyin(py)
    ipa: list[str] = []
    if initial and initial in PINYIN_INITIAL_TO_IPA:
        ipa.extend(PINYIN_INITIAL_TO_IPA[initial])
    if final and final in PINYIN_FINAL_TO_IPA:
        ipa.extend(PINYIN_FINAL_TO_IPA[final])
    return ipa if ipa else ["a"]


_ZH_INITIALS = [
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w",
]


def _split_pinyin(py: str) -> tuple[str, str]:
    """Split pinyin syllable into (initial, final)."""
    for ini in _ZH_INITIALS:
        if py.startswith(ini):
            return ini, py[len(ini):] or "i"
    return "", py


_KANA_TO_OPENJTALK: dict[str, list[str]] = {}


def _build_kana_table():
    if _KANA_TO_OPENJTALK:
        return
    vowels = {"あ": "a", "い": "i", "う": "u", "え": "e", "お": "o"}
    rows = {
        "か": ("k",), "さ": ("s",), "た": ("t",), "な": ("n",), "は": ("h",),
        "ま": ("m",), "ら": ("r",), "が": ("g",), "ざ": ("z",), "だ": ("d",),
        "ば": ("b",), "ぱ": ("p",),
    }
    v_list = ["a", "i", "u", "e", "o"]
    v_chars = "あいうえお"
    for v_c, v_p in zip(v_chars, v_list):
        _KANA_TO_OPENJTALK[v_c] = [v_p]
        # katakana
        _KANA_TO_OPENJTALK[chr(ord(v_c) + 0x60)] = [v_p]

    row_bases = [
        ("か", "k", "あいうえお"),
        ("さ", "s", "あいうえお"),
        ("た", "t", "あいうえお"),
        ("な", "n", "あいうえお"),
        ("は", "h", "あいうえお"),
        ("ま", "m", "あいうえお"),
        ("や", "y", "あ_う_お"),
        ("ら", "r", "あいうえお"),
        ("わ", "w", "あ____"),
    ]
    # Hard-code the full table for correctness
    _map = {
        "か": ["k", "a"], "き": ["k", "i"], "く": ["k", "u"], "け": ["k", "e"], "こ": ["k", "o"],
        "さ": ["s", "a"], "し": ["sh", "i"], "す": ["s", "u"], "せ": ["s", "e"], "そ": ["s", "o"],
        "た": ["t", "a"], "ち": ["ch", "i"], "つ": ["ts", "u"], "て": ["t", "e"], "と": ["t", "o"],
        "な": ["n", "a"], "に": ["n", "i"], "ぬ": ["n", "u"], "ね": ["n", "e"], "の": ["n", "o"],
        "は": ["h", "a"], "ひ": ["h", "i"], "ふ": ["f", "u"], "へ": ["h", "e"], "ほ": ["h", "o"],
        "ま": ["m", "a"], "み": ["m", "i"], "む": ["m", "u"], "め": ["m", "e"], "も": ["m", "o"],
        "や": ["y", "a"], "ゆ": ["y", "u"], "よ": ["y", "o"],
        "ら": ["r", "a"], "り": ["r", "i"], "る": ["r", "u"], "れ": ["r", "e"], "ろ": ["r", "o"],
        "わ": ["w", "a"], "を": ["o"], "ん": ["N"],
        "が": ["g", "a"], "ぎ": ["g", "i"], "ぐ": ["g", "u"], "げ": ["g", "e"], "ご": ["g", "o"],
        "ざ": ["z", "a"], "じ": ["j", "i"], "ず": ["z", "u"], "ぜ": ["z", "e"], "ぞ": ["z", "o"],
        "だ": ["d", "a"], "ぢ": ["j", "i"], "づ": ["z", "u"], "で": ["d", "e"], "ど": ["d", "o"],
        "ば": ["b", "a"], "び": ["b", "i"], "ぶ": ["b", "u"], "べ": ["b", "e"], "ぼ": ["b", "o"],
        "ぱ": ["p", "a"], "ぴ": ["p", "i"], "ぷ": ["p", "u"], "ぺ": ["p", "e"], "ぽ": ["p", "o"],
        "きゃ": ["ky", "a"], "きゅ": ["ky", "u"], "きょ": ["ky", "o"],
        "しゃ": ["sh", "a"], "しゅ": ["sh", "u"], "しょ": ["sh", "o"],
        "ちゃ": ["ch", "a"], "ちゅ": ["ch", "u"], "ちょ": ["ch", "o"],
        "にゃ": ["ny", "a"], "にゅ": ["ny", "u"], "にょ": ["ny", "o"],
        "ひゃ": ["hy", "a"], "ひゅ": ["hy", "u"], "ひょ": ["hy", "o"],
        "みゃ": ["m", "y", "a"], "みゅ": ["m", "y", "u"], "みょ": ["m", "y", "o"],
        "りゃ": ["r", "y", "a"], "りゅ": ["r", "y", "u"], "りょ": ["r", "y", "o"],
        "ぎゃ": ["gy", "a"], "ぎゅ": ["gy", "u"], "ぎょ": ["gy", "o"],
        "じゃ": ["j", "a"], "じゅ": ["j", "u"], "じょ": ["j", "o"],
        "びゃ": ["by", "a"], "びゅ": ["by", "u"], "びょ": ["by", "o"],
        "ぴゃ": ["py", "a"], "ぴゅ": ["py", "u"], "ぴょ": ["py", "o"],
        "っ": ["cl"],
        "ー": [],  # long vowel mark — extend previous
        # Small kana vowels (used in foreign loan combinations)
        "ぁ": ["a"], "ぃ": ["i"], "ぅ": ["u"], "ぇ": ["e"], "ぉ": ["o"],
        # Foreign loan combinations (ファ/ティ/ディ/ヴ etc.)
        "てぃ": ["t", "i"], "でぃ": ["d", "i"],
        "てゅ": ["t", "y", "u"], "でゅ": ["d", "y", "u"],
        "とぅ": ["t", "u"], "どぅ": ["d", "u"],
        "つぁ": ["ts", "a"], "つぃ": ["ts", "i"], "つぇ": ["ts", "e"], "つぉ": ["ts", "o"],
        "ふぁ": ["f", "a"], "ふぃ": ["f", "i"], "ふぇ": ["f", "e"], "ふぉ": ["f", "o"],
        "ふゅ": ["f", "y", "u"],
        "うぃ": ["w", "i"], "うぇ": ["w", "e"], "うぉ": ["w", "o"],
        "ゔぁ": ["v", "a"], "ゔぃ": ["v", "i"], "ゔ": ["v", "u"], "ゔぇ": ["v", "e"], "ゔぉ": ["v", "o"],
        "いぇ": ["y", "e"],
        "しぇ": ["sh", "e"], "じぇ": ["j", "e"], "ちぇ": ["ch", "e"],
        "きぇ": ["k", "y", "e"], "ぎぇ": ["g", "y", "e"],
        "くぁ": ["k", "w", "a"], "ぐぁ": ["g", "w", "a"],
    }
    _KANA_TO_OPENJTALK.update(_map)
    # Add katakana equivalents
    for k, v in list(_map.items()):
        kata = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ん" else c for c in k)
        if kata != k:
            _KANA_TO_OPENJTALK[kata] = v
    # ヴ row — add all kana-mixing variants (ヴァ, ヴぁ, etc.)
    for vu_char in ("ヴ", "ゔ"):
        _KANA_TO_OPENJTALK[vu_char] = ["v", "u"]
        for sm_a, sm_ka, phone in [("ぁ","ァ","a"),("ぃ","ィ","i"),("ぇ","ェ","e"),("ぉ","ォ","o")]:
            _KANA_TO_OPENJTALK[vu_char + sm_a] = ["v", phone]
            _KANA_TO_OPENJTALK[vu_char + sm_ka] = ["v", phone]


_ROMAJI_TO_OPENJTALK: dict[str, list[str]] = {}


def _build_romaji_table():
    if _ROMAJI_TO_OPENJTALK:
        return
    _m: dict[str, list[str]] = {
        # Vowels
        "a": ["a"], "i": ["i"], "u": ["u"], "e": ["e"], "o": ["o"],
        # Ka-row
        "ka": ["k", "a"], "ki": ["k", "i"], "ku": ["k", "u"], "ke": ["k", "e"], "ko": ["k", "o"],
        # Sa-row (Hepburn + Nihon-shiki)
        "sa": ["s", "a"], "si": ["sh", "i"], "shi": ["sh", "i"], "su": ["s", "u"], "se": ["s", "e"], "so": ["s", "o"],
        # Ta-row (Hepburn + Nihon-shiki)
        "ta": ["t", "a"], "ti": ["ch", "i"], "chi": ["ch", "i"], "tu": ["ts", "u"], "tsu": ["ts", "u"], "te": ["t", "e"], "to": ["t", "o"],
        # Na-row
        "na": ["n", "a"], "ni": ["n", "i"], "nu": ["n", "u"], "ne": ["n", "e"], "no": ["n", "o"],
        # Ha-row
        "ha": ["h", "a"], "hi": ["h", "i"], "hu": ["f", "u"], "fu": ["f", "u"], "he": ["h", "e"], "ho": ["h", "o"],
        # Ma-row
        "ma": ["m", "a"], "mi": ["m", "i"], "mu": ["m", "u"], "me": ["m", "e"], "mo": ["m", "o"],
        # Ya-row
        "ya": ["y", "a"], "yu": ["y", "u"], "yo": ["y", "o"],
        # Ra-row
        "ra": ["r", "a"], "ri": ["r", "i"], "ru": ["r", "u"], "re": ["r", "e"], "ro": ["r", "o"],
        # Wa-row + N
        "wa": ["w", "a"], "wo": ["o"], "nn": ["N"], "n": ["N"],
        # Ga-row
        "ga": ["g", "a"], "gi": ["g", "i"], "gu": ["g", "u"], "ge": ["g", "e"], "go": ["g", "o"],
        # Za-row
        "za": ["z", "a"], "zi": ["j", "i"], "ji": ["j", "i"], "zu": ["z", "u"], "ze": ["z", "e"], "zo": ["z", "o"],
        # Da-row (ぢ→じ, づ→ず phonologically)
        "da": ["d", "a"], "de": ["d", "e"], "do": ["d", "o"],
        # Ba-row
        "ba": ["b", "a"], "bi": ["b", "i"], "bu": ["b", "u"], "be": ["b", "e"], "bo": ["b", "o"],
        # Pa-row
        "pa": ["p", "a"], "pi": ["p", "i"], "pu": ["p", "u"], "pe": ["p", "e"], "po": ["p", "o"],
        # Yōon — Hepburn
        "kya": ["ky", "a"], "kyu": ["ky", "u"], "kyo": ["ky", "o"],
        "sha": ["sh", "a"], "shu": ["sh", "u"], "sho": ["sh", "o"],
        "cha": ["ch", "a"], "chu": ["ch", "u"], "cho": ["ch", "o"],
        "nya": ["ny", "a"], "nyu": ["ny", "u"], "nyo": ["ny", "o"],
        "hya": ["hy", "a"], "hyu": ["hy", "u"], "hyo": ["hy", "o"],
        "mya": ["m", "y", "a"], "myu": ["m", "y", "u"], "myo": ["m", "y", "o"],
        "rya": ["r", "y", "a"], "ryu": ["r", "y", "u"], "ryo": ["r", "y", "o"],
        "gya": ["gy", "a"], "gyu": ["gy", "u"], "gyo": ["gy", "o"],
        "ja": ["j", "a"], "ju": ["j", "u"], "jo": ["j", "o"],
        "bya": ["by", "a"], "byu": ["by", "u"], "byo": ["by", "o"],
        "pya": ["py", "a"], "pyu": ["py", "u"], "pyo": ["py", "o"],
        # Yōon — Nihon-shiki alternatives
        "sya": ["sh", "a"], "syu": ["sh", "u"], "syo": ["sh", "o"],
        "tya": ["ch", "a"], "tyu": ["ch", "u"], "tyo": ["ch", "o"],
        "zya": ["j", "a"], "zyu": ["j", "u"], "zyo": ["j", "o"],
        "dya": ["j", "a"], "dyu": ["j", "u"], "dyo": ["j", "o"],
        # Foreign loan extensions (ファ/ティ/ディ/ヴ etc.)
        "fa": ["f", "a"], "fi": ["f", "i"], "fe": ["f", "e"], "fo": ["f", "o"],
        "tsa": ["ts", "a"], "tsi": ["ts", "i"], "tse": ["ts", "e"], "tso": ["ts", "o"],
        "di": ["d", "i"], "du": ["d", "u"],
        "va": ["v", "a"], "vi": ["v", "i"], "vu": ["v", "u"], "ve": ["v", "e"], "vo": ["v", "o"],
        "ye": ["y", "e"],
        "je": ["j", "e"], "che": ["ch", "e"], "she": ["sh", "e"],
        "wi": ["w", "i"], "we": ["w", "e"],
        # L-for-R alternative (some UTAU voice banks use 'l' for ら-row)
        "la": ["r", "a"], "li": ["r", "i"], "lu": ["r", "u"], "le": ["r", "e"], "lo": ["r", "o"],
        "lya": ["r", "y", "a"], "lyu": ["r", "y", "u"], "lyo": ["r", "y", "o"],
        # Standalone consonants (UTAU articulation phonemes)
        "k": ["k"], "s": ["s"], "t": ["t"], "h": ["h"], "m": ["m"], "r": ["r"],
        "g": ["g"], "z": ["z"], "d": ["d"], "b": ["b"], "p": ["p"], "f": ["f"],
        "sh": ["sh"], "ch": ["ch"], "ts": ["ts"], "ky": ["ky"], "gy": ["gy"],
        "ny": ["ny"], "hy": ["hy"], "by": ["by"], "py": ["py"],
    }
    _ROMAJI_TO_OPENJTALK.update(_m)


def _japanese_lyric_to_ipa(lyric: str) -> list[str]:
    """Japanese lyric (hiragana/katakana/romaji) → IPA phoneme list."""
    _build_kana_table()
    _build_romaji_table()
    _ensure_s2h_path()
    from src.data.phoneme_v1 import OPENJTALK_TO_IPA

    # If all ASCII → try romaji lookup first
    if lyric.isascii():
        key = lyric.lower().strip()
        if key in _ROMAJI_TO_OPENJTALK:
            jtalk_phones = _ROMAJI_TO_OPENJTALK[key]
            ipa: list[str] = []
            for ph in jtalk_phones:
                if ph in OPENJTALK_TO_IPA:
                    ipa.extend(OPENJTALK_TO_IPA[ph])
                else:
                    ipa.append(ph)
            return ipa if ipa else ["a"]

    # Kana path: try multi-char first (きゃ before き)
    jtalk_phones: list[str] = []
    i = 0
    while i < len(lyric):
        if i + 1 < len(lyric) and lyric[i:i+2] in _KANA_TO_OPENJTALK:
            jtalk_phones.extend(_KANA_TO_OPENJTALK[lyric[i:i+2]])
            i += 2
        elif lyric[i] in _KANA_TO_OPENJTALK:
            jtalk_phones.extend(_KANA_TO_OPENJTALK[lyric[i]])
            i += 1
        else:
            jtalk_phones.append(lyric[i])
            i += 1

    ipa: list[str] = []
    for ph in jtalk_phones:
        if ph in OPENJTALK_TO_IPA:
            ipa.extend(OPENJTALK_TO_IPA[ph])
        else:
            ipa.append(ph)
    return ipa if ipa else ["a"]


def _detect_lyric_lang(lyric: str) -> str | None:
    """Auto-detect language from lyric characters. Returns 'zh', 'ja', or None."""
    for ch in lyric:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF or cp == 0x30FC:
            return "ja"
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
            return "zh"
    return None


_SILENT_LYRICS = frozenset({
    "-", "R", "r",
    "br", "BR", "breath", "Breath",
    "AP", "ap", "SP", "sp",
    "pau", "sil",
    "+", "+~", "++",
})


def _lyric_to_ipa(lyric: str, language: str) -> list[str]:
    """Convert a single note's lyric to IPA phonemes."""
    lyric = lyric.strip()
    if not lyric or lyric in _SILENT_LYRICS:
        return ["SP"]
    detected = _detect_lyric_lang(lyric)
    lang = detected or language

    # ASCII lyrics that match the romaji table should always use the
    # Japanese path, even when the global language is zh.  This avoids
    # sending e.g. "sh" through Chinese G2P (→ʂ) instead of Japanese (→ɕ).
    if lang == "zh" and lyric.isascii():
        _build_romaji_table()
        if lyric.lower() in _ROMAJI_TO_OPENJTALK:
            lang = "ja"

    if lang == "zh":
        return _chinese_lyric_to_ipa(lyric)
    elif lang == "ja":
        return _japanese_lyric_to_ipa(lyric)
    return ["a"]


def _midi_to_hz(midi: int) -> float:
    if midi <= 0:
        return 0.0
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _interpolate_pitch_points(points: list[dict], duration_ms: float, base_midi: int) -> list[float]:
    """Interpolate USTX PitchPoint[] → per-ms F0 in Hz.

    points: [{x: ms_offset, y: 0.1_semitones, shape: 'io'|'l'|...}, ...]
    Returns list of F0 values, one per ms, length = ceil(duration_ms).
    """
    n_ms = max(1, int(duration_ms))
    base_hz = _midi_to_hz(base_midi)
    if not points or base_midi <= 0:
        return [base_hz] * n_ms

    sorted_pts = sorted(points, key=lambda p: p["x"])
    result = [0.0] * n_ms

    for ms in range(n_ms):
        # Find surrounding points
        t = float(ms)
        y_val = 0.0

        if t <= sorted_pts[0]["x"]:
            y_val = sorted_pts[0]["y"]
        elif t >= sorted_pts[-1]["x"]:
            y_val = sorted_pts[-1]["y"]
        else:
            for i in range(len(sorted_pts) - 1):
                p0, p1 = sorted_pts[i], sorted_pts[i + 1]
                if p0["x"] <= t <= p1["x"]:
                    span = p1["x"] - p0["x"]
                    frac = (t - p0["x"]) / span if span > 0 else 0
                    shape = p0.get("shape", "io")
                    if shape == "l":
                        y_val = p0["y"] + (p1["y"] - p0["y"]) * frac
                    elif shape == "io":
                        frac = 0.5 - 0.5 * np.cos(frac * np.pi)
                        y_val = p0["y"] + (p1["y"] - p0["y"]) * frac
                    elif shape == "i":
                        frac = 1.0 - np.cos(frac * np.pi / 2)
                        y_val = p0["y"] + (p1["y"] - p0["y"]) * frac
                    elif shape == "o":
                        frac = np.sin(frac * np.pi / 2)
                        y_val = p0["y"] + (p1["y"] - p0["y"]) * frac
                    else:  # 'sp' = step
                        y_val = p0["y"]
                    break

        semitone_dev = y_val / 10.0  # y is in 0.1 semitones
        result[ms] = base_hz * (2.0 ** (semitone_dev / 12.0))

    return result


def _notes_to_f0_curve(
    notes: list[dict], bpm: float, total_frames: int,
) -> torch.Tensor | None:
    """Convert USTX notes with pitch data → frame-level F0 curve [T_frame] in Hz.

    Returns None if no note has custom pitch data (use Path A).
    """
    resolution = 480
    ticks_per_sec = (bpm * resolution) / 60.0

    has_pitch_data = any(
        n.get("pitch") and n["pitch"].get("data") and len(n["pitch"]["data"]) > 0
        for n in notes
    )
    if not has_pitch_data:
        return None

    f0 = np.zeros(total_frames, dtype=np.float32)

    for note in sorted(notes, key=lambda n: n["position"]):
        tone = note.get("tone", 0)
        if tone <= 0:
            continue
        pos_sec = note["position"] / ticks_per_sec
        dur_sec = note["duration"] / ticks_per_sec
        start_frame = round(pos_sec * 50)
        end_frame = round((pos_sec + dur_sec) * 50)

        pitch_data = note.get("pitch", {}).get("data", [])
        dur_ms = dur_sec * 1000

        if pitch_data:
            ms_curve = _interpolate_pitch_points(pitch_data, dur_ms, tone)
            for f in range(start_frame, min(end_frame, total_frames)):
                ms_idx = min(int((f - start_frame) / 50 * 1000), len(ms_curve) - 1)
                f0[f] = ms_curve[max(0, ms_idx)]
        else:
            hz = _midi_to_hz(tone)
            for f in range(start_frame, min(end_frame, total_frames)):
                f0[f] = hz

    return torch.from_numpy(f0)


def _is_consonant(ipa: str) -> bool:
    """Check if IPA symbol is a consonant (gets shorter duration)."""
    _vowels = {"a","e","i","o","u","ɯ","ə","ɤ","ɛ","æ","ɑ","ɔ","ʊ","ɪ","y","ɨ","ɚ","ɐ","ɒ"}
    return ipa not in _vowels and ipa not in ("SP", "AP", "br")


# ── Post mute ────────────────────────────────────────────────────────────

def _build_post_mute_envelope_from_rest(
    frame_is_rest: np.ndarray,
    n_audio_samples: int,
    sr: int = 48000,
    fade_ms: float = 5.0,
    min_rest_ms: float = 120.0,
) -> np.ndarray:
    """Build amplitude envelope from an explicit per-frame rest flag.

    frame_is_rest: bool array [T_frame] — True = intended silence.
    Uses the flag directly instead of inferring from pitch values.
    """
    spf = sr // 50
    envelope = np.ones(n_audio_samples, dtype=np.float32)
    fade_samples = int(fade_ms * sr / 1000)
    min_rest_frames = int(min_rest_ms * 50 / 1000)

    n_frames = len(frame_is_rest)
    i = 0
    while i < n_frames:
        if frame_is_rest[i]:
            start = i
            while i < n_frames and frame_is_rest[i]:
                i += 1
            length = i - start
            if length >= min_rest_frames:
                s_start = start * spf
                s_end = min(i * spf, n_audio_samples)
                envelope[s_start:s_end] = 0.0
                if fade_samples > 0 and s_start > 0:
                    f_begin = max(s_start - fade_samples, 0)
                    f_len = s_start - f_begin
                    envelope[f_begin:s_start] = np.linspace(
                        1.0, 0.0, f_len, dtype=np.float32,
                    )
        else:
            i += 1
    return envelope


# ── RVC decode ───────────────────────────────────────────────────────────

def _f0_to_coarse(f0_hz: torch.Tensor) -> torch.Tensor:
    f0_mel = 1127.0 * torch.log(1.0 + f0_hz / 700.0)
    f0_mel_min = 1127.0 * np.log(1.0 + 50.0 / 700.0)
    f0_mel_max = 1127.0 * np.log(1.0 + 1100.0 / 700.0)
    f0_mel_norm = (f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min)
    coarse = (f0_mel_norm * 254.0 + 1.0).long().clamp(1, 255)
    coarse[f0_hz < 50] = 0
    return coarse


@torch.no_grad()
def _rvc_decode(net_g, sid, hubert_feat: torch.Tensor, f0_hz: torch.Tensor, device: str):
    """50fps HuBERT+F0 → upsample 2x to 100fps → net_g.infer → audio."""
    T_min = min(hubert_feat.size(0), f0_hz.size(0))
    feat = hubert_feat[:T_min].unsqueeze(0).to(device).float()
    f0 = f0_hz[:T_min].to(device).float()

    feat_2x = F.interpolate(feat.permute(0, 2, 1), scale_factor=2, mode="nearest").permute(0, 2, 1)
    f0_2x = F.interpolate(f0.view(1, 1, -1), scale_factor=2, mode="linear", align_corners=False).view(-1)

    pitch_coarse = _f0_to_coarse(f0_2x).unsqueeze(0)
    nsff0 = f0_2x.unsqueeze(0)
    p_len = torch.tensor([feat_2x.size(1)], device=device).long()

    audio = net_g.infer(feat_2x, p_len, pitch_coarse, nsff0, sid)[0][0, 0]
    del feat, f0, feat_2x, f0_2x, pitch_coarse, nsff0, p_len
    return audio.float().cpu()


_RVC_CHUNK_FRAMES = 1500   # 30 seconds at 50fps
_RVC_OVERLAP_FRAMES = 50   # 1 second crossfade overlap


@torch.no_grad()
def _rvc_decode_chunked(net_g, sid, hubert_feat: torch.Tensor, f0_hz: torch.Tensor, device: str, sr: int):
    """Chunked RVC decode — splits long sequences to avoid CUDA OOM."""
    T = min(hubert_feat.size(0), f0_hz.size(0))
    if T <= _RVC_CHUNK_FRAMES:
        audio = _rvc_decode(net_g, sid, hubert_feat[:T], f0_hz[:T], device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio

    step = _RVC_CHUNK_FRAMES - _RVC_OVERLAP_FRAMES
    overlap_samples = int(_RVC_OVERLAP_FRAMES * sr / 50)
    n_chunks = (T - _RVC_OVERLAP_FRAMES + step - 1) // step

    logger.info(f"RVC chunked decode: {T} frames → {n_chunks} chunks "
                f"({_RVC_CHUNK_FRAMES}f each, {_RVC_OVERLAP_FRAMES}f overlap)")

    parts: list[torch.Tensor] = []
    start = 0
    while start < T:
        end = min(start + _RVC_CHUNK_FRAMES, T)
        chunk = _rvc_decode(net_g, sid, hubert_feat[start:end], f0_hz[start:end], device)
        parts.append(chunk)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if end >= T:
            break
        start += step

    if len(parts) == 1:
        return parts[0]

    result = parts[0]
    for i in range(1, len(parts)):
        ov = min(overlap_samples, len(result), len(parts[i]))
        if ov > 0:
            fade_out = torch.linspace(1.0, 0.0, ov)
            fade_in = torch.linspace(0.0, 1.0, ov)
            result[-ov:] = result[-ov:] * fade_out + parts[i][:ov] * fade_in
            result = torch.cat([result, parts[i][ov:]])
        else:
            result = torch.cat([result, parts[i]])

    return result


# ── Main wrapper class ───────────────────────────────────────────────────

class S2HWrapper:
    """Score2HuBERT + RVC synthesis wrapper. Lazy-loads models on first call."""

    def __init__(self):
        self._model = None
        self._vocab: dict | None = None
        self._rvc_model_name: str | None = None
        self._rvc_net_g = None
        self._rvc_sid = None
        self._rvc_sr: int = 48000
        self._rvc_index = None
        self._rvc_big_npy: np.ndarray | None = None
        self._device = get_device()

    def _ensure_model(self):
        if self._model is not None:
            return
        if not S2H_MODEL_PATH.exists():
            raise FileNotFoundError(f"Score2HuBERT model not found: {S2H_MODEL_PATH}")

        _ensure_s2h_path()
        from src.model.score2hubert import Score2HuBERT

        ckpt = torch.load(str(S2H_MODEL_PATH), map_location=self._device, weights_only=False)
        mc = ckpt.get("config", {}).get("model", {})
        self._model = Score2HuBERT(
            n_phones=mc.get("n_phones", 80),
            hidden_dim=mc.get("hidden_dim", 512),
            hubert_dim=mc.get("hubert_dim", 768),
            n_enc_layers=mc.get("n_enc_layers", 5),
            n_heads=mc.get("n_heads", 8),
            enc_kernel=mc.get("enc_kernel", 31),
            n_det_blocks=mc.get("n_det_blocks", 6),
            det_kernel=mc.get("det_kernel", 7),
            dropout=0.0,
        ).to(self._device)
        self._model.load_state_dict(ckpt["model"], strict=False)
        self._model.eval()

        step = ckpt.get("step", "?")
        logger.info(f"Score2HuBERT loaded: {S2H_MODEL_PATH.name} (step={step}, device={self._device})")

    def _ensure_vocab(self):
        if self._vocab is not None:
            return
        if S2H_VOCAB_PATH.exists():
            with open(S2H_VOCAB_PATH, encoding="utf-8") as f:
                self._vocab = yaml.safe_load(f)
        else:
            self._vocab = _BUILTIN_VOCAB
        logger.info(f"Vocab loaded: {len(self._vocab['phone2id'])} phones")

    def _ensure_rvc(self, model_name: str):
        if self._rvc_net_g is not None and self._rvc_model_name == model_name:
            return

        model_path = RVC_WEIGHTS_DIR / model_name
        if not model_path.exists():
            model_path = RVC_WEIGHTS_DIR / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"RVC model not found: {model_name} in {RVC_WEIGHTS_DIR}")

        rvc_infer = RVC_ROOT / "infer"
        rvc_lib = rvc_infer / "lib"
        for p in [str(rvc_lib), str(rvc_infer), str(RVC_ROOT)]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from infer_pack.models import SynthesizerTrnMs768NSFsid

        cpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        cfg = cpt.get("config", None)
        self._rvc_sr = cfg[-1] if cfg else 40000
        self._rvc_net_g = SynthesizerTrnMs768NSFsid(*cfg, is_half=False)
        self._rvc_net_g.load_state_dict(cpt.get("weight", {}), strict=False)
        self._rvc_net_g = self._rvc_net_g.to(self._device).eval()
        self._rvc_sid = torch.tensor([0], device=self._device)
        self._rvc_model_name = model_name
        logger.info(f"RVC loaded: {model_name} (sr={self._rvc_sr})")

        self._rvc_index = None
        self._rvc_big_npy = None
        basename = model_name.replace(".pth", "")
        index_path = self._find_index(basename)
        if index_path:
            try:
                import faiss
                self._rvc_index = faiss.read_index(str(index_path))
                self._rvc_big_npy = self._rvc_index.reconstruct_n(0, self._rvc_index.ntotal)
                logger.info(f"RVC index loaded: {index_path.name} ({self._rvc_index.ntotal} vectors)")
            except Exception as e:
                logger.warning(f"RVC index load failed: {e}")
                self._rvc_index = None
                self._rvc_big_npy = None
        else:
            logger.info(f"RVC index not found for '{basename}' (index retrieval disabled)")

    @staticmethod
    def _find_index(basename: str) -> Optional[Path]:
        if not RVC_INDICES_DIR.exists():
            return None
        import re
        clean = re.sub(r'_e\d+_s\d+$', '', basename).lower()
        for f in RVC_INDICES_DIR.iterdir():
            if f.suffix == ".index" and "trained" not in f.name and clean in f.name.lower():
                return f
        return None

    def _notes_to_tensors(
        self, notes: list[dict], bpm: float, language: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert USTX notes to (phonemes, pitch, durations, is_rest) tensors.

        is_rest: per-phoneme bool — True ONLY for gap-inserted SP (intended
        silence).  Note phonemes that fall back to SP_ID due to unknown IPA
        are NOT marked as rest, so the post-mute won't touch them.

        Frame positions are computed from ABSOLUTE tick positions to avoid
        cumulative rounding drift between score and audio timelines.
        """
        self._ensure_vocab()
        phone2id = self._vocab["phone2id"]
        resolution = 480

        ticks_per_sec = (bpm * resolution) / 60.0

        def tick_to_frame(tick: int) -> int:
            return round(tick / ticks_per_sec * 50)

        all_phonemes: list[int] = []
        all_pitch: list[int] = []
        all_durations: list[int] = []
        all_is_rest: list[bool] = []

        sorted_notes = sorted(notes, key=lambda n: n["position"])
        cursor_frame = 0  # current frame position (absolute)

        for note in sorted_notes:
            pos = note["position"]
            dur_ticks = note["duration"]
            tone = note.get("tone", 60)
            lyric = note.get("lyric", "a")

            note_start_frame = tick_to_frame(pos)
            note_end_frame = tick_to_frame(pos + dur_ticks)

            # Insert SP rest for gap before this note
            gap_frames = note_start_frame - cursor_frame
            if gap_frames > 0:
                all_phonemes.append(phone2id.get("SP", SP_ID))
                all_pitch.append(0)
                all_durations.append(gap_frames)
                all_is_rest.append(True)

            # G2P
            ipa_list = _lyric_to_ipa(lyric, language)
            note_frames = max(1, note_end_frame - note_start_frame)

            # Distribute frames among phonemes
            n_ph = len(ipa_list)
            if n_ph == 1:
                frame_alloc = [note_frames]
            else:
                cons_count = sum(1 for p in ipa_list if _is_consonant(p))
                vowel_count = n_ph - cons_count
                cons_frames_each = min(CONSONANT_FRAMES, max(1, note_frames // n_ph))
                remaining = note_frames - cons_count * cons_frames_each
                vowel_frames_each = max(1, remaining // max(vowel_count, 1))

                frame_alloc = []
                for p in ipa_list:
                    if _is_consonant(p):
                        frame_alloc.append(cons_frames_each)
                    else:
                        frame_alloc.append(vowel_frames_each)

                # Fix rounding — distribute remainder to last vowel
                diff = note_frames - sum(frame_alloc)
                if diff != 0:
                    for j in range(len(frame_alloc) - 1, -1, -1):
                        if not _is_consonant(ipa_list[j]):
                            frame_alloc[j] = max(1, frame_alloc[j] + diff)
                            break

            # _lyric_to_ipa returning ["SP"] means the lyric itself is a
            # silence marker (e.g. "R", "br") — treat as rest.
            lyric_is_sp = (ipa_list == ["SP"])

            for ph_ipa, frames in zip(ipa_list, frame_alloc):
                pid = phone2id.get(ph_ipa, phone2id.get("SP", SP_ID))
                midi = 0 if ph_ipa == "SP" else tone
                all_phonemes.append(pid)
                all_pitch.append(midi)
                all_durations.append(frames)
                all_is_rest.append(lyric_is_sp)

            cursor_frame = note_end_frame

        if not all_phonemes:
            all_phonemes = [SP_ID]
            all_pitch = [0]
            all_durations = [25]
            all_is_rest = [True]

        return (
            torch.tensor(all_phonemes, dtype=torch.long),
            torch.tensor(all_pitch, dtype=torch.long),
            torch.tensor(all_durations, dtype=torch.long),
            torch.tensor(all_is_rest, dtype=torch.bool),
        )

    def render(
        self,
        notes: list[dict],
        bpm: float,
        language: str = "zh",
        rvc_model: str = "",
        f0_up_key: int = 0,
        mute_strategy: str = "auto",
        index_rate: float = 0.75,
    ) -> tuple[np.ndarray, int]:
        """Full pipeline: notes → Score2HuBERT → RVC → post mute → (audio, sr).

        Uses Path B (user F0) if any note has pitch curve data, else Path A (model predicts F0).
        Returns numpy float32 array and sample rate.
        """
        self._ensure_model()
        if not rvc_model:
            raise ValueError("No RVC model selected. Please select a voice model before rendering.")
        self._ensure_rvc(rvc_model)

        phonemes, pitch, durations, is_rest = self._notes_to_tensors(notes, bpm, language)
        total_frames = int(durations.sum())

        user_f0 = _notes_to_f0_curve(notes, bpm, total_frames)
        path = "B (user F0)" if user_f0 is not None else "A (model F0)"
        idx_info = f"index_rate={index_rate}" if self._rvc_index is not None else "no index"
        rest_phones = int(is_rest.sum())
        logger.info(f"S2H render: {len(phonemes)} phones ({rest_phones} rest), {total_frames} frames, "
                     f"path={path}, lang={language}, rvc={rvc_model}, key={f0_up_key}, {idx_info}")

        with torch.no_grad():
            f0_target = user_f0.unsqueeze(0).to(self._device) if user_f0 is not None else None
            out = self._model(
                phonemes=phonemes.unsqueeze(0).to(self._device),
                pitch=pitch.unsqueeze(0).to(self._device),
                durations=durations.unsqueeze(0).to(self._device),
                f0_target=f0_target,
            )
        pred_hubert = out["hubert_pred"][0].float().cpu()
        pred_f0 = user_f0 if user_f0 is not None else out["f0_pred"][0].float().cpu()
        del out, f0_target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self._rvc_index is not None and self._rvc_big_npy is not None and index_rate > 0:
            npy = pred_hubert.numpy().astype("float32")
            score, ix = self._rvc_index.search(npy, k=8)
            weight = np.square(1.0 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            retrieved = np.sum(self._rvc_big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            pred_hubert = (
                torch.from_numpy(retrieved).float() * index_rate
                + (1 - index_rate) * pred_hubert
            )

        # Apply pitch shift
        if f0_up_key != 0:
            shift_ratio = 2.0 ** (f0_up_key / 12.0)
            pred_f0 = pred_f0 * shift_ratio

        # RVC decode (chunked to avoid CUDA OOM on long sequences)
        audio = _rvc_decode_chunked(
            self._rvc_net_g, self._rvc_sid, pred_hubert, pred_f0,
            self._device, self._rvc_sr,
        )
        sr = self._rvc_sr
        del pred_hubert, pred_f0

        # Post mute — use the explicit is_rest flag (not pitch values) to
        # determine which frames are intended silence.
        if mute_strategy in ("auto", "post", "both"):
            frame_is_rest = np.repeat(is_rest.numpy(), durations.numpy())
            expected_samples = len(frame_is_rest) * (sr // 50)
            if abs(len(audio) - expected_samples) > sr // 50:
                logger.warning(f"Post mute: audio/frame mismatch — "
                               f"audio={len(audio)}, expected={expected_samples} "
                               f"(drift={len(audio) - expected_samples} samples)")
            envelope = _build_post_mute_envelope_from_rest(
                frame_is_rest, len(audio), sr,
            )
            audio = audio * torch.from_numpy(envelope)

        # Normalize
        audio_np = audio.numpy()
        del audio
        peak = np.abs(audio_np).max()
        if peak > 0:
            audio_np = audio_np / peak * 0.95

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_np, sr

    def render_with_affine(
        self,
        notes: list[dict],
        bpm: float,
        language: str = "zh",
        rvc_model: str = "",
        f0_up_key: int = 0,
        mute_strategy: str = "auto",
        index_rate: float = 0.75,
        range_data: dict | None = None,
    ) -> tuple[np.ndarray, int]:
        """Render with affine pitch transform for RVC range extension.

        Maps out-of-range notes into the RVC model's comfort zone via affine
        transform, renders through S2H+RVC, then restores original pitches
        via WORLD F0 replacement.
        """
        if not range_data:
            return self.render(notes, bpm, language, rvc_model, f0_up_key,
                               mute_strategy, index_rate)

        comfort_low = range_data.get("comfort_low", 48)
        comfort_high = range_data.get("comfort_high", 72)
        range_low = range_data.get("range_low", comfort_low - 2)
        range_high = range_data.get("range_high", comfort_high + 2)

        voiced_notes = [n for n in notes if n.get("tone", 0) > 0]
        if not voiced_notes:
            return self.render(notes, bpm, language, rvc_model, f0_up_key,
                               mute_strategy, index_rate)

        target_midis = [n["tone"] + f0_up_key for n in voiced_notes]
        min_target = min(target_midis)
        max_target = max(target_midis)

        if min_target >= comfort_low and max_target <= comfort_high:
            logger.info("Affine: all notes in comfort zone — normal render")
            return self.render(notes, bpm, language, rvc_model, f0_up_key,
                               mute_strategy, index_rate)

        if min_target >= range_low and max_target <= range_high:
            logger.info("Affine: all notes in usable range — normal render (no WORLD needed)")
            return self.render(notes, bpm, language, rvc_model, f0_up_key,
                               mute_strategy, index_rate)

        a, b = _compute_affine(min_target, max_target, comfort_low, comfort_high)
        logger.info(f"Affine transform: [{min_target},{max_target}] → "
                     f"[{comfort_low},{comfort_high}], a={a:.3f}, b={b:.2f}")

        transformed_notes = []
        for note in notes:
            new_note = dict(note)
            if note.get("tone", 0) > 0:
                target = note["tone"] + f0_up_key
                mapped = int(round(a * target + b))
                mapped = max(comfort_low, min(comfort_high, mapped))
                new_note["tone"] = mapped
                if new_note.get("pitch") and new_note["pitch"].get("data"):
                    new_note["pitch"] = {"data": [
                        {"x": p["x"], "y": 0, "shape": p.get("shape", "io")}
                        for p in new_note["pitch"]["data"]
                    ]}
            transformed_notes.append(new_note)

        self._ensure_model()
        if not rvc_model:
            raise ValueError("No RVC model selected.")
        self._ensure_rvc(rvc_model)

        phonemes, pitch, durations, is_rest = self._notes_to_tensors(
            transformed_notes, bpm, language)
        total_frames = int(durations.sum())

        comfort_f0 = _build_f0_from_notes(transformed_notes, bpm, total_frames)
        target_f0 = _build_f0_from_notes(notes, bpm, total_frames,
                                          pitch_shift=f0_up_key)

        with torch.no_grad():
            f0_in = comfort_f0.unsqueeze(0).to(self._device)
            out = self._model(
                phonemes=phonemes.unsqueeze(0).to(self._device),
                pitch=pitch.unsqueeze(0).to(self._device),
                durations=durations.unsqueeze(0).to(self._device),
                f0_target=f0_in,
            )
        pred_hubert = out["hubert_pred"][0].float().cpu()
        del out, f0_in
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self._rvc_index is not None and self._rvc_big_npy is not None and index_rate > 0:
            npy = pred_hubert.numpy().astype("float32")
            score, ix = self._rvc_index.search(npy, k=8)
            weight = np.square(1.0 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            retrieved = np.sum(self._rvc_big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            pred_hubert = (
                torch.from_numpy(retrieved).float() * index_rate
                + (1 - index_rate) * pred_hubert
            )

        audio = _rvc_decode_chunked(
            self._rvc_net_g, self._rvc_sid, pred_hubert, comfort_f0,
            self._device, self._rvc_sr,
        )
        sr = self._rvc_sr
        del pred_hubert

        audio_np = audio.numpy()
        del audio

        import time as _time
        needs_shift = not np.isclose(a, 1.0) or not np.isclose(b, 0.0)
        if needs_shift:
            t0 = _time.perf_counter()
            if np.isclose(a, 1.0):
                shift_st = -b
                logger.info(f"Affine: uniform WORLD shift {shift_st:+.1f} semitones "
                             f"({len(audio_np)/sr:.1f}s audio)")
                audio_np = _world_uniform_shift(audio_np, sr, shift_st)
            else:
                target_np = target_f0.numpy()
                logger.info(f"Affine: per-frame WORLD F0 replacement "
                             f"({len(audio_np)/sr:.1f}s audio)")
                audio_np = _world_replace_f0(audio_np, sr, target_np)
            logger.info(f"Affine: WORLD done in {_time.perf_counter()-t0:.1f}s")

        if mute_strategy in ("auto", "post", "both"):
            frame_is_rest = np.repeat(is_rest.numpy(), durations.numpy())
            envelope = _build_post_mute_envelope_from_rest(
                frame_is_rest, len(audio_np), sr)
            audio_np = audio_np * envelope

        peak = np.abs(audio_np).max()
        if peak > 0:
            audio_np = audio_np / peak * 0.95

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_np, sr

    def detect_range(
        self,
        rvc_model: str,
        reference_pitch: int = 60,
        progress_cb=None,
    ) -> dict:
        """Test RVC model across MIDI range, return usable pitch bounds.

        Tests every semitone from MIDI 36 (C2) to 96 (C8).
        """
        import librosa

        self._ensure_model()
        self._ensure_rvc(rvc_model)

        test_bpm = 120.0
        test_lyric = "あ"
        test_dur = 480

        results = []
        midi_range = list(range(36, 97))
        total = len(midi_range)

        for i, midi in enumerate(midi_range):
            if progress_cb:
                progress_cb(i, total, midi)

            test_note = [{"position": 0, "duration": test_dur,
                          "tone": midi, "lyric": test_lyric}]
            try:
                use_index = 0.75 if self._rvc_index is not None else 0.0
                audio, sr = self.render(
                    notes=test_note, bpm=test_bpm, language="ja",
                    rvc_model=rvc_model, f0_up_key=0,
                    mute_strategy="raw", index_rate=use_index,
                )
                expected_hz = 440.0 * (2 ** ((midi - 69) / 12.0))
                f0_detected, voiced, _ = librosa.pyin(
                    audio.astype(np.float32), fmin=30, fmax=4200, sr=sr)
                voiced_mask = voiced & (f0_detected > 0)
                voiced_ratio = float(voiced_mask.sum() / max(len(voiced), 1))

                if voiced_ratio > 0.1:
                    median_f0 = float(np.nanmedian(f0_detected[voiced_mask]))
                    error_cents = 1200 * np.log2(median_f0 / expected_hz) if median_f0 > 0 else 9999
                else:
                    median_f0 = 0.0
                    error_cents = 9999.0

                err_abs = float(abs(error_cents))
                vr = float(voiced_ratio)
                results.append({
                    "midi": midi,
                    "expected_hz": round(expected_hz, 1),
                    "detected_hz": round(median_f0, 1),
                    "f0_error_cents": round(err_abs, 1),
                    "voiced_ratio": round(vr, 3),
                    "usable": bool(err_abs < 100 and vr > 0.5),
                    "comfort": bool(err_abs < 50 and vr > 0.8),
                })
            except Exception as e:
                logger.warning(f"Range test MIDI {midi} failed: {e}")
                results.append({
                    "midi": midi, "expected_hz": 0, "detected_hz": 0,
                    "f0_error_cents": 9999, "voiced_ratio": 0,
                    "usable": False, "comfort": False,
                })

        usable = [r for r in results if r["usable"]]
        comfort = [r for r in results if r["comfort"]]

        range_low = usable[0]["midi"] if usable else reference_pitch - 6
        range_high = usable[-1]["midi"] if usable else reference_pitch + 6
        comfort_low = comfort[0]["midi"] if comfort else range_low + 2
        comfort_high = comfort[-1]["midi"] if comfort else range_high - 2

        note_names = ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]

        def midi_name(m):
            return f"{note_names[m % 12]}{m // 12 - 1}"

        result = {
            "model_name": rvc_model.replace(".pth", ""),
            "reference_pitch": reference_pitch,
            "range_low": range_low,
            "range_high": range_high,
            "comfort_low": comfort_low,
            "comfort_high": comfort_high,
            "range_label": f"{midi_name(range_low)}-{midi_name(range_high)}",
            "comfort_label": f"{midi_name(comfort_low)}-{midi_name(comfort_high)}",
            "results": results,
        }

        logger.info(f"Range detection: {rvc_model} → usable {result['range_label']}, "
                     f"comfort {result['comfort_label']}")
        return result

    def render_to_file(self, output_path: str, **kwargs) -> str:
        """Render and save to WAV file. Returns the output path."""
        affine = kwargs.pop("affine_transform", False)
        range_data = kwargs.pop("range_data", None)
        if affine and range_data:
            audio, sr = self.render_with_affine(range_data=range_data, **kwargs)
        else:
            audio, sr = self.render(**kwargs)
        sf.write(output_path, audio, sr)
        logger.info(f"Saved: {output_path} ({len(audio)/sr:.2f}s)")
        return output_path


def _compute_affine(
    min_target: int, max_target: int,
    comfort_low: int, comfort_high: int,
) -> tuple[float, float]:
    """Compute minimal affine transform f(m)=a*m+b mapping [min,max]→comfort zone."""
    source_range = max_target - min_target
    comfort_range = comfort_high - comfort_low

    if source_range == 0:
        center = (comfort_low + comfort_high) / 2
        return 1.0, center - min_target

    if source_range <= comfort_range:
        a = 1.0
        if min_target < comfort_low:
            b = float(comfort_low - min_target)
        elif max_target > comfort_high:
            b = float(comfort_high - max_target)
        else:
            b = 0.0
    else:
        a = comfort_range / source_range
        b = comfort_low - a * min_target

    return a, b


def _build_f0_from_notes(
    notes: list[dict], bpm: float, total_frames: int,
    pitch_shift: int = 0,
) -> torch.Tensor:
    """Build per-frame F0 curve from notes (always returns a curve, never None)."""
    resolution = 480
    ticks_per_sec = (bpm * resolution) / 60.0
    f0 = np.zeros(total_frames, dtype=np.float32)

    for note in sorted(notes, key=lambda n: n["position"]):
        tone = note.get("tone", 0)
        if tone <= 0:
            continue
        pos_sec = note["position"] / ticks_per_sec
        dur_sec = note["duration"] / ticks_per_sec
        start_frame = round(pos_sec * 50)
        end_frame = round((pos_sec + dur_sec) * 50)
        effective_midi = tone + pitch_shift
        hz = _midi_to_hz(effective_midi)

        pitch_data = note.get("pitch", {}).get("data", [])
        if pitch_data:
            dur_ms = dur_sec * 1000
            ms_curve = _interpolate_pitch_points(pitch_data, dur_ms, effective_midi)
            for f in range(start_frame, min(end_frame, total_frames)):
                ms_idx = min(int((f - start_frame) / 50 * 1000), len(ms_curve) - 1)
                f0[f] = ms_curve[max(0, ms_idx)]
        else:
            for f in range(start_frame, min(end_frame, total_frames)):
                f0[f] = hz

    return torch.from_numpy(f0)


def _match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or trim WORLD output to match original sample count exactly."""
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)))


def _world_uniform_shift(
    audio: np.ndarray, sr: int, semitones: float,
) -> np.ndarray:
    """Uniform pitch shift via WORLD — uses dio (fast) instead of harvest."""
    import pyworld as pw

    original_len = len(audio)
    audio_f64 = audio.astype(np.float64)
    f0, t = pw.dio(audio_f64, sr)
    f0 = pw.stonemask(audio_f64, f0, t, sr)
    sp = pw.cheaptrick(audio_f64, f0, t, sr)
    ap = pw.d4c(audio_f64, f0, t, sr)

    f0_new = f0.copy()
    f0_new[f0_new > 0] *= 2.0 ** (semitones / 12.0)

    y = pw.synthesize(f0_new, sp, np.ascontiguousarray(ap), sr)
    y = _match_length(y, original_len)
    return y.astype(np.float32)


def _world_replace_f0(
    audio: np.ndarray, sr: int, target_f0_50fps: np.ndarray,
) -> np.ndarray:
    """Replace F0 in audio with target F0 curve using WORLD vocoder.

    Raw shift only — no formant modification, preserves voice character.
    """
    import pyworld as pw

    original_len = len(audio)
    audio_f64 = audio.astype(np.float64)
    f0, t = pw.dio(audio_f64, sr)
    f0 = pw.stonemask(audio_f64, f0, t, sr)
    sp = pw.cheaptrick(audio_f64, f0, t, sr)
    ap = pw.d4c(audio_f64, f0, t, sr)

    world_frames = len(f0)
    target_resampled = np.interp(
        np.linspace(0, 1, world_frames),
        np.linspace(0, 1, len(target_f0_50fps)),
        target_f0_50fps,
    )

    f0_new = f0.copy()
    voiced = target_resampled > 0
    f0_new[voiced] = target_resampled[voiced]

    y = pw.synthesize(f0_new, sp, np.ascontiguousarray(ap), sr)
    y = _match_length(y, original_len)
    return y.astype(np.float32)


_wrapper: Optional[S2HWrapper] = None


def get_s2h_wrapper() -> S2HWrapper:
    global _wrapper
    if _wrapper is None:
        _wrapper = S2HWrapper()
    return _wrapper
