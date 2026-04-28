"""
IPA-based phoneme vocabulary and G2P mapping for Score2HuBERT.

Design rationale:
    Using International Phonetic Alphabet (IPA) symbols as the universal phoneme
    representation eliminates cross-lingual transfer problems entirely. Each
    language's phonemes are mapped to IPA symbols at preprocessing time, so:

    - Chinese 'a' (in 妈) and Japanese 'a' (in あ) become the SAME token /a/.
    - Adding new languages requires only a G2P mapping table, no model changes.
    - Encoder capacity needed for cross-lingual learning is freed up.
    - Training data utilization is maximized (every /a/ counts toward the same
      embedding regardless of source language).

    For SVS specifically, IPA loses very little:
    - Tones (Chinese): irrelevant in singing — F0 carries the melody, not lexical tone
    - Pitch accent (Japanese): same reason
    - Allophone variation: handled by F0/Variance/co-articulation in the model
    - Aspiration (送气): preserved via /pʰ/ vs /p/ etc.

Phoneme decomposition:
    Pinyin "finals" like 'iang' are composite (/j/+/a/+/ŋ/). When converting to
    IPA, we DECOMPOSE them into atomic segments and SPLIT the original duration
    proportionally. This way the IPA token for /a/ is shared across:
        - Chinese 妈 (m + a)
        - Chinese 江 (j + iang → j + j + a + ŋ)  ← shares the /a/
        - Japanese あ (a)                          ← shares the /a/

Vocabulary size: ~70 IPA symbols + 7 special tokens.
"""

import yaml
from pathlib import Path
from typing import List, Optional, Dict, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Special tokens (always come first, with fixed IDs)
# ──────────────────────────────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    "<pad>",   # 0: padding
    "<sos>",   # 1: start of sequence
    "<eos>",   # 2: end of sequence
    "SP",      # 3: silence/pause (general)
    "AP",      # 4: aspiration / short breath
    "br",      # 5: breath (long, between phrases)
    "<mask>",  # 6: mask token for CFG / future MLM
]


# ──────────────────────────────────────────────────────────────────────────────
# IPA phoneme inventory (~60 symbols, covers ja/zh + most languages)
# ──────────────────────────────────────────────────────────────────────────────

IPA_VOWELS = [
    # Front
    "i",   # close front unrounded (ja い, zh i in 西)
    "y",   # close front rounded (zh ü)
    "ɪ",   # near-close near-front (English bit)
    "e",   # close-mid front (ja え, zh -ie e)
    "ɛ",   # open-mid front (zh ian -e-)
    "æ",   # near-open front (English cat)

    # Central
    "ɨ",   # close central (zh i in 资/吃, allophone)
    "ə",   # mid central schwa (zh -en, -eng)
    "ɚ",   # rhotic schwa (zh er)
    "ɐ",   # near-open central
    "a",   # open front/central (ja あ, zh a)

    # Back
    "ɑ",   # open back (English father)
    "ɒ",   # open back rounded
    "ɔ",   # open-mid back rounded (English thought)
    "o",   # close-mid back rounded (ja お, zh o)
    "ɤ",   # close-mid back unrounded (zh e in 哥)
    "ʊ",   # near-close back rounded
    "u",   # close back rounded (ja う, zh u)
    "ɯ",   # close back unrounded (ja う allophone)
]

IPA_STOPS = [
    "p",  "pʰ",  "b",   # bilabial
    "t",  "tʰ",  "d",   # alveolar
    "k",  "kʰ",  "g",   # velar
    "ʔ",                # glottal stop (ja っ)
]

IPA_AFFRICATES = [
    "ts",  "tsʰ",  "dz",   # alveolar (zh z/c)
    "tɕ",  "tɕʰ",  "dʑ",   # alveolo-palatal (zh j/q, ja ち)
    "ʈʂ",  "ʈʂʰ",  "ɖʐ",   # retroflex (zh zh/ch)
    "tʃ",  "dʒ",            # postalveolar (English ch/j)
]

IPA_FRICATIVES = [
    "f",   "v",            # labiodental
    "s",   "z",            # alveolar
    "ɕ",   "ʑ",            # alveolo-palatal (zh x, ja し)
    "ʂ",   "ʐ",            # retroflex (zh sh/r)
    "ʃ",   "ʒ",            # postalveolar
    "ç",                   # palatal (ja ひ)
    "ɸ",                   # bilabial (ja ふ)
    "x",   "ɣ",            # velar (zh h)
    "h",                   # glottal
]

IPA_NASALS = [
    "m",   # bilabial
    "n",   # alveolar
    "ɲ",   # palatal (ja にゃ)
    "ŋ",   # velar (zh -ng)
    "ɴ",   # uvular (ja ん word-final)
]

IPA_LIQUIDS_GLIDES = [
    "l",   # alveolar lateral
    "ɾ",   # alveolar tap (ja ら)
    "ɹ",   # alveolar approximant
    "j",   # palatal glide (y)
    "w",   # labiovelar glide
    "ɥ",   # labial-palatal glide
]

# Phase 2 additions: dental fricatives needed for English (think /θ/, this /ð/).
# IMPORTANT: These MUST be appended AFTER all original IPA categories so the
# token IDs of the original 66 IPA tokens (used by Plan C checkpoint) are preserved.
# These take the slots previously occupied by <unused_73>, <unused_74>.
IPA_DENTAL_FRICATIVES = [
    "θ",   # voiceless dental fricative (English think)
    "ð",   # voiced dental fricative (English this)
]

# Master list (order matters for ID stability — DO NOT reorder existing categories)
IPA_PHONEMES: List[str] = (
    IPA_VOWELS
    + IPA_STOPS
    + IPA_AFFRICATES
    + IPA_FRICATIVES
    + IPA_NASALS
    + IPA_LIQUIDS_GLIDES
    + IPA_DENTAL_FRICATIVES   # appended at end → IDs 73, 74 (was unused_73/74 in Plan C)
)


# ──────────────────────────────────────────────────────────────────────────────
# Pinyin → IPA mapping (Chinese, used by ACE-Opencpop / Opencpop)
# ──────────────────────────────────────────────────────────────────────────────

# Initials are 1-to-1 with IPA consonants
PINYIN_INITIAL_TO_IPA: Dict[str, List[str]] = {
    "b": ["p"],   "p": ["pʰ"],  "m": ["m"],   "f": ["f"],
    "d": ["t"],   "t": ["tʰ"],  "n": ["n"],   "l": ["l"],
    "g": ["k"],   "k": ["kʰ"],  "h": ["x"],
    "j": ["tɕ"],  "q": ["tɕʰ"], "x": ["ɕ"],
    "zh": ["ʈʂ"], "ch": ["ʈʂʰ"],"sh": ["ʂ"],  "r": ["ʐ"],
    "z": ["ts"],  "c": ["tsʰ"], "s": ["s"],
    "y": ["j"],   "w": ["w"],
}

# Finals are decomposed into atomic IPA segments
# Standard pinyin → IPA mapping per Chinese phonology references
PINYIN_FINAL_TO_IPA: Dict[str, List[str]] = {
    # Simple vowels
    "a":   ["a"],
    "o":   ["o"],
    "e":   ["ɤ"],
    "ê":   ["ɛ"],
    "i":   ["i"],
    "u":   ["u"],
    "v":   ["y"],   "ü": ["y"],

    # Diphthongs
    "ai":  ["a", "i"],
    "ei":  ["e", "i"],
    "ao":  ["a", "u"],
    "ou":  ["o", "u"],

    # -n
    "an":  ["a", "n"],
    "en":  ["ə", "n"],

    # -ng
    "ang": ["a", "ŋ"],
    "eng": ["ə", "ŋ"],
    "ong": ["u", "ŋ"],

    # er
    "er":  ["ɚ"],

    # i- compounds
    "ia":   ["j", "a"],
    "ie":   ["j", "e"],
    "iao":  ["j", "a", "u"],
    "iou":  ["j", "o", "u"],
    "iu":   ["j", "o", "u"],   # alias
    "ian":  ["j", "ɛ", "n"],
    "in":   ["i", "n"],
    "iang": ["j", "a", "ŋ"],
    "ing":  ["i", "ŋ"],
    "iong": ["j", "u", "ŋ"],

    # u- compounds
    "ua":   ["w", "a"],
    "uo":   ["w", "o"],
    "uai":  ["w", "a", "i"],
    "uei":  ["w", "e", "i"],
    "ui":   ["w", "e", "i"],   # alias
    "uan":  ["w", "a", "n"],
    "uen":  ["w", "ə", "n"],
    "un":   ["w", "ə", "n"],   # alias
    "uang": ["w", "a", "ŋ"],
    "ueng": ["w", "ə", "ŋ"],

    # ü- compounds
    "ve":   ["y", "e"],   "üe": ["y", "e"],
    "van":  ["y", "ɛ", "n"], "üan": ["y", "ɛ", "n"],
    "vn":   ["y", "n"],   "ün": ["y", "n"],
}


# ──────────────────────────────────────────────────────────────────────────────
# OpenJTalk → IPA mapping (Japanese, used by Kiritan / Ofuton / pyopenjtalk)
# ──────────────────────────────────────────────────────────────────────────────

# Most Japanese phonemes are atomic; the openjtalk phoneme set is small
OPENJTALK_TO_IPA: Dict[str, List[str]] = {
    # Vowels
    "a": ["a"],   "i": ["i"],   "u": ["ɯ"],   "e": ["e"],   "o": ["o"],
    "A": ["a"],   "I": ["i"],   "U": ["ɯ"],   "E": ["e"],   "O": ["o"],  # devoiced/long uppercase variants

    # Stops
    "k": ["k"],   "g": ["g"],
    "t": ["t"],   "d": ["d"],
    "p": ["p"],   "b": ["b"],

    # Fricatives
    "s": ["s"],   "z": ["z"],
    "h": ["h"],
    "f": ["ɸ"],                 # Japanese ふ
    "v": ["v"],                 # foreign loans

    # Affricates
    "ts": ["ts"],
    "ch": ["tɕ"],               # ち, ちゃ
    "sh": ["ɕ"],                # し, しゃ
    "j":  ["dʑ"],               # じ, じゃ

    # Nasals
    "m": ["m"],   "n": ["n"],
    "N": ["ɴ"],                 # word-final ん

    # Liquids/glides
    "r": ["ɾ"],                 # ら、り、る、れ、ろ
    "y": ["j"],                 # や、ゆ、よ
    "w": ["w"],                 # わ

    # Palatalized consonants (拗音) — decompose into consonant + /j/
    "ky": ["k", "j"],   "gy": ["g", "j"],
    "ty": ["t", "j"],   "dy": ["d", "j"],
    "py": ["p", "j"],   "by": ["b", "j"],
    "ny": ["ɲ"],                # にゃ - palatal nasal
    "hy": ["ç"],                # ひゃ - palatal fricative
    "my": ["m", "j"],
    "ry": ["ɾ", "j"],
    "fy": ["ɸ", "j"],

    # Special
    "cl": ["ʔ"],                # っ (geminate / glottal stop marker)
    "q":  ["ʔ"],                # alias

    # Silence/pause variants
    "pau": ["SP"],  "sil": ["SP"],  "br": ["br"],  "sp": ["SP"],
}


# ──────────────────────────────────────────────────────────────────────────────
# PhonemeVocab class
# ──────────────────────────────────────────────────────────────────────────────


class PhonemeVocab:
    """Universal IPA-based phoneme vocabulary."""

    def __init__(self, vocab_file: Optional[str] = None):
        self.phone2id: Dict[str, int] = {}
        self.id2phone: Dict[int, str] = {}
        self.lang2id: Dict[str, int] = {}
        self._next_id = 0

        if vocab_file:
            self.load(vocab_file)
        else:
            self._build_default()

    def _build_default(self):
        """Build default IPA vocabulary."""
        # 1. Special tokens (fixed IDs 0-6)
        for tok in SPECIAL_TOKENS:
            self._add(tok)

        # 2. IPA phonemes
        for ipa in IPA_PHONEMES:
            self._add(ipa)

        # 3. Language IDs
        for lang in ["ja", "zh", "en", "ko"]:
            self.lang2id[lang] = len(self.lang2id)

        # Pad to multiple of 8 for embedding efficiency
        while self._next_id % 8 != 0:
            self._add(f"<unused_{self._next_id}>")

    def _add(self, phone: str):
        if phone not in self.phone2id:
            self.phone2id[phone] = self._next_id
            self.id2phone[self._next_id] = phone
            self._next_id += 1

    def encode(self, phonemes: List[str], language: str = "ja") -> List[int]:
        """Convert IPA phoneme strings to IDs.

        NOTE: input is expected to already be IPA symbols (use convert_to_ipa
        first if you have language-specific phonemes like pinyin or openjtalk).
        """
        ids = []
        for p in phonemes:
            if p in self.phone2id:
                ids.append(self.phone2id[p])
            elif p.upper() in self.phone2id:
                ids.append(self.phone2id[p.upper()])
            else:
                # Unknown phoneme → SP (silence)
                ids.append(self.phone2id.get("SP", 0))
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id2phone.get(i, "<unk>") for i in ids]

    def get_language_id(self, lang: str) -> int:
        return self.lang2id.get(lang, 0)

    @property
    def size(self) -> int:
        return len(self.phone2id)

    @property
    def pad_id(self) -> int:
        return self.phone2id["<pad>"]

    @property
    def mask_id(self) -> int:
        return self.phone2id["<mask>"]

    def save(self, path: str):
        data = {"phone2id": self.phone2id, "lang2id": self.lang2id}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.phone2id = data["phone2id"]
        self.id2phone = {v: k for k, v in self.phone2id.items()}
        self.lang2id = data.get("lang2id", {})
        self._next_id = max(self.phone2id.values()) + 1


# ──────────────────────────────────────────────────────────────────────────────
# Conversion utilities — language-specific phonemes → IPA segments
# ──────────────────────────────────────────────────────────────────────────────


def convert_phonemes_to_ipa(
    phonemes: List[str],
    durations: List[float],
    language: str,
) -> Tuple[List[str], List[float]]:
    """Convert language-specific phonemes to IPA segments with proportional duration split.

    Args:
        phonemes: list of language-specific phoneme strings (e.g., pinyin, openjtalk)
        durations: parallel list of durations (any unit — seconds, frames, etc.)
        language: 'zh' for pinyin, 'ja' for openjtalk

    Returns:
        (ipa_phonemes, ipa_durations) — both possibly LONGER than input due to decomposition.

    Example:
        Input:  (['g', 'iang'],  [0.05, 0.30])     # zh
        Output: (['k', 'j', 'a', 'ŋ'], [0.05, 0.10, 0.10, 0.10])
    """
    if language == "zh":
        return _convert_pinyin_to_ipa(phonemes, durations)
    elif language == "ja":
        return _convert_openjtalk_to_ipa(phonemes, durations)
    elif language == "en":
        return _convert_arpabet_to_ipa(phonemes, durations)
    else:
        # Unknown language: pass through (assume already IPA)
        return phonemes, durations


def _convert_pinyin_to_ipa(
    phonemes: List[str],
    durations: List[float],
) -> Tuple[List[str], List[float]]:
    """Convert pinyin initials/finals to IPA segments.

    Pinyin initials map 1-to-1 to IPA consonants.
    Pinyin finals decompose into 1-3 IPA vowel/coda segments.
    Duration of a composite final is split equally across its IPA segments.
    """
    ipa_out: List[str] = []
    dur_out: List[float] = []

    for ph, dur in zip(phonemes, durations):
        # Strip tone digits if any (e.g., "a1" → "a")
        ph_clean = ph.rstrip("0123456789")

        # Special tokens pass through
        if ph_clean in ("SP", "AP", "br", "sil", "pau", "sp", ""):
            ipa_out.append("SP" if ph_clean else "SP")
            dur_out.append(dur)
            continue

        # Check initials first (longer keys first to avoid 'z' matching 'zh')
        if ph_clean in PINYIN_INITIAL_TO_IPA:
            segments = PINYIN_INITIAL_TO_IPA[ph_clean]
        elif ph_clean in PINYIN_FINAL_TO_IPA:
            segments = PINYIN_FINAL_TO_IPA[ph_clean]
        else:
            # Unknown — emit as silence to avoid breaking alignment
            segments = ["SP"]

        # Split duration equally across segments
        n = len(segments)
        per_seg = dur / n if n > 0 else dur
        for seg in segments:
            ipa_out.append(seg)
            dur_out.append(per_seg)

    return ipa_out, dur_out


def _convert_openjtalk_to_ipa(
    phonemes: List[str],
    durations: List[float],
) -> Tuple[List[str], List[float]]:
    """Convert OpenJTalk Japanese phonemes to IPA segments."""
    ipa_out: List[str] = []
    dur_out: List[float] = []

    for ph, dur in zip(phonemes, durations):
        ph_clean = ph.strip()

        # Special tokens
        if ph_clean in ("SP", "AP", "br", "sil", "pau", "sp", ""):
            ipa_out.append("SP" if ph_clean else "SP")
            dur_out.append(dur)
            continue

        if ph_clean in OPENJTALK_TO_IPA:
            segments = OPENJTALK_TO_IPA[ph_clean]
        else:
            segments = ["SP"]

        n = len(segments)
        per_seg = dur / n if n > 0 else dur
        for seg in segments:
            ipa_out.append(seg)
            dur_out.append(per_seg)

    return ipa_out, dur_out


# Legacy functions kept for backward compatibility (now route through IPA conversion)

def opencpop_to_phonemes(pinyin: str) -> List[str]:
    """Parse Opencpop pinyin string and convert to IPA segments.

    NOTE: Returns IPA tokens, not raw pinyin. Use this when you need a flat list
    without duration info; for proper preprocessing, use convert_phonemes_to_ipa().
    """
    parts = pinyin.strip().split()
    # Use unit durations and discard them
    ipa, _ = _convert_pinyin_to_ipa(parts, [1.0] * len(parts))
    return ipa


def kiritan_to_phonemes(label: str) -> List[str]:
    """Parse a HTS label line and convert center phoneme to IPA."""
    if "-" in label and "+" in label:
        phone = label.split("-")[1].split("+")[0]
    else:
        phone = label.strip()

    if phone in ("pau", "sil", "br", "sp"):
        return ["SP"]

    ipa, _ = _convert_openjtalk_to_ipa([phone], [1.0])
    return ipa


# ──────────────────────────────────────────────────────────────────────────────
# ARPABET → IPA mapping (English, used by GTSinger English subset)
# ──────────────────────────────────────────────────────────────────────────────
#
# ARPABET is the standard English phonetic alphabet used by CMUdict and most
# English G2P tools. Vowels carry stress markers (0=unstressed, 1=primary,
# 2=secondary). For SVS we strip stress because F0 carries pitch information.
#
# Diphthongs (AW=aʊ, AY=aɪ, EY=eɪ, OW=oʊ, OY=ɔɪ) are decomposed into TWO IPA
# segments with proportional duration split, mirroring how Pinyin diphthongs
# (ai, ei, ao, ou) are handled.
#
# AH (cut/about): mapped to /ə/ for both stressed and unstressed forms.
# Technically AH1 should be /ʌ/ but the difference is negligible for singing
# (F0 dominates) and avoids inflating the IPA inventory with another vowel.

ARPABET_TO_IPA: Dict[str, List[str]] = {
    # Monophthong vowels
    "AA": ["ɑ"],   # father, palm
    "AE": ["æ"],   # cat, bat
    "AH": ["ə"],   # cut/about (we use schwa for all stress levels)
    "AO": ["ɔ"],   # thought, dog
    "EH": ["ɛ"],   # red, bed
    "ER": ["ɚ"],   # her, bird (rhotic schwa)
    "IH": ["ɪ"],   # bit, sit
    "IY": ["i"],   # see, fleece
    "UH": ["ʊ"],   # book, foot
    "UW": ["u"],   # food, boot

    # Diphthongs (decomposed into two IPA segments)
    "AW": ["a", "ʊ"],   # how, mouth
    "AY": ["a", "ɪ"],   # high, price
    "EY": ["e", "ɪ"],   # say, face
    "OW": ["o", "ʊ"],   # show, goat
    "OY": ["ɔ", "ɪ"],   # toy, choice

    # Consonants (no stress marker)
    "B":  ["b"],
    "CH": ["tʃ"],   # church
    "D":  ["d"],
    "DH": ["ð"],    # this — uses NEW dental fricative token
    "F":  ["f"],
    "G":  ["g"],
    "HH": ["h"],
    "JH": ["dʒ"],   # judge
    "K":  ["k"],
    "L":  ["l"],
    "M":  ["m"],
    "N":  ["n"],
    "NG": ["ŋ"],    # sing
    "P":  ["p"],
    "R":  ["ɹ"],    # red (English approximant, not retroflex)
    "S":  ["s"],
    "SH": ["ʃ"],    # shoe
    "T":  ["t"],
    "TH": ["θ"],    # think — uses NEW dental fricative token
    "V":  ["v"],
    "W":  ["w"],
    "Y":  ["j"],    # yes
    "Z":  ["z"],
    "ZH": ["ʒ"],    # measure
}


def _convert_arpabet_to_ipa(
    phonemes: List[str],
    durations: List[float],
) -> Tuple[List[str], List[float]]:
    """Convert ARPABET phonemes (with stress markers) to IPA segments.

    Strips stress markers (0/1/2) before lookup. Decomposes diphthongs into
    two IPA segments with proportional duration split.
    """
    ipa_out: List[str] = []
    dur_out: List[float] = []

    for ph, dur in zip(phonemes, durations):
        ph_clean = ph.strip()

        # Special tokens pass through
        if ph_clean in ("SP", "AP", "br", "sil", "pau", "sp", ""):
            ipa_out.append("SP" if ph_clean else "SP")
            dur_out.append(dur)
            continue

        # Strip stress digit (0/1/2) from end of vowel symbols
        ph_nostress = ph_clean.rstrip("012")

        if ph_nostress in ARPABET_TO_IPA:
            segments = ARPABET_TO_IPA[ph_nostress]
        else:
            # Unknown ARPABET symbol — emit silence to avoid breaking alignment
            segments = ["SP"]

        n = len(segments)
        per_seg = dur / n if n > 0 else dur
        for seg in segments:
            ipa_out.append(seg)
            dur_out.append(per_seg)

    return ipa_out, dur_out


# ──────────────────────────────────────────────────────────────────────────────
# MFA-Japanese → IPA normalization (used by GTSinger Japanese subset)
# ──────────────────────────────────────────────────────────────────────────────
#
# GTSinger Japanese uses the Montreal Forced Aligner standard, NOT OpenJTalk.
# MFA Japanese uses IPA-style symbols directly with several conventions:
#   - Length mark `ː` for long vowels (aː, iː, eː, oː, ɨː)
#   - Combining ring below `̥` (U+0325) for devoiced vowels (i̥, ɨ̥, ɯ̥)
#   - Palatalization superscript `ʲ` (U+02B2) on consonants (mʲ, ɾʲ, bʲ)
#   - Geminate via length mark on consonants (pː, tː, nː, ɴː)
#   - IPA latin g `ɡ` (U+0261) instead of ASCII g (U+0067)
#   - Palatal stops `c`/`ɟ` (rare, mostly foreign loans)
#   - Nasalized velar approximant `ɰ̃` for non-final n
#
# Strategy: normalize all of these to existing IPA inventory tokens. Length and
# devoicing are dropped because they don't carry singing-relevant information
# (F0 + duration encode them). Palatalized consonants decompose to consonant + j.
# Geminates encode as [ʔ, base] where ʔ is the gemination/glottal-stop marker.

# Direct one-to-one substitutions (no decomposition)
_MFA_JA_DIRECT: Dict[str, str] = {
    "ɡ":  "g",    # IPA latin g → ASCII g (unicode normalization)
    "c":  "tɕ",   # voiceless palatal stop → alveolo-palatal affricate
    "ɟ":  "dʑ",   # voiced palatal stop → alveolo-palatal affricate
    "ɰ̃":  "ɴ",   # nasalized velar approximant → uvular nasal (Japanese N)
}

# Multi-segment substitutions (palatalized consonants → consonant + j glide)
_MFA_JA_PALATAL: Dict[str, List[str]] = {
    "mʲ": ["m", "j"],
    "bʲ": ["b", "j"],
    "pʲ": ["p", "j"],
    "kʲ": ["k", "j"],
    "gʲ": ["g", "j"],
    "tʲ": ["t", "j"],
    "dʲ": ["d", "j"],
    "nʲ": ["ɲ"],         # palatalized n collapses to palatal nasal
    "ɾʲ": ["ɾ", "j"],
    "hʲ": ["ç"],         # palatalized h → palatal fricative
    "fʲ": ["ɸ", "j"],
    "sʲ": ["ɕ"],
    "zʲ": ["ʑ"],
}

# Geminated consonants → [ʔ, base] (ʔ marks gemination, consistent with openjtalk `cl`)
_MFA_JA_GEMINATE: Dict[str, List[str]] = {
    "pː": ["ʔ", "p"],
    "tː": ["ʔ", "t"],
    "kː": ["ʔ", "k"],
    "nː": ["ʔ", "n"],
    "mː": ["ʔ", "m"],
    "sː": ["ʔ", "s"],
    "ɴː": ["ʔ", "ɴ"],
}


def _convert_mfa_japanese_to_ipa(
    phonemes: List[str],
    durations: List[float],
) -> Tuple[List[str], List[float]]:
    """Convert MFA-Japanese phonemes (as used by GTSinger) to existing-IPA-inventory tokens.

    Handles: length marks, devoicing diacritics, palatalization, geminates,
    unicode g normalization. All output tokens come from the existing
    IPA_PHONEMES inventory — no new tokens introduced.
    """
    ipa_out: List[str] = []
    dur_out: List[float] = []

    for ph, dur in zip(phonemes, durations):
        ph_clean = ph.strip()

        # Special tokens
        if ph_clean in ("SP", "AP", "br", "sil", "pau", "sp", ""):
            ipa_out.append("SP" if ph_clean else "SP")
            dur_out.append(dur)
            continue

        # 1. Strip combining devoicing ring (U+0325). Devoiced vowels are
        #    perceptually similar enough to bare vowels for SVS purposes.
        ph_clean = ph_clean.replace("\u0325", "")

        # 2. Multi-char palatalized consonants — try BEFORE length stripping so
        #    e.g. mʲ doesn't get partially matched as m + ʲ.
        if ph_clean in _MFA_JA_PALATAL:
            segments = _MFA_JA_PALATAL[ph_clean]
            n = len(segments)
            per_seg = dur / n if n > 0 else dur
            for seg in segments:
                ipa_out.append(seg)
                dur_out.append(per_seg)
            continue

        # 3. Geminated consonants — match BEFORE generic length stripping
        if ph_clean in _MFA_JA_GEMINATE:
            segments = _MFA_JA_GEMINATE[ph_clean]
            n = len(segments)
            per_seg = dur / n if n > 0 else dur
            for seg in segments:
                ipa_out.append(seg)
                dur_out.append(per_seg)
            continue

        # 4. Strip remaining length marks (long vowels: aː → a, iː → i, etc.)
        ph_clean = ph_clean.replace("ː", "")

        # 5. Direct unicode/category substitutions (ɡ→g, c→tɕ, ɟ→dʑ, ɰ̃→ɴ)
        if ph_clean in _MFA_JA_DIRECT:
            ph_clean = _MFA_JA_DIRECT[ph_clean]

        # 6. Single phoneme — emit as-is (vocab.encode will handle SP fallback if missing)
        ipa_out.append(ph_clean)
        dur_out.append(dur)

    return ipa_out, dur_out


# ──────────────────────────────────────────────────────────────────────────────
# GTSinger phoneme conversion wrapper
# ──────────────────────────────────────────────────────────────────────────────
#
# GTSinger represents phonemes in two ways:
#   - per-language metadata.json: suffixed with `_zh` / `_ja` / `_en`
#   - per-utterance .json (word-level): NO suffix, raw phoneme symbols
#
# This wrapper accepts BOTH and routes to the appropriate language converter
# after stripping the suffix. It also normalizes GTSinger's special markers
# (`<SP>`, `<AP>`) to our internal SP/AP.

_GTSINGER_LANG_SUFFIX = {"_zh", "_ja", "_en", "_ko"}


def convert_gtsinger_phonemes_to_ipa(
    phonemes: List[str],
    durations: List[float],
    language: str,
) -> Tuple[List[str], List[float]]:
    """Convert GTSinger phonemes to IPA segments.

    Handles both suffixed (`l_zh`, `HH_en`, `ɾʲ_ja`) and unsuffixed forms.
    Maps GTSinger's `<SP>`/`<AP>` brackets to internal SP/AP tokens.

    Args:
        phonemes: list of phoneme strings as found in GTSinger metadata
        durations: parallel list of durations (any unit)
        language: 'zh', 'ja', or 'en'

    Returns:
        (ipa_phonemes, ipa_durations) — both possibly LONGER than input due
        to diphthong/palatalization decomposition.
    """
    # Strip GTSinger's suffix if present, normalize bracket markers
    cleaned: List[str] = []
    for ph in phonemes:
        ph = ph.strip()
        # Strip language suffix if any (last 3 chars: `_xx`)
        if len(ph) > 3 and ph[-3:] in _GTSINGER_LANG_SUFFIX:
            ph = ph[:-3]
        # Normalize bracket markers to internal form
        if ph == "<SP>":
            ph = "SP"
        elif ph == "<AP>":
            ph = "AP"
        cleaned.append(ph)

    # Dispatch to language-specific converter
    if language in ("zh", "Chinese"):
        return _convert_pinyin_to_ipa(cleaned, durations)
    elif language in ("ja", "Japanese"):
        return _convert_mfa_japanese_to_ipa(cleaned, durations)
    elif language in ("en", "English"):
        return _convert_arpabet_to_ipa(cleaned, durations)
    else:
        # Unknown language — pass through
        return cleaned, durations
