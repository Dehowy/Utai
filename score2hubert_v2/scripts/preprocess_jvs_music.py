"""Score2HuBERT v2 — JVS-MuSiC preprocessing.

Source: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music
Author: Hiroki Tamaru, Shinnosuke Takamichi et al. (The Univ. of Tokyo)
License: Audio data + MPD: non-commercial research/personal use; tags CC-BY-SA 4.0.
Citation: Tamaru et al., "JVS-MuSiC: Japanese multispeaker singing-voice corpus", arXiv 2001.07044, 2020.

Pipeline:
  1. For each singer (jvs001 to jvs100):
     - Read singer_info.txt to get unique song name
     - Process song_common/wav/raw.wav (Katatsumuri, common)
     - Process song_unique/wav/raw.wav (singer-specific song)
  2. For each (audio, lyrics):
     a. Use vumichien/wav2vec2-large-xlsr-japanese-hiragana to get logits @ 50fps
     b. Greedy decode → STT hiragana (used for confidence check)
     c. CTC-segmentation force-alignment with hardcoded lyrics → per-char timestamps
     d. Confidence filter: if mean prob < -2.0 OR edit distance(STT, ref) too large → drop
     e. Convert hiragana → OpenJTalk phonemes via pyopenjtalk
     f. Distribute kana time intervals to phoneme segments
     g. Extract HuBERT + F0 from full audio
     h. Save in standard format

Output: data/processed/jvs_music/
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import hashlib
import json
import logging
import shutil
import warnings
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

import ctc_segmentation as cs
import pyopenjtalk
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from src.data.phoneme_v1 import PhonemeVocab, OPENJTALK_TO_IPA
from src.preprocessing.extract_hubert import HuBERTExtractor
from src.data.japanese_song_lyrics import get_katatsumuri, get_lyrics, UNIQUE_SONG_LYRICS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("preprocess_jvs_music")


SAMPLE_RATE = 16000
FRAME_RATE = 50
HOP = SAMPLE_RATE // FRAME_RATE  # 320
W2V_MODEL = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
W2V_BLANK = 85  # PAD token id

# Confidence threshold for filtering
# ctc-segmentation returns log-probs per char; -0.5 is excellent, -2.0 acceptable, <-3.0 likely wrong lyrics
MIN_AVG_CHAR_PROB = -3.0


def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = cur[j] + 1
            sub = prev[j] + (c1 != c2)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def parse_singer_info(path: Path) -> dict:
    """Read singer_info.txt → dict of {jvs_id: {gender, key_group, tempo, key_singer, unique_song_name}}."""
    info = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                singer_id = parts[0]
                info[singer_id] = {
                    "gender": parts[1],
                    "key_group": parts[2],
                    "tempo": int(parts[3]),
                    "key_singer": int(parts[4]),
                    "unique_song_name": " ".join(parts[5:]),
                }
    return info


@torch.no_grad()
def compute_logits(audio16k: np.ndarray, w2v_model, w2v_proc, device: str):
    """Run wav2vec2 → log_probs [T, vocab]."""
    inputs = w2v_proc(audio16k, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).to(device)
    logits = w2v_model(inputs.input_values).logits[0].cpu().numpy()
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    return log_probs, logits


def greedy_decode(log_probs: np.ndarray, char_list: list, blank_id: int) -> str:
    """CTC greedy decode."""
    pred_ids = log_probs.argmax(axis=-1)
    out = []
    prev = -1
    for pid in pred_ids:
        if pid != blank_id and pid != prev:
            out.append(char_list[pid])
        prev = pid
    return "".join(c for c in out if not c.startswith("[") and not c.startswith("<"))


def align_chars(log_probs: np.ndarray, lyrics: str, char_list: list, vocab: dict, blank_id: int):
    """Force-align lyrics chars to log_probs frames.

    Returns (segments, mean_prob) where segments is list of (start_sec, end_sec, char, prob),
    or (None, None) if alignment failed.
    """
    config = cs.CtcSegmentationParameters()
    config.char_list = char_list
    config.index_duration = 1.0 / FRAME_RATE
    config.blank = blank_id

    text_chars = [c for c in lyrics if c in vocab]
    if len(text_chars) == 0:
        return None, None

    try:
        gt_mat, utt_indices = cs.prepare_text(config, text_chars)
        timings, char_probs, state_list = cs.ctc_segmentation(config, log_probs, gt_mat)
        segments = cs.determine_utterance_segments(config, utt_indices, char_probs, timings, text_chars)
    except Exception as e:
        logger.warning(f"alignment failed: {e}")
        return None, None

    out = []
    probs = []
    for ch, (start, end, prob) in zip(text_chars, segments):
        out.append((start, end, ch, prob))
        probs.append(prob)
    return out, float(np.mean(probs))


def kana_to_openjtalk_phonemes(kana_str: str) -> list:
    """Use pyopenjtalk to convert hiragana string → list of OpenJTalk phonemes."""
    try:
        result = pyopenjtalk.g2p(kana_str)
        return result.split()
    except Exception:
        return []


def assign_phoneme_to_segments(char_segments: list) -> list:
    """For each char (start, end), get its OpenJTalk phonemes and split duration evenly.

    Returns list of (start_sec, end_sec, phoneme).
    """
    out = []
    for start, end, ch, _prob in char_segments:
        phs = kana_to_openjtalk_phonemes(ch)
        if not phs:
            continue
        n = len(phs)
        d = end - start
        per = d / n
        for i, p in enumerate(phs):
            s = start + i * per
            e = start + (i + 1) * per if i < n - 1 else end
            out.append((s, e, p))
    return out


def jp_phoneme_to_ipa_list(ph: str) -> list:
    if ph in ("pau", "sil", "sp"):
        return ["SP"]
    if ph == "br":
        return ["br"]
    if ph in OPENJTALK_TO_IPA:
        return list(OPENJTALK_TO_IPA[ph])
    return ["SP"]


@torch.no_grad()
def hubert_from_array(model, audio16k: np.ndarray, device: str, use_fp16: bool) -> torch.Tensor:
    audio = torch.from_numpy(audio16k).float().unsqueeze(0).to(device)
    if use_fp16:
        audio = audio.half()
    audio = F.pad(audio, (40, 40), mode="reflect")
    feats = model.extract_features(source=audio, padding_mask=None, output_layer=12)
    return feats[0].squeeze(0).float().cpu()


def rmvpe_to_frames(rmvpe, audio16k: np.ndarray, n_target: int) -> torch.Tensor:
    f0_100 = rmvpe.infer_from_audio(audio16k.astype(np.float32), thred=0.03)
    f0_100 = np.asarray(f0_100, dtype=np.float32)
    if f0_100.size == 0 or n_target <= 0:
        return torch.zeros(n_target, dtype=torch.float32)
    src_len = f0_100.shape[0]
    idxs = np.clip((np.arange(n_target) * 2), 0, src_len - 1)
    return torch.from_numpy(f0_100[idxs]).float()


def f0_hz_to_midi(f0_hz: float) -> float:
    if f0_hz <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(f0_hz / 440.0)


def split_for(utt_id: str) -> str:
    h = int(hashlib.md5(utt_id.encode("utf-8")).hexdigest(), 16)
    return "train" if (h % 10) < 9 else "test"


def process_audio(
    audio_path: Path, lyrics: str, song_label: str, singer_id: str, song_type: str,
    w2v_model, w2v_proc, char_list: list, w2v_vocab: dict, blank_id: int,
    hubert_model, rmvpe, vocab: PhonemeVocab,
    out_dirs: dict, device: str, use_fp16: bool,
    diagnostics: dict,
):
    """Process one audio file. Return metadata dict or None if filtered out."""
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if len(audio) < SAMPLE_RATE * 1.0:
        diagnostics["too_short"] += 1
        return None

    log_probs, _ = compute_logits(audio, w2v_model, w2v_proc, device)
    stt = greedy_decode(log_probs, char_list, blank_id)
    char_segs, mean_prob = align_chars(log_probs, lyrics, char_list, w2v_vocab, blank_id)

    if char_segs is None or mean_prob is None:
        diagnostics["align_failed"] += 1
        diagnostics["failed_samples"].append({"singer": singer_id, "song": song_label, "reason": "align_failed"})
        return None

    # Confidence filter — rely on alignment mean_prob only.
    # STT vs ref edit_distance is unreliable due to wav2vec2 word-delim "|" tokens
    # and singer vowel elongation ("でえん" vs "でん"). mean_prob is the right signal.
    ref_chars = [c for c in lyrics if c in w2v_vocab]
    edist = edit_distance(stt.replace("|", "")[:len(lyrics) * 3], "".join(ref_chars))
    edist_ratio = edist / max(1, len(ref_chars))

    if mean_prob < MIN_AVG_CHAR_PROB:
        diagnostics["low_confidence"] += 1
        diagnostics["failed_samples"].append({
            "singer": singer_id, "song": song_label,
            "reason": f"low_conf: mean_prob={mean_prob:.2f}, edist_ratio={edist_ratio:.2f}",
            "stt": stt[:80], "ref": "".join(ref_chars)[:80],
        })
        return None

    # OK: build phoneme segments
    ph_segs = assign_phoneme_to_segments(char_segs)
    if not ph_segs:
        diagnostics["no_phonemes"] += 1
        return None

    # Compute HuBERT for whole audio
    hubert = hubert_from_array(hubert_model, audio, device, use_fp16)
    T = hubert.shape[0]
    if T < 20:
        diagnostics["too_short"] += 1
        return None
    f0 = rmvpe_to_frames(rmvpe, audio, T)

    # Convert to IPA + frame-level duration, normalize sum to T
    # First: per OJT-phoneme dur in frames
    jp_phonemes = []
    jp_dur_frames = []
    for s, e, ph in ph_segs:
        df = max(1, int(round((e - s) * FRAME_RATE)))
        jp_phonemes.append(ph)
        jp_dur_frames.append(df)

    # Normalize sum to T: scale factor + diff push to argmax
    cur_sum = sum(jp_dur_frames)
    if cur_sum != T:
        scale = T / cur_sum
        rescaled = [max(1, int(round(d * scale))) for d in jp_dur_frames]
        diff = T - sum(rescaled)
        if diff != 0:
            j = int(np.argmax(rescaled))
            rescaled[j] += diff
            if rescaled[j] < 1:
                rescaled[j] = 1
                diff2 = T - sum(rescaled)
                if diff2 != 0:
                    rescaled[int(np.argmax(rescaled))] += diff2
        jp_dur_frames = rescaled

    # Convert to IPA, splitting multi-segment OJT phonemes proportionally
    ipa_phs, ipa_durs = [], []
    for ph, d in zip(jp_phonemes, jp_dur_frames):
        segs = jp_phoneme_to_ipa_list(ph)
        n = len(segs)
        if n == 1:
            ipa_phs.append(segs[0])
            ipa_durs.append(d)
        else:
            per = d // n
            rem = d - per * n
            for i, s in enumerate(segs):
                ipa_phs.append(s)
                ipa_durs.append(per + (rem if i == n - 1 else 0))

    # Encode IPA → IDs
    phoneme_ids = vocab.encode(ipa_phs, language="ja")
    L = min(len(phoneme_ids), len(ipa_durs))
    phoneme_ids = phoneme_ids[:L]
    ipa_durs = ipa_durs[:L]

    # Re-normalize after IPA expansion
    cur_sum = sum(ipa_durs)
    if cur_sum != T:
        diff = T - cur_sum
        if abs(diff) <= 5 and len(ipa_durs) > 0:
            j = int(np.argmax(ipa_durs))
            ipa_durs[j] = max(1, ipa_durs[j] + diff)

    # Estimate MIDI from F0 per IPA segment
    midi_pitches = []
    pos = 0
    for d in ipa_durs:
        if pos >= T:
            midi_pitches.append(0)
            continue
        chunk = f0[pos:pos+d]
        voiced = chunk[chunk > 30.0]
        if len(voiced) > 0:
            med = float(voiced.median())
            midi = int(round(f0_hz_to_midi(med)))
            midi_pitches.append(max(0, min(127, midi)))
        else:
            midi_pitches.append(0)
        pos += d

    # Save
    utt_id = f"jvs_music_{singer_id}_{song_type}"
    torch.save(hubert, out_dirs["hubert"] / f"{utt_id}.pt")
    torch.save(f0, out_dirs["f0"] / f"{utt_id}.pt")
    torch.save(torch.tensor(phoneme_ids, dtype=torch.long), out_dirs["phonemes"] / f"{utt_id}.pt")
    torch.save(torch.tensor(midi_pitches, dtype=torch.long), out_dirs["pitch"] / f"{utt_id}.pt")
    torch.save(torch.tensor(ipa_durs, dtype=torch.long), out_dirs["duration"] / f"{utt_id}.pt")

    diagnostics["accepted"] += 1
    return {
        "id": utt_id,
        "n_phones": L,
        "n_frames": T,
        "language": "ja",
        "lang_id": vocab.get_language_id("ja"),
        "singing_ratio": 1.0,
        "dataset": "jvs_music",
        "singer": singer_id,
        "song": song_label,
        "song_type": song_type,
        "split": split_for(utt_id),
        "alignment_mean_prob": float(mean_prob),
        "alignment_edist_ratio": float(edist_ratio),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "jvs_music_ver1"))
    ap.add_argument("--out-root", default=str(REPO_ROOT / "data" / "processed"))
    ap.add_argument("--hubert-model",
                    default=str(REPO_ROOT / "RVC/RVC20240604Nvidia/assets/hubert/hubert_base.pt"))
    ap.add_argument("--rmvpe-model",
                    default=str(REPO_ROOT / "RVC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt"))
    ap.add_argument("--vocab", default=str(REPO_ROOT / "data/processed/ace_opencpop/vocab.yaml"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--limit-singers", type=int, default=0,
                    help="0=all, else limit to first N singers (for testing)")
    ap.add_argument("--skip-common", action="store_true", help="skip Katatsumuri (common song)")
    ap.add_argument("--skip-unique", action="store_true", help="skip singer-unique songs")
    args = ap.parse_args()

    vocab = PhonemeVocab(args.vocab)

    logger.info("Loading wav2vec2 (Japanese hiragana CTC)...")
    w2v_proc = Wav2Vec2Processor.from_pretrained(W2V_MODEL)
    w2v_model = Wav2Vec2ForCTC.from_pretrained(W2V_MODEL).to(args.device).eval()
    w2v_vocab = w2v_proc.tokenizer.get_vocab()
    char_list = sorted(w2v_vocab.keys(), key=lambda k: w2v_vocab[k])

    logger.info("Loading HuBERT...")
    hubert_ext = HuBERTExtractor(args.hubert_model, layer=12, device=args.device, use_fp16=args.fp16)
    hubert_model = hubert_ext.model

    logger.info("Loading RMVPE...")
    sys.path.insert(0, str(REPO_ROOT / "RVC" / "RVC20240604Nvidia"))
    sys.path.insert(0, str(REPO_ROOT / "RVC" / "RVC20240604Nvidia" / "infer"))
    from infer.lib.rmvpe import RMVPE
    rmvpe = RMVPE(args.rmvpe_model, is_half=args.fp16, device=args.device)

    out_dir = Path(args.out_root) / "jvs_music"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = {n: (out_dir / n) for n in ("hubert", "f0", "phonemes", "pitch", "duration")}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.vocab, out_dir / "vocab.yaml")

    root = Path(args.root)
    singer_info = parse_singer_info(root / "singer_info.txt")
    logger.info(f"Loaded info for {len(singer_info)} singers")

    katatsumuri_lyrics = get_katatsumuri()

    diagnostics = {
        "accepted": 0,
        "align_failed": 0,
        "low_confidence": 0,
        "no_phonemes": 0,
        "too_short": 0,
        "missing_lyrics": 0,
        "failed_samples": [],
    }

    all_meta = []
    singer_ids = sorted(singer_info.keys())
    if args.limit_singers > 0:
        singer_ids = singer_ids[:args.limit_singers]

    pbar = tqdm(singer_ids, desc="jvs_music", unit="singer")
    for singer_id in pbar:
        info = singer_info[singer_id]
        # Common song
        if not args.skip_common:
            common_path = root / singer_id / "song_common" / "wav" / "raw.wav"
            if common_path.exists():
                m = process_audio(
                    common_path, katatsumuri_lyrics, "katatsumuri", singer_id, "common",
                    w2v_model, w2v_proc, char_list, w2v_vocab, W2V_BLANK,
                    hubert_model, rmvpe, vocab, out_dirs, args.device, args.fp16,
                    diagnostics,
                )
                if m: all_meta.append(m)
        # Unique song
        if not args.skip_unique:
            unique_path = root / singer_id / "song_unique" / "wav" / "raw.wav"
            song_name = info["unique_song_name"]
            unique_lyrics = get_lyrics(song_name)
            if not unique_lyrics:
                diagnostics["missing_lyrics"] += 1
                continue
            if unique_path.exists():
                m = process_audio(
                    unique_path, unique_lyrics, song_name, singer_id, "unique",
                    w2v_model, w2v_proc, char_list, w2v_vocab, W2V_BLANK,
                    hubert_model, rmvpe, vocab, out_dirs, args.device, args.fp16,
                    diagnostics,
                )
                if m: all_meta.append(m)

        pbar.set_postfix(
            ok=diagnostics["accepted"],
            low_conf=diagnostics["low_confidence"],
        )

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=0)

    # Save diagnostics
    with open(out_dir / "preprocessing_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    total_frames = sum(e["n_frames"] for e in all_meta)
    logger.info(f"\n=== JVS-MuSiC preprocessing summary ===")
    logger.info(f"Accepted: {diagnostics['accepted']}")
    logger.info(f"Total: {total_frames} frames = {total_frames/FRAME_RATE/3600:.2f}h")
    logger.info(f"Filtered out:")
    logger.info(f"  align_failed: {diagnostics['align_failed']}")
    logger.info(f"  low_confidence: {diagnostics['low_confidence']}")
    logger.info(f"  no_phonemes: {diagnostics['no_phonemes']}")
    logger.info(f"  too_short: {diagnostics['too_short']}")
    logger.info(f"  missing_lyrics: {diagnostics['missing_lyrics']}")


if __name__ == "__main__":
    main()
