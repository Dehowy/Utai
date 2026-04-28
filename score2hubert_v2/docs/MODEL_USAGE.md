# Score2HuBERT v2 Run 9 — Model Usage Reference

**This is the authoritative reference for using the shipped Run 9 model.**

Last updated: 2026-04-17 (Run 9 complete, user-validated).

---

## 1. Quick reference

| 项 | 值 |
|---|---|
| Checkpoint | `archive/run9_final/best.pt` (638 MB) |
| Architecture | Score2HuBERT v2, 54.63M params |
| Input | `(phonemes, pitch, durations)` phone-level tensors + optional `f0_curve` |
| Output | `hubert_pred [T_frame, 768]`, `f0_pred [T_frame]`, `voicing_logit [T_frame]` |
| Frame rate | **50 fps** (audio sample = HuBERT frame × 320 at 16kHz internal) |
| Downstream | RVC (HiFi-GAN) decodes HuBERT + F0 → **48kHz mono audio** |
| Post-process | `apply_midi_mute.py` — **default on** (see Section 6) |

---

## 2. Architecture diagram

```
Score input (USTX / piano roll)
  │
  ├── phonemes    [T_phone] int     ← vocab: data/processed/ace_opencpop/vocab.yaml
  ├── pitch       [T_phone] int     ← MIDI note 0-108 (0 = rest)
  └── durations   [T_phone] int     ← frames at 50fps
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│ Score2HuBERT.forward(phonemes, pitch, durations, f0_target=None)
│                                                          │
│   step 1: pitch[phonemes==SP_ID] = 0   ← SP fix           │
│   step 2: ScoreEncoder (Conformer 5L)  → phone_hidden [T_phone, 512]
│   step 3: LengthRegulator              → frame_hidden [T_frame, 512]
│   step 4: F0Predictor(frame_hidden)    → f0_pred, voicing_logit
│   step 5: F0CondProj(f0)               → f0_emb
│   step 6: DetHubertHead(frame_h+f0_emb) → hubert_pred [T_frame, 768]
│                                                          │
└──────────────────────────────────────────────────────────┘
  │
  ▼
RVC (yachiyo_runami_e350_s10500.pth or ACE-singer1 or user's own)
  Input:  hubert_pred [T, 768] + f0 [T]
  Output: audio [n_samples] @ 48kHz mono
  │
  ▼
apply_midi_mute.py (default on)
  Input:  audio + score (phonemes/pitch/duration)
  Output: audio with SP rest regions fade-out-muted
  │
  ▼
Final audio → DAW track / file export
```

---

## 3. Input tensor specs

All inputs are **phone-level** (NOT frame-level). Model's LengthRegulator expands
to frame-level internally via `durations` sum.

### `phonemes` — phoneme IDs
- **Shape**: `[B, T_phone]` or `[T_phone]` (batch inferred)
- **Dtype**: `torch.long` (int64)
- **Value range**: `0` to `vocab.n_phones - 1` (80 in current setup)
- **Source**: map lyrics + G2P → IPA → vocab.encode()
- **Vocab file**: `data/processed/ace_opencpop/vocab.yaml`

### `pitch` — MIDI note per phoneme
- **Shape**: `[B, T_phone]` or `[T_phone]`
- **Dtype**: `torch.long` (int64)
- **Value range**: `0` for rest/silent phonemes, typical melody range `36-108` (C2-C8)
- **Example**: `[69, 69, 71, 73, 71, 0, 0, 69]` — A4, A4, B4, C#5, B4, rest, rest, A4
- **SP handling**: model.forward() automatically sets `pitch[phonemes==SP_ID]=0` (SP_ID=3). But
  preprocessing is already fixed — `fix_sp_pitch.py` has zeroed SP positions in data.

### `durations` — frames per phoneme
- **Shape**: `[B, T_phone]` or `[T_phone]`
- **Dtype**: `torch.long` (int64)
- **Unit**: frames at 50fps (1 frame = 20ms)
- **Minimum**: 1 (any phoneme must have ≥1 frame)
- **Typical**: 3-30 frames (60ms-600ms)
- **Computation**: `durations[i] = round(phoneme_i_duration_seconds * 50)`

### `f0_curve` (optional, inference only)
- **Shape**: `[B, T_frame]` or `[T_frame]`
- **Dtype**: `torch.float32`
- **Unit**: Hz (e.g. 440.0 for A4)
- **Rest frames**: `0.0`
- **Use case**: Vocaloid/UTAU power users who want precision F0 control; bypasses
  model's F0 prediction. See Section 4.3 (Path A deployment).

---

## 4. Inference API

### 4.1 Load model

```python
import torch
from src.model.score2hubert import Score2HuBERT
import yaml

# Load config
with open("configs/train_run9.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Construct model with same config as training
model = Score2HuBERT(
    n_phones=cfg["model"]["n_phones"],         # 80
    hidden_dim=cfg["model"]["hidden_dim"],     # 512
    hubert_dim=cfg["model"]["hubert_dim"],     # 768
    n_enc_layers=cfg["model"]["n_enc_layers"], # 5
    n_heads=cfg["model"]["n_heads"],           # 8
    enc_kernel=cfg["model"]["enc_kernel"],     # 31
    n_det_blocks=cfg["model"]["n_det_blocks"], # 6
    det_kernel=cfg["model"]["det_kernel"],     # 7
    dropout=cfg["model"]["dropout"],           # 0.1
)

# Load checkpoint
ckpt = torch.load("archive/run9_final/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)  # strict=False for robustness
model.eval()
model.cuda()  # or cpu
```

**Note**: `strict=False` allows robustness across minor arch additions (e.g. pos_pe buffer)
but always verify no unexpected missing keys in loaded state.

### 4.2 Path A inference — model predicts F0 (default)

User provides only phonemes + MIDI + duration. Model's F0Predictor produces natural
singing F0 curve (with vibrato, portamento, intonation drift learned from training data).

```python
# Prepare tensors
phonemes = torch.tensor(phoneme_ids, dtype=torch.long, device="cuda")   # [T_phone]
pitch = torch.tensor(midi_values, dtype=torch.long, device="cuda")      # [T_phone]
durations = torch.tensor(frame_counts, dtype=torch.long, device="cuda") # [T_phone]

# Inference
with torch.no_grad():
    out = model.synthesize(phonemes, pitch, durations)
    # or: out = model.forward(phonemes.unsqueeze(0), pitch.unsqueeze(0), durations.unsqueeze(0))

hubert = out["hubert"]    # [1, T_frame, 768]  — pred HuBERT
f0 = out["f0"]            # [1, T_frame]       — pred F0 in Hz
```

### 4.3 Path B inference — user provides F0 curve

For users with hand-drawn F0 / exact MIDI-snap. Bypasses F0Predictor's output (but
F0Predictor still runs internally for voicing_logit).

```python
# User-provided F0 curve
f0_curve = torch.tensor(user_f0_hz, dtype=torch.float32, device="cuda")  # [T_frame]

with torch.no_grad():
    out = model.synthesize(phonemes, pitch, durations, f0_curve=f0_curve)

hubert = out["hubert"]    # [1, T_frame, 768]
f0 = out["f0"]            # [1, T_frame] — same as input f0_curve
```

### 4.4 Full `forward()` (training or advanced)

```python
out = model.forward(
    phonemes,      # [B, T_phone]
    pitch,         # [B, T_phone]
    durations,     # [B, T_phone]
    phone_mask=None,   # [B, T_phone] True=pad; optional
    f0_target=None,    # [B, T_frame] teacher-force; optional
)

# Returns full dict:
out["hubert_pred"]        # [B, T_frame, 768]
out["f0_pred"]            # [B, T_frame] predicted f0 in Hz
out["voicing_logit"]      # [B, T_frame] unnormalized voicing scores
out["note_pitch_frame"]   # [B, T_frame] long — MIDI expanded to frames (for F0 loss)
out["phone_hidden"]       # [B, T_phone, 512]
out["frame_hidden"]       # [B, T_frame, 512]
out["pos_pred"]           # [B, T_frame, 2] sigmoid pos_in_phone + pos_in_note
```

---

## 5. RVC integration

### 5.1 Model pair assignment

| Target voice | RVC model | Notes |
|---|---|---|
| **JA (Yachiyo Runami)** | `D:/MyDev/MSSTRVC/RVC/RVC20240604Nvidia/assets/weights/yachiyo_runami_e350_s10500.pth` | Main JA singer, trained Run 8 |
| **ZH (ACE singer1)** | `D:/MyDev/MSSTRVC/score2hubert_v2/archive/run8_sofa_aug/rvc_ace_singer1/ace_zh_singer1.pth` | Native ZH voice, avoids "Chinese-accented-by-JA-RVC" artifact |
| User's own | custom | Use RVC training pipeline with any target singer |

### 5.2 RVC input format

RVC expects:
- `hubert` tensor `[T_frame, 768]` at 50fps
- `f0` tensor `[T_frame]` at 50fps, in Hz (or semitone offset depending on RVC version)
- Output: audio at RVC's sample rate (typically 48kHz for `yachiyo_runami`)

See `scripts/eval_audio.py` for reference integration (it does exactly this pipeline for
evaluation; same functions work in production).

### 5.3 `eval_audio.py` as production template

The 4 internal variants it generates demonstrate useful inference modes:
- `pred_full`: model predicts both HuBERT and F0 → Path A equivalent
- `h_pred_f0gt`: pred HuBERT + user F0 → Path B equivalent (diagnose F0 quality)
- `hgt_f0pred`: GT HuBERT + pred F0 → diagnose F0 quality in isolation
- `gt_ceiling`: GT HuBERT + GT F0 → upper bound for dataset

Production uses `pred_full` (Path A) or `h_pred_f0gt` (Path B) — others are for eval only.

---

## 6. Post mute (CRITICAL — default on)

After RVC produces audio, apply `scripts/apply_midi_mute.py` to kill residual electric
artifacts in SP rest regions.

### 6.1 Why it's essential

Run 9 trained `pred_hubert[SP frames]` to ~0 successfully. However **RVC's HiFi-GAN**
is sensitive to:
- Residual F0 values at SP frames (model predicts ~0 Hz but not exactly 0)
- Tiny HuBERT residues (~0.001-0.01) at SP frames

These combine to produce perceptible soft electric buzz / static in rests. Post mute
silences the audio amplitude at these regions with a smooth fade.

### 6.2 CLI usage

```bash
python scripts/apply_midi_mute.py \
  --in-dir <dir with sampleN_*.wav> \
  --out-dir <output dir> \
  --dataset <dataset_name> \
  --utt-ids <utt_id_1> <utt_id_2> ...
```

`--dataset` must match a directory in `data/processed/` because the script loads
`phonemes.pt`, `pitch.pt`, `duration.pt` from `data/processed/<dataset>/`.

### 6.3 Programmatic usage

```python
from pathlib import Path
from scripts.apply_midi_mute import apply_mute

apply_mute(
    in_wav=Path("raw/pred_full.wav"),
    out_wav=Path("muted/pred_full.wav"),
    utt_id="kiritan_01_000",       # used to look up tensors from data/processed/<dataset>/
    dataset="kiritan",
    # Defaults (DO NOT CHANGE without reading memory):
    fade_ms=5.0,                   # 音尾 fade-out (only)
    min_rest_ms=120.0,             # shorter MIDI=0 gaps treated as labeling artifact (not muted)
    note_onset_fade_ms=0.0,        # ← 0! never re-enable (produces glitch)
    voicing_dir=None,              # optional voicing-gate (not used currently)
)
```

### 6.4 DAW integration (planned)

For DAW inference, phonemes/pitch/duration come from USTX parsing, NOT from
`data/processed/`. Integrate mute logic directly in `synth_wrapper.py` post-RVC step:

```python
# Pseudocode
audio = rvc_decode(hubert, f0)  # 48kHz mono
from scripts.apply_midi_mute import build_envelope
frame_midi = build_frame_midi_from_ustx(ustx_notes)  # skip data/processed/ lookup
envelope = build_envelope(frame_midi, len(audio), sr=48000)
audio_muted = audio * envelope
```

### 6.5 Advanced user toggle (per user directive)

Expose in DAW advanced settings as dropdown:

| Option | Behavior | Target user |
|---|---|---|
| **"Auto" (default)** | Apply post mute | Most users |
| "Raw" | No mute, use RVC output directly | Users with pristine backends or custom DSP |
| "Inline only" | `--hubert-mute combined` at model forward, no post | (Note: for Run 9 this is ≈ no-op; keep for future models) |
| "Both" | Inline + post | Maximum defensive (same as Auto for Run 9) |

**Critical**: "Auto" MUST use current `apply_midi_mute.py` defaults (no 音头 fade). Never
re-enable rising fade or note-onset ramp — audible GLITCH at attack.

---

## 7. Vocab.yaml reference

```yaml
# Located: data/processed/ace_opencpop/vocab.yaml
# All datasets share this vocab (copied during preprocessing).

phone2id:
  <pad>: 0
  <sos>: 1
  <eos>: 2
  SP: 3      # silence/rest (main pause token)
  AP: 4      # aspiration/breath (ACE-only; Run 9 may not use since SOFA collapses to SP)
  br: 5      # breath marker
  <mask>: 6
  # ... IPA phonemes 7-72 ...
  <unused_73-79>: 73-79

lang2id:
  ja: 0
  zh: 1
  en: 2  # unused (no EN data after Run 3)
  ko: 3  # reserved
```

**SP_ID = 3** is hardcoded in multiple places (`model.forward`, `trainer.compute_loss`,
`apply_midi_mute.py`). Do not change without updating all.

### 7.1 G2P for Chinese (pinyin → IPA)

Use `src/data/phoneme_v1._convert_pinyin_to_ipa(phonemes, durations)`. Input: opencpop-ext
tokens (`zh`, `ch`, `sh`, `a`, `ai`, `an`, ...). Output: IPA segments matching vocab.

Example:
```python
from src.data.phoneme_v1 import _convert_pinyin_to_ipa
ipa, durs = _convert_pinyin_to_ipa(
    ["sh", "a", "n", "SP"],
    [5.0, 10.0, 5.0, 3.0]
)
# → ipa: ['ʂ', 'a', 'n', 'SP']; durs: [5.0, 10.0, 5.0, 3.0]
# (ang/ian/uan etc. will split into multiple IPA segments)
```

### 7.2 G2P for Japanese (OpenJTalk → IPA)

Use `src/data/phoneme_v1._convert_openjtalk_to_ipa()`. Input: OpenJTalk phonemes
(`a i u e o, k g s z t d n h b p m y r w, N, sh ch j ts, ky gy...`). Output: IPA.

Typical pipeline: `pyopenjtalk.extract_fullcontext(hiragana_text)` → parse labels →
phoneme list → `_convert_openjtalk_to_ipa`.

### 7.3 English (NOT SUPPORTED)

Vocab has IPA mappings from `ARPABET_TO_IPA` (in `phoneme_v1.py`) but **EN not in
Run 9 training data**. Output will be unreliable for pure English. Users wanting
English should use romaji/katakana phonetic spelling with a JA voice model.

---

## 8. Frame rate and timing

- **HuBERT/model internal**: 50 fps (1 frame = 20ms at 16kHz audio internal)
- **Conversion**: `T_frame = sum(durations)` where durations are in 50fps frames
- **RVC output**: typically 48kHz audio → `audio_samples_per_frame = 48000 / 50 = 960`
- **Apply mute matches**: `SAMPLES_PER_FRAME = 960` in apply_midi_mute.py

**Sample rate path**:
```
Source audio   48kHz  →  resample 16kHz  →  HuBERT 50fps  →  RVC decoder  →  48kHz output
```

---

## 9. Known limits / gotchas

### 9.1 Sample length
- `max_frames=500` during training (10s @ 50fps)
- Inference can handle longer sequences but VRAM scales O(T²) in Conformer self-attention
- **Practical limit**: ~2000 frames (40s) on 12GB VRAM
- For longer: split into chunks with overlap, crossfade

### 9.2 Language scope
- **JA and ZH well-trained**. EN experimental at best (no EN in Run 9 data).
- Cross-language: IPA allows some transfer, but safest: match RVC target language.
- ACE-trained ZH voice + JA RVC = audible JA accent on ZH; use ACE-singer1 RVC for ZH.

### 9.3 Silent/rest handling
- Model already trained for SP=0. Don't add extra SP-specific post-processing without
  checking feedback_post_mute_audio_head.md.
- Post mute `min_rest_ms=120` means <120ms MIDI=0 gaps are KEPT voiced (treated as
  labeling artifacts). Score inputs with genuine short rests < 120ms will NOT be
  muted — usually fine for singing contexts.

### 9.4 Checkpoint loading edge case
- Always use `strict=False` for `load_state_dict`. Model arch may have tiny buffer
  additions (pos_pe) between versions that aren't in the state_dict. Zero-init fallbacks
  are correct — this is expected behavior.

### 9.5 Device/dtype
- Checkpoint stored in fp32 state_dict. Inference in fp16 possible via `.half()` but
  untested at deployment; recommend fp32 for reliability.
- Model works on CUDA or CPU. CPU ~10× slower but usable for debugging.

---

## 10. Performance benchmarks (RTX 3080 Ti)

| Mode | Speed |
|---|---|
| Training (batch 112, max_frames 500, fp16) | 0.7-1.0 step/s |
| Inference (batch 1, 500 frames) | ~50ms / forward |
| Eval_audio CPU (AMD 8845HS, 1 sample × 4 variants) | ~23s |
| Eval_audio GPU (1 sample × 4 variants) | ~1s |
| apply_midi_mute (per wav) | ~0.05s |

---

## 11. Training reproducibility

If user wants to retrain or fine-tune:
- Config: `configs/train_run9.yaml` (restored from `archive/run9_final/train.yaml`)
- Script: `python scripts/train.py --config configs/train_run9.yaml`
- Resume: `python scripts/train.py --config configs/train_run9.yaml --resume archive/run9_final/best.pt`

**Fine-tune workflow** (if adding new data):
1. Preprocess new data following existing scripts (preprocess_kiritan_itako.py pattern)
2. Add to `data.train_dirs` in config
3. **Lower LR** from 2e-4 to 2e-5 for fine-tune stability
4. **Resume from best.pt** — don't train from scratch
5. 5-10K additional steps usually enough
6. **Monitor per-lang val** — fine-tune can cause catastrophic forgetting on other langs

---

## 12. Files checklist

To deploy Run 9 model in a new environment, need:

**Required**:
- `archive/run9_final/best.pt` (model weights)
- `data/processed/ace_opencpop/vocab.yaml` (phone2id mapping)
- `src/model/` (Python model definition)
- `src/data/phoneme_v1.py` (G2P for ZH + JA)
- `RVC/.../yachiyo_runami_*.pth` (RVC decoder)

**Strongly recommended**:
- `scripts/apply_midi_mute.py` (default-on post processing)
- `scripts/eval_audio.py` (reference integration)
- `configs/train_run9.yaml` (model construction parameters)

**For fine-tuning**:
- All above + `src/training/trainer.py` + `src/data/dataset.py` + any new preprocessed data

---

**End of reference.**
