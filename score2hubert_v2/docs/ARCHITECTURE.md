# Score2HuBERT v2 — Architecture

## What this is

Neural SVS frontend: musical score -> HuBERT features + F0 -> RVC decoder -> audio.
Replaces DiffSinger in the Utai DAW. Bypasses the audio domain entirely
(no mel -> vocoder -> audio -> HuBERT round-trip).

## Pipeline

```
Score input (from DAW):
  phonemes: IPA token sequence       (what to sing)
  pitch:    MIDI note per phoneme    (what pitch)
  duration: frames per phoneme       (how long)

Score2HuBERT model:
  ScoreEncoder (Conformer 4L, ~35M)
    phoneme_embed + pitch_embed + duration_proj -> input_proj -> Conformer -> phone_hidden
  LengthRegulator
    phone_hidden [B, T_phone, 512] -> frame_hidden [B, T_frame, 512]
  F0Predictor (~6M)
    frame_hidden + note_pitch -> f0_hz [B, T_frame]
    (bypassed when user provides pitch_curve from DAW)
  F0CondProj
    f0_hz -> f0_emb [B, T_frame, 512]
  DetHubertHead (~8M, 4 conv blocks)
    [frame_hidden; f0_emb] -> hubert_pred [B, T_frame, 768]

RVC (external, not trained):
  hubert_pred + f0 -> voice conversion -> final audio
  User controls: voice model, pitch_curve (vibrato), loudness curve
```

Total trainable: ~49M params.

## What was removed (vs v1)

| Removed | Why |
|---------|-----|
| flow_decoder (45M) | HuBERT is 80% deterministic given score. Flow matching fights this. |
| discrete_head | Broke cross-language phoneme sharing (token overlap 7%). |
| tech_proj | Phase 3 proved insufficient. Technique = RVC timbre + loudness curve. |
| lang_embed | IPA same-phoneme cross-language cos=0.876. Not needed. |
| singing_ratio input | Simplify. Rap mode can be added later without architecture change. |
| VarianceAdaptor | Energy/voicing predictors contributed <0.15 loss weight. User draws loudness in DAW. |
| hybrid/refiner | Continuous regression IS correct for this task. |
| EMA | Pathological for warm-start (v1 Phase C finding). Not using warm-start anyway. |

## Encoder input design

3-way concat (not 4-way like v1):
```
phone_emb  = Embedding(80, 512)         # IPA tokens
pitch_emb  = PitchEmbedding(512)        # sinusoidal + octave/pitch-class
dur_emb    = Linear(1, 256) -> SiLU -> Linear(256, 512)  # log-duration

combined   = cat([phone_emb, pitch_emb, dur_emb], dim=-1)  # [B, T, 1536]
x          = input_proj(combined)                           # [B, T, 512]
x          = x + pos_enc(x)
x          = conformer_blocks(x)
```

No lang_embed, no singing_ratio_proj, no tech_proj.

## DetHubertHead

```
in_proj  : Linear(2*512, 512)   # [frame_hidden; f0_emb]
4x block : Conv1d(512, 512, k=7) + GroupNorm(8) + GELU + Dropout(0.1), residual
out_proj : Linear(512, 768)
```

~8M params. RF = 28 frames = 560ms (syllable-level). Encoder provides global context.

## F0Predictor

Dual-mode singing/rap, but singing_ratio is hardcoded to 1.0 for now.
5 conv layers + 2 BiLSTM + singing_head (deviation from note) + voicing_head.
User provides pitch_curve in production -> F0Predictor is bypassed.

## Training

- Loss: L1 + 0.5*(1-cos) on HuBERT + log-F0 L1 + voicing BCE
- Optimizer: AdamW, lr 1.5e-4, warmup 1000 steps, cosine decay
- Batch: 48, FP16, no EMA
- From scratch (random init), mixed zh+ja data
- Data sampling: upsample ja ~4x to balance with zh

## Data

| Source | Lang | Duration | Entries | Source URL |
|--------|------|----------|---------|------------|
| ACE-Opencpop | zh | 121.8h | 100K | ACE Studio |
| Kiritan Singing DB | ja | 3.5h | 50 songs | https://zunko.jp/ |
| Itako Singing DB | ja | 1.5h | 60 songs | https://zunko.jp/ |
| UTAU zunko.jp (9 banks) | ja | 10.6h | 5,387 | https://zunko.jp/ |
| つくよみちゃんUTAU | ja | 1.5h | 1,245 | https://tyc.rei-yumesaki.net/material/utau/ |
| **Total** | | **139h** | | |

Ratio: zh 122h : ja 17h = 7:1. Upsample ja 4x -> effective 2:1.

### UTAU data notes
- Isolated phonemes/short syllables, no coarticulation context
- HuBERT variance diagnostic: UTAU std ratio vs ACE = 1.033 (3.3% more), std pattern cos = 0.990
- Conclusion: UTAU features live in identical HuBERT space as ACE. Safe to mix.
- Kiritan/Itako (5h continuous singing) provides coarticulation examples

### Key preprocessing facts
- HuBERT frame rate: 50fps (HuBERT-Base stride=320 at 16kHz)
- F0: RMVPE at 100fps, interpolated to 50fps to match HuBERT
- IPA phoneme vocab: ~80 tokens (66 IPA + 7 special + padding)

## License

**Derived model is NON-COMMERCIAL** (Kiritan/Itako/UTAU zunko.jp restrictions).
つくよみちゃん: AI training permitted, check source URL for commercial terms.

## File structure

```
score2hubert_v2/
├── src/
│   ├── model/
│   │   ├── modules.py          # Conformer block, LengthRegulator, etc.
│   │   ├── encoder.py          # ScoreEncoder (3-way input, Conformer)
│   │   ├── det_head.py         # DetHubertHead (conv regression)
│   │   ├── f0_predictor.py     # F0 prediction (singing/rap dual-mode)
│   │   └── score2hubert.py     # Full model assembly
│   ├── data/
│   │   ├── phoneme.py          # IPA vocabulary + G2P tables
│   │   ├── dataset.py          # Dataset class
│   │   └── collate.py          # Batch padding
│   ├── training/
│   │   └── trainer.py          # Training loop
│   ├── preprocessing/
│   │   ├── extract_hubert.py   # HuBERT feature extraction
│   │   ├── extract_f0.py       # F0 extraction (RMVPE)
│   │   └── prepare_dataset.py  # Per-dataset preprocessing
│   └── inference/
│       └── synthesizer.py      # End-to-end inference
├── scripts/
│   ├── train.py
│   ├── preprocess.py
│   └── eval_audio.py
├── configs/
│   └── train.yaml              # Single config
├── docs/
│   └── ARCHITECTURE.md         # This file
├── archive/                    # All v1 artifacts (don't use)
├── data/ -> junction to preprocessed data
├── kiritan/ -> junction to raw ja data
├── RVC/ -> junction to RVC models
└── venv/ -> junction to Python env
```

## v1 experiment history (for context only)

Full v1 history at `archive/docs_v1_reference/`. Key lessons:
1. Plan A (flow matching): learned F0 perfectly but HuBERT was "generic voice" — no lyrics
2. Plan B (AdaLN fix): never implemented, superseded by Plan C
3. Plan C (det_head): cos 0.854 on ACE-Opencpop, clear audio with lyrics
4. Phase 2 (multilingual): F0 truncation bug, fixed. GTSinger human singing exposed det_head averaging.
5. Phase 3-5 (tech/hybrid/discrete): all insufficient. Continuous regression remains correct.
6. Key finding: HuBERT space is ~80% deterministic given score (cross-singer cos=0.80).
   Flow matching wastes capacity on the 20% speaker variation that RVC replaces anyway.
