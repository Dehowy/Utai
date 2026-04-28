# Score2HuBERT: Technical Architecture Document

## 1. Motivation & Problem Statement

### 1.1 Current Pipeline Limitations

The existing Utai synthesis pipeline operates as:

```
Score (USTX) → DiffSinger → mel-spectrogram → Vocoder → Audio(wav) → RVC HuBERT Encoder → HuBERT features → RVC Decoder → Audio(wav)
```

This architecture suffers from three fundamental inefficiencies:

1. **Redundant codec cycle**: DiffSinger generates audio from a mel-spectrogram via a vocoder, only for RVC to immediately decompose that audio back into HuBERT features. This audio→feature round-trip introduces quantization noise and information loss inherent to any audio-domain intermediate.

2. **Latency compounding**: DiffSinger's diffusion process requires 50-100 denoising steps, followed by vocoder synthesis, then RVC's HuBERT extraction + decoder inference. Total latency per utterance: 5-15 seconds.

3. **Error accumulation**: Artifacts from DiffSinger's vocoder (spectral leakage, phase discontinuities) propagate into RVC's feature space, producing "metallic" or "synthetic" timbre artifacts that the RVC decoder cannot correct.

### 1.2 Proposed Architecture

```
Score → Score2HuBERT (Flow Matching) → HuBERT features + F0 → RVC Decoder (net_g) → Audio
```

Key insight: **bypass the audio domain entirely**. The Score2HuBERT model learns a direct mapping from musical scores to the internal representation space that RVC's decoder already understands.

### 1.3 Theoretical Advantages

| Metric | Old Pipeline | New Pipeline | Improvement |
|--------|-------------|-------------|-------------|
| Inference steps | 50-100 (diffusion) + vocoder + HuBERT | 4-10 (flow matching) + decoder | ~10x faster |
| Information loss | 2 codec cycles (audio bottleneck) | 0 codec cycles | Lossless feature transfer |
| Trainable voice bank | DiffSinger voice bank (hours of data + specialized training) | RVC model (5 min audio + simple training) | ~100x lower barrier |
| Expressiveness | Limited by DiffSinger's mel-space | Full 768-dim HuBERT representation | Higher dimensionality |

---

## 2. Model Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       Score2HuBERT Model                        │
│                                                                 │
│  ┌───────────────┐    ┌──────────────┐    ┌─────────────────┐  │
│  │ Score Encoder  │───▶│   Length     │───▶│   Variance      │  │
│  │  (Conformer)   │    │  Regulator   │    │   Adaptor       │  │
│  │  T_phone→512   │    │ phone→frame  │    │ +energy,breath  │  │
│  └───────────────┘    └──────────────┘    └────────┬────────┘  │
│         │                                          │            │
│         │                                   ┌──────▼──────┐    │
│         │                                   │     F0       │    │
│         │                                   │  Predictor   │    │
│         │                                   │ (sing + rap) │    │
│         │                                   └──────┬───────┘   │
│         │                                          │            │
│         │                                   ┌──────▼──────┐    │
│         │                                   │ Flow Match.  │    │
│         │                                   │  Decoder     │    │
│         │                                   │  (OT-CFM)    │    │
│         │                                   │ →HuBERT 768  │    │
│         │                                   └──────┬───────┘   │
└─────────┼──────────────────────────────────────────┼────────────┘
          │                                          │
          │                           ┌──────────────▼──────────────┐
          │                           │ RVC Decoder (net_g)          │
          │                           │ HuBERT[768] + F0 → Audio    │
          │                           │ (frozen, pre-trained)        │
          │                           └─────────────────────────────┘
```

### 2.2 Score Encoder (Conformer)

**Purpose**: Extract contextualized representations from the musical score.

**Input dimensions** (per phoneme):
- Phoneme ID: integer, embedded to 512-dim
- MIDI pitch: 0-127, embedded via sinusoidal + octave/pitch-class decomposition → 512-dim
- Duration: integer frames, log-scaled + MLP → 512-dim
- Singing ratio: float [0,1], MLP → 512-dim

**Architecture**:
```
Input: [B, T_phone, 512×4] → Linear → [B, T_phone, 512]
    ↓
Sinusoidal Positional Encoding
    ↓
Conformer Block × 6:
    ├── FFN(½) + residual
    ├── Multi-Head Self-Attention (8 heads, relative positional bias) + residual
    ├── Convolution Module (depthwise conv, kernel=31) + residual
    └── FFN(½) + residual + LayerNorm
    ↓
Output: [B, T_phone, 512]
```

**Design rationale**:
- Conformer chosen over Transformer for its superior performance on sequential audio tasks (combining global attention with local convolution)
- Relative positional encoding handles variable-length sequences without position truncation
- SwiGLU activation in FFN for better gradient flow
- Kernel size 31 in ConvModule captures ~300ms local context at phoneme level

**Parameter count**: ~15.7M

### 2.3 Pitch Embedding Design

Pitch is encoded through a hybrid scheme:

1. **Sinusoidal component** (256-dim): Captures the continuous, harmonic nature of pitch. Adjacent semitones have high cosine similarity, reflecting perceptual similarity.

2. **Octave embedding** (128-dim, learned): 12 octave buckets. Captures octave-level register information that affects vocal production mechanism (chest voice vs. head voice).

3. **Pitch class embedding** (128-dim, learned): 12 semitone buckets. Captures scale-degree function (tonic, dominant, etc.) which affects musical phrasing.

This decomposition is inspired by pitch perception theory — we perceive pitch as having both a "height" (continuous frequency) and a "chroma" (cyclic, repeating across octaves).

### 2.4 Length Regulator

Expands phoneme-level hidden states to frame-level:
```
[B, T_phone, 512] → repeat_interleave(durations) → [B, T_frame, 512]
```

During training: uses ground-truth durations from forced alignment.
During inference: uses durations from the score (given by the DAW) or predicted by the optional duration predictor.

### 2.5 Variance Adaptor

Predicts frame-level prosodic features and injects them into the hidden state:

| Feature | Architecture | Purpose |
|---------|-------------|---------|
| Energy | Conv1D×3 → scalar | Loudness/dynamics control |
| Breathiness | Conv1D×3 → scalar | Aspiration level (breathy↔clear) |
| Voicing | Conv1D×2 → scalar | Vocal fold tension |

Each predictor outputs a scalar value per frame, which is projected back to 512-dim and added to the hidden state. During training, teacher forcing uses ground-truth values extracted from audio.

### 2.6 F0 Predictor (Dual-Mode)

**Architecture**:
```
[condition, note_pitch_embedding] → Conv1D×5 (kernel=5) → BiLSTM×2 → {singing_head, rap_head, voicing_head}
```

**Singing mode** (singing_ratio → 1.0):
- Input: frame-level hidden + note pitch
- Predicts: log-scale **deviation** from note pitch
- Output: `F0 = note_Hz × exp(deviation)`
- Captures vibrato, portamento, natural pitch variation around the written note

**Rap mode** (singing_ratio → 0.0):
- Input: frame-level hidden (no pitch reference)
- Predicts: **absolute** log-F0 contour
- Models speech-like intonation patterns (rising questions, falling statements, etc.)

**Interpolation**: `F0 = ratio × singing_F0 + (1 - ratio) × rap_F0`

This allows smooth transitions between singing and rap within a single song, which is common in modern J-pop, K-pop, and hip-hop fusion genres.

**Voicing head**: Binary classification (voiced/unvoiced) handles consonant breaks and breath sections.

**Parameter count**: ~4.2M

### 2.7 Flow Matching Decoder (OT-CFM)

**Why Flow Matching over Diffusion?**

| Property | Diffusion (DDPM/score-based) | Flow Matching (OT-CFM) |
|----------|------------------------------|----------------------|
| Transport path | Curved (through noise schedule) | Straight (linear interpolation) |
| Required steps | 50-100 for quality | 4-10 sufficient |
| Training objective | Score function ∇log p | Velocity field v(t,x) |
| ODE solving | Requires SDE/ODE conversion | Direct ODE |
| Training stability | Sensitive to noise schedule | Robust (no schedule needed) |

**Optimal Transport CFM (OT-CFM)**:

Standard CFM uses Gaussian conditional probability paths. OT-CFM uses optimal transport plans, which produce straighter paths between noise and data distributions. This means:
- Fewer integration steps needed
- Lower approximation error
- Better sample quality per compute budget

**Training**:
```python
t ~ Uniform[0, 1]
x_0 ~ Normal(0, I)              # noise
x_1 = target HuBERT features    # data
x_t = (1 - t) × x_0 + t × x_1  # linear interpolation (OT path)
v* = x_1 - x_0                  # target velocity (constant along path!)
Loss = ||v_θ(x_t, t, condition) - v*||²
```

**Vector Field Network (DiT architecture)**:
```
Input: x_t [B, T, 768] (noisy features)
Time: t → sinusoidal → MLP → d_cond (time embedding)

DiT Block × 6:
    ├── AdaLN(time_emb) → Self-Attention (8 heads) + residual
    ├── AdaLN(time_emb) → Cross-Attention(encoder_cond) + residual
    └── AdaLN(time_emb) → FFN + residual

Output: v_θ [B, T, 768] (predicted velocity)
```

**Inference (ODE solving)**:
```python
x_0 ~ Normal(0, temperature²·I)
for step in range(N_steps):
    t = step / N_steps
    v = v_θ(x_t, t, condition)
    x_t = x_t + v × dt    # Euler step
```

Higher-order solvers (midpoint, RK4) available for better accuracy at the same step count.

**Classifier-Free Guidance (CFG)**:

During training, randomly zero out the condition with probability 15%. At inference:
```
v_guided = v_uncond + w × (v_cond - v_uncond)
```
where `w > 1` strengthens the condition's influence. Default `w = 2.0`.

**Parameter count**: ~38.5M

### 2.8 Total Model Size

| Component | Parameters |
|-----------|-----------|
| Score Encoder | 15.7M |
| F0 Predictor | 4.2M |
| Variance Adaptor | 2.1M |
| Flow Matching Decoder | 38.5M |
| Other (embeddings, projections) | 1.5M |
| **Total** | **~62M** |

This is significantly smaller than DiffSinger (~100-200M) while targeting a simpler output space (768-dim features vs. mel-spectrogram + vocoder).

---

## 3. Feature Space Alignment with RVC

### 3.1 HuBERT Feature Specification

Our model must generate features that are "in-distribution" for RVC's pre-trained decoder (net_g). The specifications:

| Property | Value | Source |
|----------|-------|--------|
| Model | HuBERT-Base (fairseq) | `hubert_base.pt` |
| Layer | 12 (V2) / 9 (V1) | All project models are V2 |
| Dimension | 768 | Fixed by architecture |
| Frame rate | 50 fps | 16kHz / 320 internal hop (HuBERT-Base) |
| Dtype | float32 | Stored as fp32 |
| Upsampling | 2× nearest-neighbor before decoder | Applied in inference |

### 3.2 F0 Specification

| Property | Value |
|----------|-------|
| Range | 50-1100 Hz |
| Frame rate | 50 fps |
| Unvoiced | 0.0 Hz |
| Coarse quantization | mel-scale → 1-255 bins |
| Fine representation | continuous Hz (float32) |

Coarse F0 formula (matching RVC):
```
f0_mel = 1127 × log(1 + f0/700)
f0_norm = (f0_mel - mel_min) / (mel_max - mel_min)
f0_coarse = round(f0_norm × 254 + 1)  ∈ [1, 255]
```

### 3.3 RVC Compensation

#### Problem
RVC's decoder (net_g) was trained on HuBERT features extracted from real audio. These features have specific statistical properties:
- Bounded dynamic range per dimension
- Specific inter-dimension correlations
- Temporal smoothness characteristics

Our generated features might be "too sharp" or "too flat" in dimensions where net_g has learned specific response curves.

#### Solution: Roundtrip Compensation

```
H_gt = HuBERT(original_audio)           # Ground truth features
audio_recon = net_g(H_gt, F0)           # RVC reconstruction
H_roundtrip = HuBERT(audio_recon)       # Re-extracted features
Δ = H_gt - H_roundtrip                  # Information lost by net_g
H_target = H_gt + α × Δ                # Compensated training target
```

By training our model to output `H_target` instead of `H_gt`, we "over-express" the details that net_g tends to smooth, so the final output after RVC processing is closer to the original.

The compensation factor `α ∈ [0, 1]` controls compensation strength:
- `α = 0`: no compensation (train on raw HuBERT features)
- `α = 0.35`: moderate compensation (recommended starting point)
- `α = 1.0`: full compensation (may cause instability)

#### Per-Dimension Analysis
Not all 768 dimensions are equally affected. Empirical analysis shows:
- Dimensions 0-255 (lower-level features): low Δ, minimal compensation needed
- Dimensions 256-511 (mid-level): moderate Δ, primary compensation targets
- Dimensions 512-767 (high-level semantic): variable Δ, careful tuning needed

---

## 4. Singing vs. Rap: The Vocal Continuum

### 4.1 Why Traditional SVS Fails at Rap

Traditional singing voice synthesis assumes:
- Each phoneme has a **stable target pitch** (the written note)
- Pitch varies smoothly (vibrato, portamento) around this target
- Duration is metrically quantized to the musical grid

Rap violates all three assumptions:
- Pitch is **contour-based**, not target-based (rising/falling speech intonation)
- Pitch varies rapidly and unpredictably (emphasis, emotion)
- Duration is **rhythm-driven**, not metrically strict

### 4.2 HuBERT's Natural Advantage

HuBERT was trained on **speech** data via self-supervised learning. It encodes:
- Phonetic content (what sounds are being produced)
- Prosodic structure (rhythm, stress, intonation)
- Speaker characteristics (timbre, vocal quality)

**Crucially, HuBERT makes no distinction between singing and speaking**. Both are just different operating points in the same continuous vocal production space. Rap sits naturally between speech and singing — exactly where HuBERT is most comfortable.

### 4.3 Implementation: Continuous Singing Ratio

Instead of a binary flag, we use `singing_ratio ∈ [0.0, 1.0]`:

| Value | Mode | F0 Prediction | Typical Use |
|-------|------|--------------|-------------|
| 1.0 | Pure singing | Deviation from note pitch | Ballad, opera |
| 0.7 | Melodic rap | Weighted combination | K-pop rap verses |
| 0.5 | Sing-rap | Equal interpolation | Modern J-pop |
| 0.3 | Rhythmic speech | Mostly absolute F0 | Spoken word |
| 0.0 | Pure rap | Absolute F0 contour | Hip-hop verse |

This gradient allows the model to handle the full spectrum of vocal styles, including:
- **Sprechgesang** (half-spoken, half-sung) common in musical theater
- **Rap-singing** transitions within a single phrase
- **Vocal fry** and other non-standard phonation modes

### 4.4 Training Strategy for Rap Support

**Phase 1**: Train on singing data only (singing_ratio = 1.0 for all samples)
**Phase 2**: Add speech data with singing_ratio = 0.0
**Phase 3**: Create synthetic "rap" data by processing speech through singing-style durations, with singing_ratio = 0.3

The model learns to interpolate between singing and speech in HuBERT feature space, naturally discovering the "rap" region.

---

## 5. Training Strategy

### 5.1 Multi-Phase Training

#### Phase 1: Deterministic Pre-training (optional fast-start)
- **Objective**: L1 loss on HuBERT features directly (no flow matching)
- **Purpose**: Establish strong encoder baseline
- **Steps**: ~20k
- **Result**: Encoder learns phoneme → feature mapping

#### Phase 2: Chinese Pre-training (ACE-Opencpop)
- **Data**: ~130 hours of Chinese singing (30 singers, 106k segments)
- **Full model**: Encoder + Flow Matching + F0 Predictor + Variance
- **Steps**: ~100k
- **LR**: 2e-4 with 4000-step warmup → cosine decay
- **Purpose**: Learn general singing synthesis with sufficient multi-speaker data

#### Phase 3: Japanese Fine-tuning
- **Data**: Kiritan (~1h) + Ofuton-P (~1h) + GTSinger-ja
- **Strategy**: Lower LR (5e-5), partial encoder freeze (first 3 layers)
- **Steps**: ~50k
- **Purpose**: Adapt to Japanese phonology

#### Phase 4: Rap Augmentation (optional)
- **Data**: Add Japanese speech data with singing_ratio = 0
- **Steps**: ~20k
- **Purpose**: Enable speech/rap mode

### 5.2 Loss Function

```
L_total = λ_flow × L_flow + λ_f0 × L_f0 + λ_var × L_var + λ_ms × L_ms

where:
  L_flow = ||v_θ(x_t, t, c) - (x_1 - x_0)||²     (OT-CFM velocity MSE)
  L_f0   = |log F0_pred - log F0_gt|_voiced         (log-F0 L1, voiced frames only)
           + 0.5 × BCE(voicing_pred, voicing_gt)    (voicing classification)
  L_var  = ||energy_pred - energy_gt||²             (per variance dimension)
  L_ms   = Σ_s L_flow(avg_pool_s(target), avg_pool_s(condition))  (multi-scale)
```

Default weights: λ_flow = 1.0, λ_f0 = 1.0, λ_var = 0.2, λ_ms = 0.3

### 5.3 Data Augmentation

| Augmentation | Method | Probability | Effective Multiplier |
|-------------|--------|------------|---------------------|
| Pitch shift | ±12 semitones (shift MIDI + F0, keep HuBERT) | 50% | ~12x |
| Time stretch | ×0.9-1.1 (resample durations + features) | 30% | ~1.3x |
| Noise injection | SNR 20-40dB on HuBERT features | 10% | ~1.1x |

**Pitch shift** is the most powerful augmentation because HuBERT features are largely pitch-invariant (content encoding). We shift the input (MIDI pitch, F0 target) while keeping the HuBERT target unchanged, teaching the model that the same phonetic content can exist at different pitches.

### 5.4 Regularization

- **Gradient clipping**: max norm = 1.0
- **Weight decay**: 0.01 (AdamW)
- **Dropout**: 0.1 (all layers)
- **EMA**: decay = 0.9999 (use for inference)
- **Mixed precision**: FP16 training for memory efficiency

---

## 6. Dataset Details

### 6.1 ACE-Opencpop (Pre-training, recommended)

| Property | Value |
|----------|-------|
| Language | Chinese (Mandarin) |
| Total duration | ~130 hours |
| Segments | ~106k |
| Speakers | 30 (synthesized via ACE Studio with manual tuning) |
| Sample rate | 48kHz → 16kHz (for HuBERT) |
| Annotations | Phone-level: phonemes, note MIDI, note timing, lyrics, tempo |
| Format | HuggingFace datasets (espnet/ace-opencpop-segments) |
| License | CC BY-NC 4.0 |

### 6.1b Opencpop (Pre-training, smaller alternative)

| Property | Value |
|----------|-------|
| Language | Chinese (Mandarin) |
| Total duration | ~5.2 hours |
| Utterances | 3756 |
| Speaker | 1 female |
| Sample rate | 24kHz (original) → 16kHz (for HuBERT) |
| Annotations | Phone-level: phonemes, note pitch, note duration, slur flag |
| Format | WAV + custom transcription format |
| License | CC BY-NC 4.0 |

### 6.2 Kiritan (Fine-tuning)

| Property | Value |
|----------|-------|
| Language | Japanese |
| Total duration | ~1 hour |
| Speaker | 1 (Tohoku Kiritan character voice) |
| Annotations | HTS mono labels (phone boundaries in 100ns) |
| Format | WAV + HTS labels + optional MusicXML |
| License | Free for non-commercial use |

### 6.3 Ofuton-P (Fine-tuning)

| Property | Value |
|----------|-------|
| Language | Japanese |
| Total duration | ~1 hour |
| Annotations | HTS labels |
| License | Free for non-commercial use |

### 6.4 GTSinger (Fine-tuning)

| Property | Value |
|----------|-------|
| Language | Multi-lingual (Japanese subset used) |
| Annotations | JSON per utterance (phones, notes, durations) |
| Format | WAV + JSON |
| Source | HuggingFace: GTSinger/GTSinger |

### 6.5 Cross-Lingual Transfer Justification

Pre-training on Chinese (Opencpop) and fine-tuning on Japanese is motivated by:

1. **HuBERT is language-agnostic**: Self-supervised features encode universal phonetic/prosodic information, not language-specific patterns.

2. **Shared phonetic inventory**: Chinese and Japanese share several phonemes (vowels a/i/u/e/o, many consonants). The encoder's phoneme → feature mapping transfers partially.

3. **Singing universals**: Musical concepts (note pitch, duration, vibrato, portamento) are language-independent. The F0 predictor and flow matching decoder learn singing physics, not language.

4. **Published validation**: Research has demonstrated successful Chinese → Japanese SVS transfer learning (e.g., ACE-Opencpop → Kiritan).

---

## 7. Inference Pipeline

### 7.1 Standard Inference

```python
# Input from DAW
phonemes = ["ja_k", "ja_o", "ja_n", "ja_n", "ja_i", "ja_ch", "ja_i", "ja_w", "ja_a"]
pitches  = [60, 60, 62, 62, 64, 64, 64, 62, 62]  # MIDI
durations = [30, 30, 30, 30, 30, 30, 30, 30, 60]  # frames @ 100fps

# Score2HuBERT inference (~50ms for 3 seconds of audio on RTX 3080 Ti)
hubert_features, f0 = model.synthesize(phonemes, pitches, durations)

# Direct RVC decoder bypass (~20ms)
audio = rvc_net_g.infer(hubert_features, f0, speaker_id)
```

Total inference: **~70ms** for 3 seconds of audio (vs. ~5-15 seconds with old pipeline).

### 7.2 Chunked Generation (Long Sequences)

For sequences longer than ~12 seconds (1200 frames):
1. Split into 500-frame chunks with 50-frame overlap
2. Generate each chunk independently
3. Cross-fade overlap regions (linear blend)

### 7.3 User Pitch Curve Override

When the user draws a pitch curve in the DAW (UTAU/SynthV style):
1. The drawn curve bypasses the F0 predictor entirely
2. Passed directly as `pitch_curve` parameter
3. Model generates HuBERT features conditioned on the user's F0

This is critical for the "tuning" workflow that vocal synth users expect.

---

## 8. Integration with Utai DAW

### 8.1 Proposed API Changes

Replace the current two-step synthesis API:

```
# OLD: Score → DiffSinger → Audio → RVC → Audio
POST /api/synth/render/content  → UtaiRenderer.exe → WAV → RVC → WAV

# NEW: Score → Score2HuBERT → RVC decoder → Audio
POST /api/synth/render/content  → Score2HuBERT → RVC direct → WAV
```

### 8.2 Backend Integration Points

The `synth_wrapper.py` would be extended with a `Score2HuBERTBackend`:

```python
class Score2HuBERTBackend:
    def render(self, ustx_content: str, rvc_model: str) -> str:
        project = parse_ustx(ustx_content)
        phonemes, pitches, durations = extract_score_features(project)
        hubert, f0 = self.model.synthesize(phonemes, pitches, durations)
        audio = self.rvc_decode(hubert, f0, rvc_model)
        return save_wav(audio)
```

### 8.3 Real-Time Preview Potential

With ~70ms inference for 3 seconds of audio, near-real-time preview becomes feasible:
- User draws a note → preview in <100ms
- Note-level caching: only re-render changed notes
- Background pre-rendering of surrounding context

---

## 9. Known Limitations & Future Work

### 9.1 Current Limitations

1. ~~**Single-speaker pre-training**~~: Resolved by using ACE-Opencpop (30 speakers, ~130h).

2. **No phoneme duration prediction**: Currently requires explicit durations from the score. Adding a duration predictor would enable text-only input.

3. **No explicit vibrato modeling**: Vibrato is implicitly captured in the F0 predictor. Explicit vibrato parameters (rate, depth, shape) would give users finer control.

4. **RVC decoder is frozen**: The decoder isn't fine-tuned for our generated features. Joint fine-tuning could improve quality.

### 9.2 Future Directions

1. **Multi-language expansion**: Add English, Korean by simply fine-tuning on respective datasets.

2. **Streaming inference**: Generate features frame-by-frame for real-time synthesis.

3. **Expression transfer**: Learn to transfer expression (dynamics, vibrato, timing) from a reference performance to a new score.

4. **Score-conditioned voice conversion**: Use the Score2HuBERT features as a better conditioning signal for RVC, enabling pitch-accurate voice conversion.

5. **End-to-end joint training**: Train Score2HuBERT and a lightweight decoder jointly, potentially eliminating the need for a separate RVC model.
