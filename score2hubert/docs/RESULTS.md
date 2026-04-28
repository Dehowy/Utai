# Score2HuBERT Training Results

This document records training experiments, observed metrics, and analyses.
Each "run" section is a complete training attempt with configuration and outcomes.

---

## Run 1: Phase 1 Pre-training (IPA + 90M model)

### Setup
- **Date**: 2026-04-08
- **Hardware**: NVIDIA RTX 3080 Ti (12GB), Windows 11
- **Dataset**: ACE-Opencpop, ~120h, 30 singers, 100,000 valid samples
  - Train: 90,000 / Val: 5,000 / Test: 5,000
- **Phoneme vocabulary**: IPA (80 tokens = 66 IPA + 7 special + 7 padding)
- **Model size**: 90.5M params
  - Encoder: 4-layer Conformer, d_model=512, d_ff=2048 (~37M)
  - Variance Adaptor: 1.4M
  - F0 Predictor: 6.6M
  - Flow Matching Decoder: 4-layer DiT, d_hubert=768, d_ff=2048 (~45M)
  - F0 condition projection: 0.1M
- **Training**: 50,000 steps, batch_size=16, lr=2e-4 (cosine warmup 2000)
- **Loss config**: flow_matching=1.0, f0=1.0, energy=0.1, voicing=0.05, **multiscale_hubert disabled**
- **Optimization**: AdamW(0.9, 0.98), EMA(0.9999), FP16 mixed precision, grad_clip=1.0

### Key Architectural Decisions

1. **IPA phoneme vocabulary** instead of language-specific tokens — eliminates cross-lingual transfer problem at the cost of needing G2P mapping tables. Pinyin finals decompose into atomic IPA segments (e.g., `iang` → `[j, a, ŋ]`) with proportional duration split.

2. **Reduced model size** (90M from original 128M) — encoder 4L instead of 6L, flow 4L instead of 6L. The original 6L was over-provisioned for our task complexity. Original architecture doc claimed 62M but the actual 6L config built 128M (the doc was wrong about per-layer parameter counts).

3. **Disabled multiscale loss** — saves ~50% compute for marginal benefit. This single change moved estimated training time from 5+ days to ~14 hours.

4. **HuBERT 50fps** (not 100fps as originally documented) — corrected to match HuBERT-Base's actual frame rate (16kHz / 320 internal hop stride = 50fps). F0 is extracted at ~100fps then resampled to match.

5. **Stochastic ground-truth voicing** during training — F0 predictor uses GT voicing mask (teacher forcing) instead of hard-thresholding predicted voicing, which preserves gradient flow.

### Training Speed

- **Actual: ~0.86 s/step** at batch_size=16 on RTX 3080 Ti (better than the 1.0s/step benchmark)
- **Total wall-clock time: 16.7 hours** (03:34 → 20:15 local time)
- VRAM peak: ~6.7 GB (less than half of 12 GB available)
- GPU utilization: 60-70%

### Training Curves

| Step | train/total | train/flow | train/f0 | val/total | val/f0_mae |
|---|---|---|---|---|---|
|     50 | 1.951 | 1.107 | 0.490 | — | — |
|    250 | 1.246 | 0.843 | 0.186 | — | — |
|    500 | 0.754 | 0.468 | 0.116 | — | — |
|   1000 | 0.571 | 0.341 | 0.086 | 1.963 | 876 cents |
|   2000 | 0.521 | 0.319 | 0.076 | 1.943 | 871 cents |
|   3000 | 0.460 | 0.277 | 0.072 | — | — |
|   5000 | 0.396 | 0.246 | 0.063 | — | — |
|   6000 | 0.367 | 0.235 | 0.061 | 1.692 | 876 cents |
|  10000 | 0.330 | 0.210 | 0.057 | — | — |
|  12000 | 0.304 | 0.200 | 0.054 | 1.196 | 859 cents |
|  19000 | 0.279 | 0.196 | 0.049 | 0.636 | 347 cents |
|  25000 | 0.275 | 0.194 | 0.046 | 0.436 | 214 cents |
|  31000 | 0.264 | 0.188 | 0.044 | 0.329 | 140 cents |
|  39000 | 0.250 | 0.176 | 0.042 | 0.260 | 77 cents |
|  44000 | 0.249 | 0.180 | 0.041 | 0.241 | 58.6 cents |
|  48000 | 0.258 | 0.186 | 0.041 | 0.231 | 53.1 cents |
|  49000 | 0.252 | 0.180 | 0.041 | 0.234 | 52.8 cents |
| **50000** | **0.247** | **0.178** | **0.041** | **0.232** | **52.4 cents** |

#### Key observations from the curves

- **Flow loss**: 1.107 → 0.178 (6.2× reduction). Sharpest drop in steps 100-1000, then steady decline.
- **F0 loss**: 0.490 → 0.041 (12× reduction). Converged faster than flow loss (typical).
- **Validation lag from EMA**: Initial val_loss was much higher than train (1.96 vs 0.57 at step 1000) because EMA (decay 0.9999) takes ~10k steps to fully refresh. By step 19k, val started reflecting actual model quality.
- **Final convergence**: train_loss 0.247 vs val_loss 0.232 — val is slightly LOWER than train, indicating the model is well-regularized (no overfitting). The narrower val/train gap confirms EMA is stable and we're not overfitting on the 90k training samples.
- **F0 accuracy**: 52.4 cents median error ≈ half a semitone, which is competitive with dedicated F0 prediction models.

### Inference Quality Evaluation

#### Test on 20 random test-split samples (n_steps=10, midpoint, cfg=2.0)

| Metric | Value |
|---|---|
| HuBERT MSE (vs GT) | 0.169 ± 0.020 |
| HuBERT cosine sim (vs GT) | 0.191 ± 0.058 |
| HuBERT std ratio (pred/gt) | **0.965** |
| F0 median cents error | **22.0 ± 9.4** |

**Interpretation:**
- **HuBERT std ratio 0.965** is excellent — generated features have nearly the same variance as ground truth, meaning the model is producing in-distribution features.
- **F0 median error 22 cents** is excellent (about 1/5 of a semitone), well within musical perception threshold.
- **HuBERT cosine sim 0.19** is **expected** for a stochastic generative model. OT-CFM samples from a conditional distribution `p(HuBERT | score)`. Since singing has natural variation (a singer can produce slightly different HuBERT features for the same score), a single sample won't match a specific GT exactly. The relevant question is whether the samples are "in-distribution," which the std ratio confirms.
- **HuBERT MSE 0.169** is moderate. Comparable models in literature don't usually report this metric directly because it conflates "wrong content" with "stochastic variation."

#### Inference parameter sweep (5 samples)

| n_steps | solver | cos sim | MSE |
|---|---|---|---|
| 4 | euler | **0.235** | **0.113** |
| 10 | midpoint | 0.159 | 0.166 |
| 20 | midpoint | 0.166 | 0.164 |
| 10 | rk4 | 0.170 | 0.161 |

Surprisingly, **fewer steps work better** for this metric. Possible explanations:
1. With fewer steps, the integration is coarser and acts more like averaging (closer to conditional mean)
2. With more steps, the trajectory is more faithful to the velocity field but produces more "individual" stochastic samples
3. CFG scale 2.0 may be amplifying noise direction with more steps

Recommendation: **default to n_steps=4 with euler** for inference (also fastest).

#### CFG and temperature sweep (5 samples, midpoint, n_steps=10)

| cfg_scale | temperature | cos | MSE |
|---|---|---|---|
| 1.0 | 0.5 | 0.185 | 0.270 |
| **1.0** | **1.0** | **0.193** | **0.130** |
| 1.5 | 0.5 | 0.195 | 0.287 |
| 1.5 | 1.0 | 0.164 | 0.151 |
| 2.0 | 0.5 | 0.206 | 0.289 |
| 2.0 | 1.0 | 0.165 | 0.160 |
| 3.0 | 0.5 | 0.198 | 0.330 |
| 3.0 | 1.0 | 0.144 | 0.192 |

**Best by MSE: cfg=1.0, temp=1.0** (no CFG guidance, full noise temperature)

The fact that CFG doesn't help here suggests:
- The model has learned the score conditioning well enough that explicit guidance is unnecessary
- Or our CFG dropout rate (0.15) during training was too high, weakening the unconditional pass
- Or the metric (cos sim with single GT sample) doesn't reward CFG benefits

#### End-to-end audio generation (12 files in `test_outputs/`)

For 3 test samples, generated 4 audio variants each:
- `sample{N}_gt_ceiling.wav` — RVC quality ceiling: GT HuBERT + GT F0
- `sample{N}_pred_full.wav` — Full Score2HuBERT: predicted HuBERT + predicted F0
- `sample{N}_h_pred_f0gt.wav` — Ablation: predicted HuBERT + GT F0 (isolates HuBERT quality)
- `sample{N}_hgt_f0pred.wav` — Ablation: GT HuBERT + predicted F0 (isolates F0 quality)

These should be auditioned manually to assess perceptual quality. The numerical metrics above don't capture subjective quality.

### Analysis

#### What worked well

1. **Training stability**: Loss curves are smooth, no divergence, no overfitting. The 90M model with 90k training samples was a good size match.
2. **F0 prediction**: 52.4 cents val MAE → 22 cents median test error. The dual-mode F0 predictor with note pitch conditioning learned music-aware pitch dynamics.
3. **Feature distribution matching**: std ratio 0.965 means generated features look statistically like real HuBERT features.
4. **IPA decomposition**: Worked seamlessly. The Chinese pinyin → IPA segments (e.g., `iang` → `[j, a, ŋ]`) with duration splitting integrated cleanly into the training pipeline.
5. **Speed**: 0.86 s/step with batch 16 and 90M params was much faster than initial estimates. The reduction from 6L to 4L + disabling multiscale was the key win.

#### What needs investigation

1. **Cosine similarity is low (0.19)**: Need to understand if this is OK (because OT-CFM is stochastic) or indicates a real problem. The audio test will help clarify.

2. **CFG doesn't help**: Surprising — usually CFG improves conditional generation. Possible the model is already "confident enough" without it, or our CFG drop_prob 0.15 was too aggressive.

3. **Few-step inference is better numerically**: We need to verify this holds in audio quality. If true, this is a great speed win.

4. **EMA decay 0.9999 is too slow** for 50k-step runs. With only 50k steps, EMA doesn't fully catch up until ~30k. Should use 0.999 or 0.9995 for shorter runs to get meaningful validation earlier.

#### What could be improved

1. **Learning rate schedule**: Cosine warmup-decay reached min_lr around step 35k. Loss was still decreasing slightly at the end. Could benefit from a higher final LR or longer training.

2. **Variance adaptor not fully utilized**: Energy/breathiness losses had weight 0.1 and 0.0 respectively. Could enable breathiness for better expressiveness control.

3. **Singing/rap continuum not exercised**: All training data was singing (singing_ratio=1.0). The rap mode is untrained.

### Audio Quality Assessment (subjective listening)

**Listened to all 12 generated WAV files. Key findings:**

| Variant | Quality Description |
|---|---|
| `gt_ceiling` (GT HuBERT + GT F0) | Reference quality through RVC. Clear lyrics, natural singing. |
| `hgt_f0pred` (GT HuBERT + **pred** F0) | **Approaches ceiling quality.** Occasional unclear pronunciation but mostly indistinguishable from ceiling. F0 prediction is excellent. |
| `h_pred_f0gt` (**pred** HuBERT + GT F0) | **No lyrics at all.** Sounds like noisy humming with vague articulation. Melody is correct (because F0 is GT) but phonetic content is completely lost. |
| `pred_full` (pred HuBERT + pred F0) | Similar to `h_pred_f0gt`. Correct melody, no clear lyrics, layer of noise. |

### Critical Diagnosis: HuBERT Content Loss

**The model has learned the marginal HuBERT distribution but failed to learn the conditional structure (score → content mapping).**

Evidence:
1. **F0 prediction is excellent** (`hgt_f0pred` ≈ ceiling). No work needed here.
2. **Predicted HuBERT contains no phonetic content** (`h_pred_f0gt` has no lyrics). The flow decoder produces features that look statistically like HuBERT (std ratio 0.965) but they're essentially random samples from the marginal distribution, not conditional on the score.
3. **CFG sweep showed conditional/unconditional collapse**: cfg_scale=1.0 and cfg_scale=2.0 produce nearly identical results. This means `v_cond ≈ v_uncond`, i.e., the model learned essentially the same thing in both modes — the unconditional pass dominates.

**Mechanism (best hypothesis):** With CFG drop_prob=0.15 and a flow matching loss that only constrains feature statistics (not specific content), the model found a "cheating path":
- Ignore phoneme content from the encoder
- Use F0 to control pitch (via the explicit F0 condition projection)
- Generate features from a learned marginal distribution that looks like "average singing voice"

This produces the symptoms we observe: correct melody (from F0), HuBERT-like statistics (from learned marginal), but no phonetic content (because phoneme conditioning was bypassed).

### Plan A: Force conditional learning (next experiment)

Three changes to `flow_matching.py` and `score2hubert.py`:

1. **Reduce CFG drop_prob from 0.15 → 0.05**
   - Less unconditional training, more conditional training
   - Forces the model to use the encoder output

2. **Selective CFG dropping** — when zeroing condition for CFG, only zero the encoder/phoneme part. **Keep F0 embedding** in the unconditional path.
   - F0 is a "hard" user-provided constraint, not something CFG should guide
   - This makes CFG specifically learn to amplify phoneme content, which is what we want

3. **Add reconstruction loss** alongside flow matching loss:
   - Run a single Euler step from x_t to predict x_1
   - Compute L1 + (1 - cosine_sim) on this predicted HuBERT vs GT HuBERT
   - This directly supervises content fidelity, not just feature distribution
   - Loss weight: maybe 0.5 (auxiliary, not primary)

**Expected outcome:** The model is forced to use the encoder condition strongly, breaking the "marginal distribution shortcut". Should produce HuBERT features that contain phonetic content.

**Resume strategy:** Resume from `checkpoints/pretrain_opencpop/best.pt` (step 48000). The basic feature distribution is learned; we just need to redirect the loss landscape toward conditional learning. Expected additional training: 15-25k steps (~5-8 hours).

### What's NOT the problem (ruled out)

- **Encoder capacity**: 4 layers is enough. The phoneme info is *available* at the encoder output; the issue is downstream.
- **Model size**: Larger model wouldn't fix this — would just learn the same shortcut more efficiently.
- **F0 prediction**: Working great. Don't touch.
- **IPA decomposition**: Working as designed. Phoneme tokens are correctly assembled.
- **Dataset**: 100k samples, balanced across 30 singers. Plenty of data.
- **Frame alignment**: Verified clean (HuBERT/F0/duration all match per sample).

### Rejected alternative plans

- **Plan C (just make model bigger)**: Wouldn't fix the root cause. The model is using the wrong learning signal, not running out of capacity.
- **Plan D (change loss formulation, e.g., x0-prediction)**: High risk, requires rewriting trainer. Plan A targets the diagnosed issue more surgically.

### Discovered bug: synthesizer.py RVC interface

Found during inference testing: the existing `src/inference/synthesizer.py:_rvc_decode()` function:
1. Upsamples HuBERT features 2x via `F.interpolate(scale_factor=2)` — this is incorrect, RVC's encoder expects feat and pitch at the same length
2. Permutes feat to `[B, 768, T]` before passing to `net_g.infer()` — this is also wrong, RVC expects `[B, T, 768]`

**Working interface** (verified):
```python
# feat: [B, T, 768], NOT permuted, NOT 2x-upsampled
# f0: [B, T], same length as feat
audio = net_g.infer(feat, phone_lengths, f0_coarse, f0_hz, sid)[0][0,0]
```

The synthesizer.py needs to be patched accordingly. This bug means inference via the official synthesizer wrapper would have failed; the test in `test_outputs/` used a corrected ad-hoc implementation.

---

## Run 2: Plan A — selective CFG + reconstruction loss

### Setup
- **Date**: 2026-04-08 (start: 22:08), launched after Run 1 audio analysis
- **Hardware**: Same as Run 1 (RTX 3080 Ti, Windows 11)
- **Dataset**: Same — ACE-Opencpop, IPA preprocessing already done
- **Resume from**: `checkpoints/pretrain_opencpop/best.pt` (Run 1 step 48000, val 0.231)
- **Target**: Step 48000 → 75000 (~27k more steps)

### Pipeline optimizations applied between Run 1 and Run 2

Before launching Plan A, profiled and fixed GPU under-utilization:
- **`num_workers: 0 → 4`** (the "Windows hangs" myth was busted — workers spawn fine, the original 'hang' was just startup latency)
- **`persistent_workers: true`, `prefetch_factor: 4`** — keeps the worker pool alive between epochs and buffers ahead
- **`batch_size: 16 → 32`** — verified safe (batch 48 spilled into Windows shared memory, batch 32 peaks at 7.7 GB allocated)

Result: step time **749ms → 281ms (5.3x throughput improvement)** at clean profile, GPU util 27% → 95%. See `scripts/profile_step.py` for the per-phase breakdown tool.

### Code changes for Plan A

Three surgical changes targeting the diagnosed marginal-vs-conditional learning failure:

1. **CFG drop_prob 0.15 → 0.05** (`configs/pretrain_opencpop.yaml`)
   - Less unconditional training; more samples force the model to use the encoder.

2. **Selective CFG dropping** (`src/model/score2hubert.py:forward`)
   - CFG drop now zeros only the encoder phoneme content, NOT the F0 embedding.
   - Removed the CFG drop logic from `flow_matching.py:compute_loss` — caller handles it.
   - Rationale: F0 is a "hard" user-given condition; CFG should learn to amplify phoneme content (the part that was being ignored), not F0 (which is correct).

3. **Reconstruction loss** (`src/model/score2hubert.py:forward`)
   - Single Euler step from `x_t` to estimate `x_1` using the velocity already computed in `compute_loss`. No extra forward pass — `flow_matching.compute_loss` now also returns `pred_v, t, x_t` for this purpose.
   - Loss: `L1(x_1_pred, hubert_target) + 0.5 * (1 - cosine_sim(x_1_pred, hubert_target))`
   - Weight: 0.5 in total loss
   - Directly supervises content fidelity, breaking the "marginal distribution shortcut".

### Hyperparameter changes for resume

| Field | Run 1 final | Run 2 (Plan A) | Reason |
|---|---|---|---|
| `lr` | 2e-4 | 1.5e-4 | Resuming from converged ckpt; gentler steps |
| `warmup_steps` | 2000 | 1000 | Short warmup for resume |
| `total_steps` | 50000 | 75000 | Extended by 25k for fine-tuning |
| `cfg.drop_prob` | 0.15 | 0.05 | Force conditional learning |
| `loss.reconstruction` | (not present) | 0.5 | New auxiliary loss |
| `batch_size` | 16 | 32 | GPU optimization |
| `num_workers` | 0 | 4 | GPU optimization |

### Trainer fix discovered during smoke test

`scheduler.load_state_dict()` includes `base_lrs`, which silently overrides the new optimizer lr from config. Without an override, the new `lr: 1.5e-4` was being ignored and the old `lr: 2e-4` was applied with the new schedule (giving 1.33x higher effective lr than intended). Fixed in `trainer.py:_load_checkpoint` — now logs `Overriding scheduler base_lr [...] -> [new]` when this happens.

### Training Speed

- **Actual: ~0.49 s/step** at batch_size=32 on RTX 3080 Ti (after GPU optimization)
- **Total wall-clock time: 3h 41min** (22:08 → 01:49, ~27k steps)
- VRAM peak: ~7.7 GB allocated (nvidia-smi showed ~12 GB used incl. PyTorch cache, close to ceiling but no OOM)

### Training Curves (actual)

| Step | train/total | train/flow | train/f0 | train/recon_l1 | train/recon_cos | val/total | val/f0_mae |
|---|---|---|---|---|---|---|---|
| 48050 | 0.373 | 0.180 | 0.041 | 0.122 | 0.751 | — | — |
| 49000 | 0.377 | 0.188 | 0.041 | 0.118 | 0.762 | 0.2333 | 53.9 |
| 50000 | 0.372 | 0.183 | 0.041 | 0.119 | 0.762 | 0.2378 | 54.7 |
| 51000 | — | — | — | — | — | 0.2329 | 52.7 |
| 52000 | — | — | — | — | — | 0.2317 | 51.4 |
| **53000** | 0.377 | 0.188 | 0.040 | 0.117 | 0.767 | **0.2280** | 52.2 |
| 54000 | — | — | — | — | — | 0.2314 | 50.9 |
| 55000 | — | — | — | — | — | 0.2291 | 51.1 |
| 56000 | 0.367 | 0.181 | 0.040 | 0.118 | 0.754 | 0.2314 | 49.9 |
| 57000 | — | — | — | — | — | 0.2309 | 50.0 |
| 58000 | — | — | — | — | — | 0.2291 | 50.0 |
| **59000** | — | — | — | — | — | **0.2267** | 48.8 |
| 60000 | 0.372 | 0.182 | 0.040 | 0.120 | 0.751 | 0.2273 | 49.8 |
| 61000 | — | — | — | — | — | 0.2296 | 49.5 |
| 62000 | — | — | — | — | — | 0.2290 | 49.8 |
| **63000** | — | — | — | — | — | **0.2237** | 49.0 |
| 64000 | 0.366 | 0.181 | 0.039 | 0.118 | 0.763 | 0.2274 | 49.2 |
| **65000** | — | — | — | — | — | **0.2225** | 49.2 | ← best.pt saved |
| 66000 | — | — | — | — | — | 0.2273 | 48.6 |
| 67000 | — | — | — | — | — | 0.2258 | 48.9 |
| 68000 | 0.366 | 0.184 | 0.039 | 0.118 | 0.768 | 0.2246 | **48.3** |
| 69000 | — | — | — | — | — | 0.2258 | 48.1 |
| 70000 | — | — | — | — | — | 0.2271 | 48.0 |
| 71000 | — | — | — | — | — | 0.2234 | **47.8** |
| 72000 | 0.363 | 0.179 | 0.039 | 0.119 | 0.762 | 0.2237 | 47.6 |
| 73000 | — | — | — | — | — | 0.2251 | 47.5 |
| 74000 | — | — | — | — | — | 0.2229 | 47.5 |
| **75000** | **0.367** | **0.181** | **0.039** | **0.118** | **0.758** | **0.2270** | **47.3** |

#### Key observations from the curves

- **val_loss drop**: 0.2309 (Run 1 final, no recon term) → 0.2225 (Plan A best, includes recon contribution) — a net improvement despite the new term in the total. Best at step 65000.
- **f0_mae drop**: 53.1 → 47.3 cents (−10.9%). Continued improving to the very last step.
- **Train loss flat**: `flow` stayed around 0.18, `recon_cos` oscillated 0.75-0.77 with no clear trend. The numerical signals are subtle — real progress should show in the audio.
- **EMA catchup effect**: The first ~2k steps after resume showed val REGRESSION (0.2333 → 0.2378) before recovery. This matches the EMA decay time constant (0.9999 → ~10k steps to catch up). The real Plan A signal became visible around step 53000+.

### Audio Quality Assessment

Eval audio generated with `scripts/eval_audio.py --seed 42 --min-frames 350`:

- **Run 1 long baseline**: `test_outputs_run1_long/` (generated 2026-04-08 22:10 from `best_run1.pt`)
- **Plan A output**: `test_outputs_plan_a/` (generated 2026-04-09 01:55 from new `best.pt` at step 65000)

Both use the **same 3 samples** (seed 42 → identical picks):
- sample1: `ace_acesinger_26#2017000671` (393 frames, ~7.9s)
- sample2: `ace_acesinger_12#2015000587` (381 frames, ~7.6s)
- sample3: `ace_acesinger_19#2019000752` (376 frames, ~7.5s)

For each sample, 4 ablations:
- `gt_ceiling`  — GT HuBERT + GT F0 (RVC ceiling)
- `pred_full`   — pred HuBERT + pred F0 (full Score2HuBERT)
- `h_pred_f0gt` — pred HuBERT + GT F0 (isolates HuBERT quality)
- `hgt_f0pred`  — GT HuBERT + pred F0 (isolates F0 quality)

### Listening test outcome (2026-04-09)

**F0 pathway: PERFECT.** `hgt_f0pred` is effectively indistinguishable from `gt_ceiling` across all 3 samples. The F0 predictor has reached its ceiling.

**HuBERT content: STILL COMPLETELY BROKEN.** Both `pred_full` and `h_pred_f0gt` sound like the model is vocalizing "啊啊啊" (generic open-vowel sounds) with some noise. Phonetic content (consonants, specific vowels, lyric identity) is absent. The model has NOT learned to use phoneme conditioning.

There is a subtle improvement over Run 1 — slightly cleaner edges, slightly less muddy — but this is explainable entirely by:
1. F0 more accurate → rhythm and pitch feel cleaner
2. Marginal HuBERT distribution got polished further → less noise
3. More total training steps → general convergence cleanup

**None of these represent actual content learning.** The `recon_cos` train metric sitting flat at 0.758 ± 0.01 across 27,000 steps was the honest signal: zero content improvement.

### Diagnosis: architectural shortcut in DiTBlock

Inspecting `src/model/modules.py:226-276` (`DiTBlock.forward`):

```python
h = self.adaln_self(x, time_emb)     # AdaLN conditions on TIME ONLY
x = x + self.self_attn(h, mask=mask)

h = self.adaln_cross(x, time_emb)    # AdaLN conditions on TIME ONLY
x = x + self.cross_attn(h, mask=mask, kv=cond)   # ← condition enters HERE AND ONLY HERE

h = self.adaln_ff(x, time_emb)       # AdaLN conditions on TIME ONLY
x = x + self.ff(h)
```

All three `AdaLN` modules modulate activations based on `time_emb`, not `cond`. The encoder-side condition (phoneme + F0) enters the flow decoder **only** through one `cross_attn` call per block.

Cross-attention is the weakest form of conditioning — the model can learn to place near-zero attention weight on the kv sequence, effectively setting the contribution to zero. And our model apparently did exactly that, because:
- `recon_cos` never moved past ~0.76 (the ceiling achievable by ignoring phoneme content and using F0 + marginal distribution)
- Plan A's selective CFG and reconstruction loss, which operate at the loss level, couldn't compensate for the architectural shortcut

On top of that, `AdaLN.__init__` uses `nn.init.zeros_` for the projection, meaning all three AdaLN layers start as identity and only the time_emb pathway ever gets an active signal.

**Net result:** The only two signals that could influence the flow decoder are (a) time_emb via AdaLN and (b) condition via cross-attention. The model learned to use (a) for noise-scheduling and F0 embedding propagation, and learned to ignore (b). This left phoneme identity stranded at the encoder output with no reliable path to the HuBERT prediction.

### Next step: Plan B diagnostic

Before committing to a full architectural change, run a 5-10 minute diagnostic: freeze everything except a small deterministic head, and train the head to regress HuBERT directly from `[frame_hidden; f0_emb]`. Script: `scripts/diag_det_head.py`.

- If L1 drops clearly and cos rises → `frame_hidden` contains phoneme info, and Plan B (add cond-AdaLN in DiTBlock) has a real chance
- If L1 and cos barely move → encoder is the problem, need bigger surgery

### Outcome summary

- Numerical: val_loss 0.2309 → 0.2225, f0_mae 53.1 → 47.3 cents
- Perceptual: F0 perfect, HuBERT content unchanged (still "啊啊啊" with no lyrics)
- Verdict: **Plan A succeeded on F0 polish, failed on content learning due to architectural shortcut in DiTBlock that loss-level fixes cannot address**

---

## Plan B/C diagnostic series (2026-04-09 night)

After Plan A failed perceptually, ran two diagnostics in sequence to understand
the fundamental nature of the problem.

### Diagnostic 1: Deterministic head can learn content from frame_hidden

**Setup**: Froze encoder/variance/F0 paths, added a tiny 4.86M parameter
deterministic head taking `[frame_hidden | f0_emb]` and regressing HuBERT via
L1 + 0.5*(1-cos). Trained for 500 steps. Generated audio via det_head → RVC.

**Numerical** (script: `scripts/diag_det_head.py`):

| step | L1 | cos sim | delta L1 | delta cos |
|---|---|---|---|---|
| baseline (random init) | ~0.72 | ~0.01 | — | — |
| 100 | 0.187 | 0.509 | −0.534 | +0.499 |
| 250 | 0.176 | 0.578 | −0.546 | +0.568 |
| 500 | **0.169** | **0.622** | **−0.555** | **+0.612** |

Both checkpoints (Plan A best.pt and Run 1 best_run1.pt) gave nearly identical
curves (cos delta < 0.003), confirming Plan A's training did NOT damage the
encoder's frame_hidden representation.

**Perceptual** (output: `test_outputs_diag_det_plan_a/`, `test_outputs_diag_det_run1/`):

The "啊啊啊" noise from Plan A is GONE. Articulation begins to form — still
blurry, but users with prior knowledge of the song can identify specific
lyrics (sample2 was identifiable as singing "若不是你突然闯进我生活，我怎会
把死守的寂寞放任了" from 《绿色》).

**Conclusion**: `frame_hidden` DOES contain phoneme-identity information.
The encoder is NOT the bottleneck. The flow decoder is — it has the information
available via cross-attention but architecturally chose to ignore it.

### Diagnostic 2: HuBERT speaker invariance

**Hypothesis** (user, 2026-04-09): HuBERT was designed by its authors to remove
timbre and preserve content. If true, the same lyric sung by different singers
should produce HuBERT features that are very similar — meaning our model's
multi-speaker training data is NOT actually multi-modal at the HuBERT level.
This would mean averaging across speakers is fine, no speaker conditioning
needed.

**Setup** (script: `scripts/diag_hubert_speaker_invariance.py`): ACE-Opencpop
has 3,285 segments where all 30 singers performed the same lyric with
identical n_frames (already temporally aligned). Sampled 100 segments,
computed pairwise cosine similarity across all 30 singers per frame, averaged.

**Results**:

| Measurement | N | mean cos | std | p10 | p50 | p90 |
|---|---|---|---|---|---|---|
| **Cross-singer same-segment** (the hypothesis) | 100 segs | **0.796** | 0.025 | 0.757 | 0.799 | 0.828 |
| **Single singer vs 30-singer mean** (det model ceiling) | 1500 | **0.892** | 0.034 | 0.853 | 0.897 | 0.922 |
| Cross-segment same-singer (random baseline 1) | 500 | 0.141 | 0.054 | 0.077 | 0.137 | 0.206 |
| Cross-singer cross-segment (random baseline 2) | 485 | 0.140 | 0.052 | 0.080 | 0.135 | 0.201 |

**Interpretation**:

1. **HuBERT IS largely speaker-invariant** (cos 0.80 across singers vs cos 0.14
   on random baseline = +0.66 signal). Same lyric across 30 different singers
   produces HuBERT features that are 5.7x more similar to each other than
   to random content.

2. **The two random baselines are essentially identical** (0.141 vs 0.140) —
   shuffling singers barely changes HuBERT cos sim. Variance is almost entirely
   from content, not from speaker.

3. **Deterministic regression has cos ceiling ~0.89** — if our det head perfectly
   learned the average HuBERT features for given content, it would achieve cos
   sim ~0.89 against any individual singer. So the diagnostic det head's 0.62
   has 0.27 of headroom. A bigger head + longer training should approach 0.89.

4. **HuBERT is content-biased but not 100% content-only**: cos 0.80 is high
   but not 0.95+, meaning ~20% of HuBERT's variance still encodes
   speaker/expression. Not a problem for our use case because RVC strips and
   replaces speaker info downstream anyway.

### Crucial insight: cos sim is misleading as a content-quality metric

Comparing the two paths:

| Model | cos sim (vs single GT singer) | Audio quality |
|---|---|---|
| Plan A flow decoder (`recon_cos`) | **0.76** (higher) | "啊啊啊", no content |
| Diagnostic det head (4.86M, 500 steps) | **0.62** (lower) | Blurry but identifiable lyrics |

**The model with HIGHER cos sim has WORSE perceptual quality.** This breaks the
naive assumption that cos sim ≈ content fidelity.

**Explanation**: Cos sim aggregates over all 768 HuBERT dimensions. Phoneme
identity likely lives in a small subspace (maybe 50-150 dimensions). The flow
decoder optimized the high-variance dimensions correctly (speaker timbre, F0
correlates, energy envelope) but missed the low-variance phoneme subspace —
giving high cos but wrong content. The det head's L1 loss treats all 768
dimensions equally, so even at lower overall cos sim, it correctly hits the
phoneme subspace.

**Lesson**: For evaluating content-aware HuBERT prediction, cos sim alone is
not enough. We need either (a) audio listening tests, or (b) a phoneme-classifier
probe on the predicted features that specifically measures the content subspace.

### Why Flow Matching is unsuited for HuBERT prediction (theory)

Combining the diagnostic results with first-principles reasoning:

**HuBERT prediction is mostly a deterministic mapping**:
- HuBERT was designed to be content-discriminative and speaker/style-invariant
- Diagnostic 2 confirmed this empirically: 80% of HuBERT variance is content
- Given a score (phoneme + pitch + duration), the "correct" HuBERT features
  are essentially determined; only ~20% multi-modal variance from residual
  speaker/expression
- The phoneme dimensions in particular are deterministic given the score

**Flow matching is designed for multi-modal distributions**:
- It learns velocity fields that map noise to a distribution
- Stochastic sampling at inference is a feature, not a bug
- It excels when the target has natural variation (e.g., "an image of a cat")
- It struggles when the target is essentially deterministic — sampling
  stochasticity actively fights the deterministic target
- Multiple noise trajectories all leading to the same output is wasted capacity

**The mismatch in our setup**:
- ~80% of the HuBERT prediction task is deterministic (content)
- ~20% is multi-modal (speaker/expression variation)
- Flow matching's strengths apply to the 20%, not the 80%
- And the 80% is what we actually care about (it's the lyrics!)
- Result: FM "succeeds" on the multi-modal global features (it learned speaker
  distribution, F0 contour averages, energy envelopes — which sound like real
  singing voice) but "fails" on the deterministic content
- Recon_cos plateau at 0.76 in Plan A is exactly this: model reached the
  cross-speaker similarity ceiling (0.79 measured) with global features, but
  cannot push higher because phoneme dims are stochastic in its representation

**The right architecture for HuBERT prediction is deterministic regression**:
- L1/L2 loss on full 768 dims directly supervises every dimension
- No stochasticity to fight against
- Limited only by encoder capacity and head size
- Diagnostic 1 directly verified this: a 4.86M deterministic head extracts
  recognizable lyrics where 45M flow decoder produced "啊啊啊"

This is a generalizable insight: **HuBERT space favors regression over generation**
because it's a content-deterministic representation by design.

### Future research direction: hybrid det + flow refiner

If after Plan C the det head's output sounds too "averaged" (e.g., loses
expression detail), the natural next step is a 2-stage architecture:

```
score → encoder → frame_hidden + f0_emb
                       ↓
                  det_head → coarse_hubert (~0.89 cos ceiling, content-correct)
                       ↓
                  flow_decoder → refined_hubert (adds back the 20% multi-modal detail)
                       ↓
                  RVC → audio
```

Stage 1 (det head) handles the 80% deterministic content. Stage 2 (flow
matching) handles the 20% multi-modal expression. Each tool used for what
it's best at. The flow_decoder's existing 75k step training (which learned
the marginal distribution well) actually becomes useful in this role.

This is similar to DiffSinger (mel boundary + diffusion) and NaturalSpeech 2
(predictive backbone + latent diffusion). It's a known-good pattern in TTS/SVS
that we now have empirical justification for in our setting.

For now: ship Plan C alone first. If quality is acceptable, done. If not,
escalate to hybrid.

---

## Run 3: Plan C — Deterministic head training (2026-04-09)

### Setup

- **Date**: 2026-04-09 (start ~04:40, complete 08:16, ~3.6h wall clock including a stop-and-restart due to EMA discovery)
- **Hardware**: Same RTX 3080 Ti (12GB), Windows 11
- **Dataset**: Same ACE-Opencpop, IPA preprocessed
- **Init from**: `checkpoints/pretrain_opencpop/best.pt` (Plan A step 65000) via the new `--init-from` flag (model + EMA shadow only, fresh optimizer/scheduler/step)
- **Architecture**: Plan A architecture + new `DetHubertHead` module
- **Mode**: `model.mode: deterministic` — bypasses flow_decoder entirely in `forward()` and `synthesize()`. flow_decoder kept in state_dict (frozen, excluded from optimizer) for future hybrid training.

### Deterministic head architecture

`src/model/det_head.py`:

```
DetHubertHead:
    in_proj   : Linear(2*512 -> 512)              ~524K
    4x block  : [Conv1d(512, 512, k=7, pad=3)
                 GroupNorm(8, 512)
                 GELU
                 Dropout(0.1)] residual           ~7.34M
    out_proj  : Linear(512 -> 768)                ~393K
    Total: ~8.26M params
```

Input: `concat([frame_hidden, f0_emb])` along feature dim → `[B, T, 1024]`. Output: `[B, T, 768]` HuBERT features. Receptive field: 4 × 7 = 28 frames ≈ 560ms (syllable-level local context). Encoder above already provides global context via Conformer self-attention, so a pure-conv head is sufficient.

**Sizing rationale**: 1.7× the diagnostic head (4.86M, frozen-encoder) which already produced audible lyrics in 100 steps. End-to-end training with adaptive encoder doesn't need a bigger head.

### Code changes for Plan C

| File | Change |
|---|---|
| `src/model/det_head.py` | NEW — DetHubertHead module |
| `src/model/score2hubert.py` | Added `mode` flag; `forward` and `synthesize` now branch on mode; det path bypasses flow_decoder |
| `src/training/trainer.py` | Added `init_from` entry (loads model + EMA shadow only, with `strict=False`); det mode freezes flow_decoder and excludes it from optimizer; best-checkpoint criterion changed from `val_total_loss` to `val_recon_l1` (more reliable content metric) |
| `scripts/train.py` | Added `--init-from` CLI flag (parallel to `--resume`) |
| `configs/pretrain_opencpop_det.yaml` | NEW — Plan C config: mode=deterministic, det_head config, det loss weight, EMA disabled, separate ckpt/log dirs |
| `scripts/eval_audio.py` | Fixed: handle missing keys when loading old flow ckpt into new det-mode model class (det_head missing → kept at fresh init) |

### The EMA pitfall (CRITICAL DISCOVERY — 2026-04-09 ~05:30)

The most important finding of this run, recorded here in detail because it's a non-obvious pathology that future warm-start + fresh-head experiments will likely re-encounter.

**Symptom**: At step 1000-5000, train metrics looked great (recon_l1 0.13, recon_cos 0.82, f0_loss 0.04 — all in line with what a healthy converging model should show), but **val metrics were catastrophically bad**:

| Step | val recon_l1 | val recon_cos | val f0_mae |
|---|---|---|---|
| 1000 | 0.7434 | -0.001 | **104.8 cents** |
| 2000 | 0.5372 | 0.035 | 69.7 |
| 3000 | 0.3902 | 0.085 | 67.4 |
| 4000 | 0.3090 | 0.148 | 76.0 |
| 5000 | 0.2620 | 0.223 | **85.0** |

Two anomalies stood out:
1. val recon_l1 was 5× train recon_l1 — much wider gap than EMA cold start could explain
2. val f0_mae was 2× Plan A's 47.3 cents and **getting worse** as training progressed

`f0_predictor` is a Plan A module that should give Plan A-quality predictions when run with Plan A weights. EMA shadow IS Plan A's converged weights at init time, with only ~10% drift after 1000 steps (0.9999 decay). It should NOT produce 2× worse f0_mae. Something was actively breaking the val pass.

**Diagnostic series** (3 scripts, written and run while training continued in the background):

1. `scripts/diag_ema_load.py` — Verified that `init_from` correctly loads Plan A weights and EMA shadow. With CPU forward, both LIVE and EMA-applied paths gave f0_mae ≈ 47 cents on a 5-batch subset, then ≈ 47 cents on full 5000-sample val with batch=32. **Init_from is correct.**

2. `scripts/diag_trainer_validate.py` — Built a Trainer with init_from, immediately called `_validate()` at step 0 with no training. Result: **f0_mae = 49.24 cents** ✓. The trainer's val pipeline works correctly when there's been no training drift.

3. `scripts/diag_live_vs_ema.py` — **The smoking gun.** Loaded `step_5000.pt` from the running training and ran val twice — once using `model_state` directly (LIVE) and once after replacing param.data with `ema_shadow` (EMA-applied):

| Metric | LIVE (model_state) | EMA-applied (shadow) |
|---|---|---|
| f0_mae | **54.5 cents** | 84.9 cents |
| recon_l1 | **0.1294** | 0.2620 |
| recon_cos_sim | **0.8205** | 0.2230 |
| total_loss | **0.2797** | 0.7272 |

**The live model was healthy. EMA-applied was dysfunctional.** The trainer's val (which applies EMA) and the synthesizer (which prefers ema_shadow on load) were both reading from a broken model. We were training a good model and saving / evaluating a bad one.

### Why it happened (mechanism)

EMA decay 0.9999: `shadow_new = 0.9999 * shadow_old + 0.0001 * live_new`. Standard practice — provides smoothing, makes evaluation stable, gives a "time average" of the converging model.

Our setup violated a hidden assumption of EMA:
- **Encoder/variance/f0 (warm)**: shadow starts at Plan A's converged values
- **det_head (fresh)**: shadow starts at fresh PyTorch init (no Plan A counterpart to load)

Det_head, being fresh, produces near-random output → very large early losses → very large gradients flowing back through `det_head.in_proj` into `frame_hidden` and from there into the encoder. The encoder gets pushed off Plan A's local minimum in directions optimal for the new det objective. **Live encoder drifts noticeably within the first 1-2k steps.**

EMA shadow lags far behind (decay 0.9999 ⇒ ~10k step time constant). At step 1000, shadow ≈ 90.5% Plan A + 9.5% drifted-live for the encoder.

The pathology: **weight-space linear interpolation between two qualitatively different model regimes is not output-space linear interpolation**. Plan A's encoder has its weights arranged to serve the flow_decoder (one specific frame_hidden representation). Drifted-live encoder is mid-transition to serving det_head (a different representation). Their 90/10 mix in weight space lands in a region that serves neither task — the resulting `frame_hidden` is essentially noise from any downstream consumer's perspective.

The downstream consumers:
- **det_head shadow** is also broken (90.5% fresh + 9.5% partly-trained → still mostly random) → val recon_cos ≈ 0
- **f0_predictor live = Plan A weights**, but its **input** (frame_hidden from the broken-mix encoder) is now distributionally OOD → f0 prediction collapses → val f0_mae explodes from 47 → 104

### Why standard EMA is normally safe

In standard fine-tuning (no fresh modules), all params start warm. Early gradients are small (already near a local minimum). Live moves slowly. Shadow tracks closely. The blend at any time is between two **nearby points in a single basin of attraction** — no pathological interpolation, output is a smooth average.

Our scenario specifically violates this: the fresh head provides a CONSTANT source of LARGE gradients for thousands of steps, pulling live params far from their starting basin while shadow remains anchored to the start. The two endpoints of the interpolation are in **different basins**.

This is, as far as I can find, undocumented in the TTS/SVS literature. Most warm-start + new-head papers either skip EMA, use very low decay, or freeze the warm body during initial head training.

### The fix

After diagnosing at step 5000, decided to disable EMA entirely:

1. Waited for step_7500.pt save event (preserves additional ~2500 steps of training progress)
2. Killed the training process (`taskkill /F /PID 16728`)
3. Edited `configs/pretrain_opencpop_det.yaml`: `ema.enabled: false`
4. Verified the resume path with a 10-step smoke test in a temporary ckpt dir
5. Resumed: `python scripts/train.py --config configs/pretrain_opencpop_det.yaml --resume checkpoints/pretrain_opencpop_det/step_7500.pt`

Trainer correctly skipped EMA build (`use_ema = False`), so `_validate()` no longer applied EMA, and `_save_checkpoint` no longer wrote `ema_shadow` to the dict. Synthesizer.py's "prefer ema_shadow" code path naturally falls through to `model_state` when `ema_shadow` is absent, so inference is also clean.

**First val after resume (step 8000)**: `val_l1=0.1253, val_cos=0.834, val_f0_mae=56.7 cents` — instantly aligned with train metrics. The pathology was fully resolved.

### Training trajectory (post-fix, EMA disabled)

| Step | train recon_l1 | train recon_cos | val recon_l1 | val recon_cos | val f0_mae |
|---|---|---|---|---|---|
|  8000 | 0.1278 | 0.825 | 0.1253 | 0.834 | 56.7 |
| 10000 | 0.1268 | 0.829 | 0.1236 | 0.838 | 51.0 |
| 12000 | 0.1249 | 0.833 | 0.1224 | 0.842 | 52.9 |
| 14000 | 0.1241 | 0.838 | 0.1216 | 0.845 | 48.8 |
| 16000 | 0.1226 | 0.842 | 0.1206 | 0.847 | 48.6 |
| 18000 | 0.1219 | 0.845 | 0.1198 | 0.850 | **46.7** ← first time below Plan A's 47.3 |
| 20000 | 0.1215 | 0.845 | 0.1195 | 0.850 | 48.1 |
| 22000 | 0.1210 | 0.847 | 0.1190 | 0.852 | 46.4 |
| 24000 | 0.1207 | 0.849 | 0.1188 | 0.853 | 45.5 |
| 26000 | 0.1204 | 0.851 | 0.1185 | 0.853 | 44.7 |
| 28000 | 0.1201 | 0.852 | 0.1184 | 0.854 | 44.8 |
| **30000** | **0.1199** | **0.854** | **0.1183** | **0.854** | **44.6** |

#### Observations

- **train ≈ val** for all metrics throughout the post-fix phase. Zero overfit. The model is in a healthy regime.
- **val f0_mae 44.6** is **−5.7% better than Plan A's 47.3** — encoder finished adapting to det loss without losing F0 quality (which was the worry).
- **val recon_cos 0.854** is **+12.4% above Plan A's marginal-distribution shortcut value of 0.76**, AND only 0.038 below the theoretical regression ceiling of 0.892 (single-singer-vs-mean from speaker invariance diagnostic). The remaining gap is the irreducible singer-style variance that no deterministic model can predict.
- **val recon_l1 plateau ≈ 0.12** — stopped improving meaningfully after step 24k. Cosine LR schedule is helping squeeze the last bit but returns are diminishing.

### Audio quality (subjective listening)

Generated 5 samples × 4 ablations from final `best.pt` and the equivalent from Plan A's `best.pt` (same seed=42, same min_frames=350, identical sample picks for direct A/B):

- `test_outputs_plan_c_final/sample{1..5}_*.wav` — Plan C
- `test_outputs_plan_a_5samples/sample{1..5}_*.wav` — Plan A reference
- (also exists: `test_outputs_plan_c_step14k/` — interim 2-sample check at step 14000)

**User listening verdict (2026-04-09 morning)**: SUCCESS.

> "清晰可识别歌词 虽然和ceiling还有一些细微差距 但是也只是细微差距了"

`pred_full` and `h_pred_f0gt` from Plan C produce **clearly recognizable lyrics**, where Plan A's same files were "啊啊啊" open-vowel vocalizations with no phonetic content. The remaining gap to `gt_ceiling` is small enough to be characterized as "fine detail" rather than missing content. F0 path remains perfect (`hgt_f0pred` ≈ ceiling, same as Plan A).

This validates:

1. **The "HuBERT favors regression" hypothesis** is correct in practice, not just on the diagnostic. A 8M deterministic head outperforms a 45M flow decoder for this content-prediction task.
2. **Plan A's architectural shortcut is fully bypassed** by switching architectures rather than trying to fix the flow decoder via loss-level tweaks.
3. **The flow_decoder ↔ det_head decoupling design works**: training, eval, and inference all branch correctly on `model.mode`, and Plan A checkpoints can still load (with flow_decoder used for synthesis) if needed.
4. The ~26 MB det_head adds negligible deployment cost while providing the entire content learning capacity.

### Outcome summary

| Metric | Plan A (best of 75k flow training) | Plan C (30k det training) | Δ |
|---|---|---|---|
| val_loss equivalent | 0.2225 (flow MSE based) | 0.1183 (L1 + 0.5*(1-cos)) | (different formulation) |
| val recon_l1 | 0.118 (train, marginal) | **0.1183** (val, real) | aligned |
| val recon_cos_sim | 0.76 (train, marginal-shortcut) | **0.854** (real content) | +0.094 |
| val f0_mae | 47.3 cents | **44.6 cents** | −2.7 cents |
| Audio: pred_full | "啊啊啊" no lyrics | **clear lyrics** | qualitative win |
| Audio: h_pred_f0gt | "啊啊啊" no lyrics | **clear lyrics** | qualitative win |

**Verdict**: Plan C is the production model. flow_decoder is preserved in state_dict for the future hybrid experiment but is no longer the primary path. Phase 2 (Japanese fine-tuning) will warm-start from Plan C's `best.pt`.

### Lessons / things to remember

1. **EMA is not unconditionally safe.** When warm-starting with a new fresh-init module, the EMA blend can produce a dysfunctional model that's worse than either the warm shadow or the live training state. Detection: train healthy + val pathological + gap doesn't close.

2. **Always have a `live vs ema` diagnostic** ready. `scripts/diag_live_vs_ema.py` is 30 lines and definitively answers the question.

3. **Best-checkpoint criterion matters more than people think.** The original `val_total_loss` criterion would have been polluted by the EMA pathology. Switching to `val_recon_l1` (the deterministic content metric) at least made the symptom legible.

4. **Mode-flag architecture branching pays off.** Building both `flow_decoder` and `det_head` into the same model class with a `mode` selector cost ~30 lines and means we can ship both modes from one checkpoint format. The decoupling also makes the future hybrid (det warm-start + flow refiner) a small additional change rather than a rewrite.

5. **Don't trust cos sim alone as a content metric.** The Plan A path had higher cos (0.76) than the diagnostic det head (0.62) but produced WORSE audio. cos sim weights all dimensions equally; phoneme content lives in a small subspace that L1 supervises better.

---

## Issues Encountered & Fixes

### Pre-training pipeline issues found and fixed

1. **HuBERT frame rate mismatch** (critical)
   - **Problem**: Documented as 100fps, but HuBERT-Base actually outputs 50fps (16kHz / 320 internal hop). F0 was extracted at 100fps, causing 2x frame count mismatch with HuBERT.
   - **Fix**: Changed `FRAME_RATE = 50` in preprocessing. F0 is now resampled to match HuBERT's actual frame count using `torch.nn.functional.interpolate`. Durations are recomputed proportionally.

2. **Model parameter count vs documentation**
   - **Problem**: Architecture doc claimed 62M, actual config built 128M. Encoder was documented 15.7M but actually 54.8M; flow decoder documented 38.5M but actually 65.9M.
   - **Fix**: Reduced encoder/flow to 4 layers each, total ~90M. Documentation updated.

3. **`torchcodec` audio decoder dependency on Windows**
   - **Problem**: `datasets` 4.x hardcodes torchcodec for audio decoding, which requires FFmpeg shared libraries unavailable on standard Windows installations.
   - **Fix**: Bypass via Arrow format (`with_format("arrow")`) and decode audio bytes directly with soundfile. Updated `prepare_ace_opencpop` to use this approach.

4. **`binary_cross_entropy` autocast incompatibility**
   - **Problem**: `F.binary_cross_entropy` is unsafe under torch AMP autocast.
   - **Fix**: Switched to `F.binary_cross_entropy_with_logits` using cached voicing logits.

5. **F0 predictor non-differentiable hard threshold**
   - **Problem**: `f0_hz * (voicing > 0.5).float()` was applied during training, breaking gradient flow.
   - **Fix**: Use ground-truth voicing mask during training (teacher forcing), only use predicted threshold during inference.

6. **`num_workers=4` hang on Windows**
   - **Problem**: Windows multiprocessing with PyTorch DataLoader workers caused training to hang at startup.
   - **Fix**: Use `num_workers=0` on Windows. Single-threaded data loading works fine.

7. **RMVPE F0 extractor import path**
   - **Problem**: RMVPE uses `from infer.lib import jit`, requires RVC root on sys.path (not just `infer/`).
   - **Fix**: Add both RVC root and `infer/` to sys.path in `_load_rmvpe`.

8. **RVC inference shape mismatch** (during eval)
   - **Problem**: Original `synthesizer.py` upsamples HuBERT 2x before passing to RVC and uses `feat.permute(0,2,1)` to make `[1, 768, T]`. Both are wrong — RVC's `enc_p` expects `[B, T, 768]` (NOT permuted) and length-matched feat/F0.
   - **Fix**: Pass feat as `[B, T, 768]` directly without permutation. Don't upsample 2x. Match feat and F0 to common length T. `synthesizer.py` should be patched accordingly.

---

## Methodology Notes

### Why IPA over language-specific phonemes?

For SVS specifically, the cross-language information loss in IPA mapping is minimal:
- Lexical tones (Chinese) become irrelevant in singing — F0 is determined by melody
- Pitch accent (Japanese) — same reason
- Aspiration (送气) — preserved in IPA via `pʰ` vs `p`
- Vowel allophones — minor differences imperceptible after RVC processing

The benefits are substantial:
- Universal vocabulary: every language's /a/ is the same token
- No cross-lingual fine-tuning needed (model is multi-lingual from day one)
- Smaller encoder needed (no cross-lingual learning capacity required)
- Better data efficiency (all language data trains the same embeddings)

### RVC as the quality bottleneck

The RVC decoder (NSF-HiFiGAN variant) is designed for speed/simplicity, not high fidelity:
- Smooths fine HuBERT detail by ~10-20%
- Limited to 24-40kHz output
- Adds characteristic "RVC artifacts"

This means the practical quality difference between a 60M and 128M Score2HuBERT model is largely lost after passing through RVC. We optimize for "good enough HuBERT features" rather than "perfect HuBERT features".

### Why disable multiscale loss?

The multiscale_hubert loss does flow matching loss at 3 temporal scales (1, 5, 10 frames). The intent is to encourage both local and global consistency. In practice:
- It does ~3x extra forward passes through the flow decoder per training step
- The auxiliary signal it provides is limited (the main flow loss already encodes everything)
- Weight is only 0.3, so its actual contribution to gradient is small
- For our use case (single-pass training with limited compute), the speed gain is far more valuable than the marginal quality benefit

### Why 50k steps instead of 100k?

Looking at the convergence curves:
- F0 loss converged around step 25-30k
- Flow loss was still slowly decreasing at step 50k
- Val loss reached 0.232 at step 50k, was 0.241 at step 44k → only 4% improvement in last 6k steps

A second 50k phase (resuming from checkpoint) might yield 5-10% additional improvement on flow loss but likely diminishing returns. The decision to extend depends on audio quality assessment.

---

## Future Work

After Phase 1 pre-training completes, planned next steps:

1. **Inference quality evaluation** ← **next**
   - Listen to the 12 generated test audio files
   - Assess intelligibility, naturalness, artifacts
   - Compare ablations to identify HuBERT vs F0 bottlenecks

2. **Phase 2: Japanese fine-tuning**
   - Add Kiritan + Ofuton-P + GTSinger-ja datasets
   - Lower LR (5e-5), partial encoder freeze
   - 10-20k additional steps
   - Should be efficient due to IPA shared vocabulary

3. **Rap mode training (Phase 3)**
   - Use GTSinger speech-singing paired data
   - `singing_ratio` continuous parameter
   - Test with rap-style inputs (no fixed pitch)

4. **RVC compensation analysis**
   - Run roundtrip analysis on 500-1000 samples
   - Compute α offset per dimension
   - Apply during fine-tuning

5. **Variance adaptor expansion**
   - Add user controls for breathiness, energy
   - Test interactive control during inference

6. **Inference speed optimization**
   - n_steps=4 seems best — verify perceptually
   - Combined with smaller model = potential <100ms latency for real-time DAW preview

---

## Run 4: Phase 2 — Multilingual fine-tune (2026-04-09 afternoon)

### Setup
- Init from Plan C: `--init-from checkpoints/pretrain_opencpop_det/best.pt`
- Dataset: `multi_lang_4` = ace_opencpop (100k zh singing) + gtsinger_zh (10188) + gtsinger_ja (2831) + gtsinger_en (6891) = **119,910 entries**, 95% singing / 5% speech
- Config: `configs/finetune_multi_det.yaml` (det mode, EMA off, batch 32, 25k steps, lr 7e-5, warmup 1000)
- Wall time: ~4.25 hours (25k steps + 50 vals)

### Added vs Plan C
- **English support**: new `θ`/`ð` IPA tokens at IDs 73/74 (taking `<unused_73>`, `<unused_74>` slots — backwards compatible with Plan C checkpoint)
- **ARPABET → IPA G2P** in `src/data/phoneme.py`
- **MFA-Japanese → IPA** normalization (GTSinger Japanese uses MFA standard, NOT openjtalk — different symbol set)
- **GTSinger preprocessing** via `prepare_gtsinger`: reads per-language `processed/processed/<lang>/metadata.json`, handles cp437/utf-8 mojibake from Mac zip extraction, resolves wav paths via `_build_gtsinger_path_index`, extracts 6 technique flags (UNUSED in Phase 2 — see Phase 3), saves tensors
- **Speech support** via GTSinger's Paired_Speech_Group: parses per-utterance JSON with word-level phoneme timing, sets `singing_ratio=0.0`, `midi=0`, activates `rap_head` in F0 predictor for the first time (was dormant in Plan C)

### Results (Phase 2 ORIGINAL, step 25000)
- val_recon_l1: 0.1285
- val_recon_cos: 0.810
- val_f0_mae: 65.2 cents

### Audio listening verdict
- zh on gtsinger: **off-key** — user report: "pred_full 跟 ceiling 几乎一样跑调"
- ja/en on gtsinger: off-key
- speech: "古神低语" feeling (low pitched + weird, but words clear)

### Diagnosis: F0 truncation bug

Original `_process_gtsinger_{singing,speech}` used:
```python
f0 = f0_extractor.extract(wav_path)[:n_frames]  # WRONG
```

RMVPE outputs F0 at **100fps**. HuBERT is at **50fps**. Truncating to `n_frames` (a 50fps count)
keeps only the **first half** of the audio's F0 contour, stored as if at 50fps. Downstream
code interprets it as 50fps → pitch plays at 2x speed vs HuBERT → off-key.

Evidence on a real 12.394-sec sample:
- RMVPE output: 1240 samples @ 100fps ✓
- Stored f0: 622 samples (first 6.2 sec of audio stretched to appear as full 12.4 sec)
- Stored hubert: 619 frames @ 50fps (correct — covers full 12.4 sec)

This was a NEW code path; the existing `prepare_ace_opencpop` / `prepare_opencpop` had always
interpolated F0 correctly. The rewrite missed it.

**Downstream training effect**: model trained on wrong F0 targets learned to reproduce the
wrong stretched pattern. pred_full was off-key in the SAME way as gt_ceiling (not just because
of stored data, but because the model had baked it in).

### Fix: re-extract F0 + resume training

1. Fixed code: `torch.nn.functional.interpolate(f0_raw.view(1,1,-1), size=n_frames, mode='linear')`
2. `_reextract_f0_gtsinger.py`: targeted re-extraction of only f0/*.pt files (~20 min, no HuBERT re-run)
3. Hardlink gotcha: `Path.replace()` atomic rename broke the multi_lang_4 hardlinks. Had to
   delete `multi_lang_4/f0/` and re-run merge to re-hardlink.
4. `scripts/train.py --init-from checkpoints/finetune_multi_det/best.pt --override training.max_steps=5000 training.optimizer.lr=3e-5 ... paths.checkpoint_dir=checkpoints/finetune_multi_det_fix`
5. Wall time: ~53 min for 5000 steps + 20 vals

### Results (Phase 2 FIX, step 5000)
- val_recon_l1: **0.1259** (-2% vs original, on the CORRECTED targets)
- val_recon_cos: **0.826** (+2%)
- val_f0_mae: **61.8 cents** (-5.2%)

### Audio listening verdict (fix)
- zh on gtsinger: **pitch correct**, articulation slightly fuzzy
- ja/en on gtsinger: pitch correct
- speech: still has the "speech F0 through singing-trained RVC" quirk (unrelated to our model)

### Key finding: Phase 2 fix on ACE-Opencpop

Diagnostic test `_diag_phase2fix_on_acopencpop.py`: generated 5 long ACE-Opencpop test samples
with the Phase 2 fix model. User listening verdict: **quality is equal to or slightly BETTER
than Plan C on ACE-Opencpop**. Theory: the corrected F0 gradients during resume training let
the encoder learn cleaner representations, which benefits det_head's HuBERT prediction as well.

This means Phase 2 fix is strictly > Plan C (better on zh AND supports ja/en/speech). **Plan C
is effectively superseded as the production model** — though it's preserved as historical baseline.

### Known limitation: GTSinger articulation fuzziness

On GTSinger singing samples, user reports audibly fuzzy articulation. This is NOT a regression —
it's det_head's averaging behavior exposed by data with high intra-phoneme variance (human
expressive singing has more variance than ACE-Studio synthesis). Same det_head, same averaging,
just visible on harder data.

**Phase 3 (tech embedding) addresses this** — see `docs/PHASE3_TECH_EMBEDDING.md`.

---

## Phase 3: Technique Embedding (2026-04-10)

### Setup
- **Date**: 2026-04-10 (trained 23:15 → 00:32, ~77 min)
- **Init from**: `checkpoints/finetune_multi_det_fix/best.pt` (Phase 2 fix, step 5000)
- **Config**: `configs/finetune_multi_det_tech.yaml`
- **Architecture change**: Added `encoder.tech_proj = nn.Linear(6, 512)` zero-initialized.
  Encodes 6 per-phoneme GTSinger technique flags (mix_voice, falsetto, breathy, pharyngeal,
  vibrato, glissando) into the encoder hidden state. Zero-init ensures backwards compatibility.
- **Training**: 8000 steps, batch 48, lr=5e-5, EMA disabled
- **Data**: same `multi_lang_4` (119,910 entries). Tech flags exist for gtsinger (19,910 entries);
  ace_opencpop falls back to zeros (dataset class handles it).

### Results (Phase 3, step 7750 best)
- val_recon_l1: **0.1251** (from 0.1259, Δ=-0.0008, -0.6%)
- val_recon_cos: **0.828** (from 0.826, Δ=+0.002)
- val_f0_mae: **60.2 cents** (from 61.8, slight improvement)
- New params: +3584 (tech_proj weight 512×6 + bias 512)

### tech_proj weight analysis (post-training)
```
tech_proj.weight.std = 0.0116 (from 0.0 at init)
Per-technique column norms:
  glissando:   0.384  (learned most)
  falsetto:    0.329
  breathy:     0.291
  vibrato:     0.192
  pharyngeal:  0.160
  mix:         0.101  (learned least)
Reference: input_proj.weight.std = 0.018 → tech_proj is 64% of input_proj scale
```

### Audio listening verdict
- **Breathy/pharyngeal groups**: Technique IS learned — user can feel difference from control.
  But articulation improvement is marginal. User: "不确定是心理作用还是真的有区别".
- **Control group**: Similar to Phase 2 fix (as expected — tech flags all zero).
- **ace_opencpop**: Unchanged (tech_proj(0)=0, backwards compat verified).
- **General zh/ja/en singing**: Articulation WORSE than technique-specific groups (higher
  intra-group variance in the mixed pool).
- **Speech**: High intra-group variance, flat/robotic prosody, occasional "啊啊啊".
- **rap_in_song**: **BUG — extreme high-pitched screaming** on rap segments (see fix below).

### Rap mode bug (discovered & fixed 2026-04-10)

**Symptom**: rap_in_song eval produced "极高音调尖叫" on the rap (singing_ratio=0) segments.

**Root cause**: `f0_predictor.rap_head` was trained only on speech samples where MIDI pitch=0.
When rap mode is applied to a singing sample with real MIDI pitch (40-65), `note_pitch_proj`
produces feature vectors the rap_head never saw → F0 prediction goes to 584-709 Hz (3x GT).

**Fix**: In `synthesize()`, zero out MIDI pitch for phonemes where `singing_ratio < 0.5` before
feeding to encoder. This matches the training data pattern (speech has pitch=0). Fix is
inference-only (no retraining needed). After fix: rap section F0 dropped from 584 Hz to 165 Hz
(GT = 205 Hz) — reasonable speech range.

**Verdict**: Phase 3 tech embedding is the right direction but insufficient. Det_head's L1
averaging is the fundamental bottleneck, not lack of conditioning. Hybrid architecture (Phase 4)
is the next step.

---

## Phase 4: Hybrid Architecture — det_head + flow refiner (2026-04-10)

### Motivation
Phase 3 proved det_head's L1 averaging can't be fixed by adding more conditioning. A flow-matching
refiner on the residual (GT - coarse) was proposed to model the multi-modal distribution that L1
averages away.

### Phase 4 v1: Wrong noise scale (2026-04-10 ~02:00-03:30)

**Setup**:
- **New module**: `FlowRefiner` — 5.56M params, 4 conv blocks (k=7, d_hidden=384), OT-CFM on residual
- **Conditioning**: concat(frame_hidden[512] + hubert_coarse[768] + f0_emb[512]) = 1792-dim
- **Freeze**: Everything frozen except flow_refiner. Det_head output detached (`.detach()`).
- **Training**: 15000 steps, batch 48, lr=2e-4, EMA disabled
- **Config**: `configs/finetune_hybrid.yaml`, augmentation disabled

**Results (v1, step 6000 best)**:
- Training loss: 1.49 → 0.59 (converged)
- val_recon_l1 (refined, temp=1.0): **0.565** — 4.5x WORSE than coarse (0.125)

**Diagnosis: OT-CFM noise/signal mismatch**

Critical diagnostic revealed the problem:
```
Residual (hubert_gt - hubert_coarse) std:  0.1759
OT-CFM initial noise (x_0 ~ N(0,1)) std:  1.0000
Noise/signal ratio:                        5.7x ← noise overwhelms residual!
```

Temperature sweep on v1:
| temp | l1 | vs coarse |
|------|--------|-----------|
| 0.00 | 0.1198 | **-4.2% (BETTER)** |
| 0.05 | 0.1234 | -1.4% |
| 0.10 | 0.1331 | +6.4% (worse) |
| 1.00 | 0.5647 | +352% (disaster) |

**Key discovery**: At temp=0, the refiner acts as a deterministic residual corrector and
IMPROVES upon coarse by 4.2%. The velocity field IS learned correctly — but ODE integration
from N(0,1) starting point is too far. The refiner works; the noise scale was wrong.

### Phase 4 v2: Corrected noise scale (2026-04-10 ~10:30-12:00)

**Fix**: Changed `x_0 = randn() * noise_scale` where `noise_scale=0.18` (matched to residual std).

**Results (v2, step 13500 best)**:
- Training loss: 0.40 → 0.042 (converges 4x faster than v1!)
- val_recon_l1 (refined, temp=1.0): **0.185** — still worse than coarse but MUCH better than v1 (0.565)

Temperature sweep on v2:
| temp | l1 | cos | vs coarse |
|------|--------|-------|-----------|
| 0.00 | 0.1203 | 0.842 | -3.8% |
| 0.10 | 0.1212 | 0.841 | -3.1% |
| 0.30 | 0.1274 | 0.830 | ≈ same |
| 0.50 | 0.1389 | 0.808 | +11% |
| 1.00 | 0.1823 | 0.728 | +46% |

### Audio listening verdict (v2)
- **temp=0 (deterministic)**: Audibly improved over Phase 3. Articulation better. But still
  "大舌头" — sounds slightly slurred.
- **temp=0.5 (stochastic)**: Similar to temp=0. **Stochastic sampling does NOT help articulation**.
- **breathy/pharyngeal groups**: More obviously improved than other groups (confirms tech embedding
  contribution is real).
- **Not production quality yet**: User verdict "勉强能接受 但是还不足以作为成品的 articulation".

### GAN-vs-MSE insight

The stochastic refiner produces samples from the DISTRIBUTION P(hubert|score), but the GT is
one SPECIFIC sample. L1/cos against GT is mathematically guaranteed to be worse for stochastic
vs deterministic (because the mean minimizes expected L1). But audio quality could theoretically
be BETTER (sharper, more expressive). In practice: no audible benefit over deterministic.

### The fundamental insight: L1 regression = averaging, regardless of architecture

The entire Phase 3→4 arc proved one thing: **as long as the target is continuous 768-dim features
and the loss is L1/L2, the output will converge to the conditional mean.** Adding conditioning
(Phase 3 tech embedding) or modeling the residual distribution (Phase 4 refiner) provides
marginal improvements but cannot overcome the fundamental averaging behavior.

**The root cause of "大舌头" is continuous regression, not model capacity or architecture.**

This motivates Phase 5: switch from continuous regression to **discrete token classification**.

---

## Phase 5: Discrete HuBERT Tokens (planned, 2026-04-10)

See `docs/PHASE5_DISCRETE_HUBERT.md` for the full design and implementation plan.

**Core idea**: Replace continuous 768-dim HuBERT regression with discrete codebook token prediction.
K-means quantize the existing HuBERT features into 1024 tokens, predict token indices via
cross-entropy loss. Cross-entropy has NO averaging — each frame is classified to a specific
sharp codebook entry. `codebook[argmax(logits)]` → continuous features for RVC.

**Why this should work**: The "slurred" articulation is an artifact of L1 loss averaging across
multi-modal continuous targets. Discrete classification eliminates this: you predict token #437
or token #892, not some blurry average of both.

---

## Phase 5b: Language-Balanced Discrete Training (2026-04-10)

Hard-oversampled ja (39x) and en (16x) in training data. After 10000 steps:
- ja cos: 0.543 → 0.545 (Δ=+0.003, negligible)
- zh cos: 0.743 → 0.735 (Δ=-0.008, regressed)
- Conclusion: **Frozen encoder is the ja/en bottleneck**, not head training data.

## Phase 6: F0 Fine-tune (2026-04-10)

Unfroze only f0_predictor, 8000 steps, lr=3e-4, f0 loss weight=3.0.
- f0_mae: 60.2 → 60.0 cents (no improvement)
- Conclusion: f0_predictor is at capacity on this data. Needs cleaner targets or post-processing.

## Pitch Correction Post-Processing (2026-04-10)

Added ±100 cents clamping to MIDI note in synthesize(). Inference-only, no training.
Effective at preventing large pitch drift while preserving vibrato range.

## Critical Diagnostic: Cross-Language Phoneme Analysis (2026-04-10)

Measured HuBERT feature similarity for the same IPA phoneme across languages (singing only):

| Comparison | Feature cos | Token overlap (4096) |
|---|---|---|
| zh-ja (same phoneme, both singing) | **0.876** | **6.5%** |
| zh-en (same phoneme, both singing) | **0.840** | **8.0%** |
| same-lang singing-speech | 0.794 | 13.4% |

**Key finding**: Cross-language same-phoneme features are VERY similar (cos 0.88) but map
to completely different tokens (7% overlap) because 4096 tokens over-resolve nuisance factors
(singer timbre, recording environment, coarticulation). This means:

1. **Continuous regression was correct** — the feature space IS shared across languages
2. **Discretization was a detour** — over-resolution breaks cross-language phoneme sharing
3. **Plan C (det_head on ACE-Opencpop) was optimal** — clear articulation on low-variance data,
   and cross-language IPA transfer works naturally in continuous space

## Architecture Reset Decision (2026-04-10)

Full rationale in `docs/ARCHITECTURE_RESET.md`.

**Decision**: Roll back to Plan C det_head, strip unnecessary complexity, add minimal ja/en
phoneme extension + user-drawn pitch/loudness curves. The simplest architecture was the
best all along — the articulation problem was caused by GTSinger's data variance, not by
architectural limitations.

