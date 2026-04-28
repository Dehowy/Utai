# Phase 4: Hybrid Architecture (Det + Flow Refiner)

Written 2026-04-10 ~02:00. This is the **live plan** for Phase 4 — update this file as design evolves.

## Why

Phase 3 proved that **det_head's L1 averaging is the fundamental articulation bottleneck**, not lack of conditioning. Adding 6 technique flags (Phase 3 tech embedding) produced measurable but perceptually insufficient improvement (tech_proj learned from zero but its std=0.0116 was too weak vs input_proj std=0.0182). The deterministic regression ceiling is real.

**The core problem**: L1/L2 regression converges to `E[HuBERT | score]` — the conditional mean. When intra-phoneme variance is high (human expressive singing), the mean is "fuzzy" because it averages across all valid realizations. No amount of conditioning can fix this if the loss function is L1/L2 and the target is multi-modal.

**The solution**: Add a small flow-matching refiner that models the **residual** between GT HuBERT and det_head's mean prediction. Det_head produces the coarse mean; refiner adds back the 20% multi-modal variation. This is the standard two-stage approach (cf. DiffSinger, NaturalSpeech 2, ProDiff).

## Architecture

```
Score → Encoder → frame_hidden
                    ↓
              ┌─ det_head → hubert_coarse (L1 mean)
              │
              └─ flow_refiner → hubert_refined
                    ↑
                 residual = GT - hubert_coarse
                 (refiner learns to predict this residual)
```

### Flow Refiner (NEW module, ~8-15M params)

**NOT the existing 45M flow_decoder.** Build a purpose-fit small module:

```python
class FlowRefiner(nn.Module):
    """Small OT-CFM module that refines det_head output.
    
    Input:  x_0 = noise, conditioned on (frame_hidden, hubert_coarse, f0_emb)
    Output: residual prediction (added to hubert_coarse to get hubert_refined)
    
    Architecture options (pick one):
    A) 4x DiT blocks with AdaLN on frame_hidden (like existing flow_decoder but smaller)
    B) 4x Conv1d blocks (local refiner, RF ~28 frames) + cross-attn to frame_hidden
    C) 2x Conformer blocks (reuse encoder's block design)
    """
```

**Key design choices:**

| Choice | Option A (DiT-based) | Option B (Conv-based) | Option C (Conformer) |
|--------|---------------------|----------------------|---------------------|
| Param count | ~12M | ~8M | ~10M |
| Context | Global (self-attn) | Local (~28 frames) + cross-attn | Global (self-attn) |
| Matches existing code | Yes (reuse DiTBlock) | Partially | Yes (reuse ConformerBlock) |
| Risk | DiTBlock AdaLN problem (same root cause as Plan A) | Safe — no AdaLN shortcut | Safe |

**Recommendation: Option B (Conv-based)** — same architecture as det_head but takes residual as target instead of raw HuBERT. Simple, proven, no AdaLN shortcut risk. The refiner doesn't need global context because det_head already provides a good coarse prediction — refiner just needs to sharpen local details.

Alternative: **Option C (Conformer)** if we need global context for long-range expression consistency.

### Conditioning

The refiner is conditioned on:
1. `frame_hidden` (encoder output, phone-level → expanded to frames) — carries all score information
2. `hubert_coarse` (det_head output) — the mean to refine away from
3. `f0_emb` (projected F0, same as what det_head sees)

Conditioning enters via **concatenation** (not AdaLN — learned from Plan A failure):
```python
cond = torch.cat([frame_hidden, hubert_coarse, f0_emb], dim=-1)  # [B, T, 512+768+256]
cond_proj = nn.Linear(512+768+256, d_refiner)  # project to refiner hidden dim
```

### Training Flow (OT-CFM on residual)

```python
# During training:
hubert_coarse = det_head(frame_hidden, f0_emb).detach()  # STOP gradient to det_head
residual_gt = hubert_gt - hubert_coarse                   # target residual
# OT-CFM: x_0 ~ N(0, 1), x_1 = residual_gt
# loss = MSE(predicted_velocity, target_velocity) on the residual

# During inference (T steps):
hubert_coarse = det_head(frame_hidden, f0_emb)
residual_pred = flow_refiner.sample(noise, cond, n_steps=T)  # ODE solve
hubert_refined = hubert_coarse + residual_pred
```

**Critical: `.detach()` on hubert_coarse.** The refiner should NOT backprop through det_head. Det_head is already converged; refiner learns to model the residual distribution.

### Inference Steps

OT-CFM on a residual distribution should converge much faster than on full HuBERT (because the residual has lower entropy). Expected: **1-4 Euler steps** sufficient (vs 10-50 for full-distribution flow matching). Start with 4, ablate down.

## Training Strategy

### What to freeze

| Module | Freeze? | Why |
|--------|---------|-----|
| Encoder | **Yes** | Already converged. Phase 3 tech_proj is in there too. |
| det_head | **Yes** | Its output is the conditioning for the refiner. Changing it during training would be a moving target. |
| f0_predictor | **Yes** | Not related to HuBERT quality. |
| variance_adaptor | **Yes** | Same. |
| flow_refiner | **No** | The only trainable module. |
| flow_decoder (legacy 45M) | **Ignore** | Not used. Still in state_dict, still frozen. |

Trainable params: **refiner only (~8-15M)**.

### Init

- Refiner is NEW (random init)
- `--init-from checkpoints/finetune_multi_det_tech/best.pt` loads everything except refiner
- EMA: **disabled** (same pitfall as Plan C warm-start — fresh module + warm components)

### Loss

```python
loss = flow_matching_loss(refiner_output, residual_gt)  # MSE on velocity field
# Optionally add a small L1 reconstruction loss on hubert_refined vs hubert_gt:
# loss += 0.1 * L1(hubert_coarse + refiner_output, hubert_gt)
```

### Config sketch

```yaml
model:
  mode: hybrid  # NEW mode: det_head + flow_refiner
  flow_refiner:
    d_hidden: 384
    n_blocks: 4
    kernel_size: 7
    d_cond: 1536  # 512 + 768 + 256
    dropout: 0.1
    sigma_min: 1.0e-4
    ot_ode: true

training:
  batch_size: 32  # conservative until profiled
  lr: 2.0e-4      # higher lr for fresh module
  max_steps: 15000
  # Only refiner params in optimizer (everything else frozen)
```

### Data

Same `data/processed/multi_lang_4` as Phase 3. No new preprocessing needed — det_head output is computed on-the-fly during training via frozen forward pass.

### Expected wall time

- 15000 steps * ~500ms/step (fwd through frozen encoder+det → refiner fwd+bwd) ≈ 125 min
- Plus validation ≈ 30 min
- **Total: ~2.5 hours**

## Implementation Steps

### Step 1: Build FlowRefiner module (~30 min)
- New file `src/model/flow_refiner.py`
- Conv1d-based (like det_head but with OT-CFM sampling)
- Forward: `(noisy_residual, time_emb, cond) → velocity prediction`
- Sample: Euler ODE solve over T steps

### Step 2: Add `mode: hybrid` to Score2HuBERT (~20 min)
- `score2hubert.py`: new branch in forward() and synthesize()
- forward(): `hubert_coarse = det_head(...).detach()` → `residual_gt = hubert_gt - hubert_coarse` → `flow_loss = refiner(residual_gt, cond)`
- synthesize(): `hubert_coarse = det_head(...)` → `residual = refiner.sample(noise, cond, n_steps)` → `return hubert_coarse + residual`

### Step 3: Trainer updates (~15 min)
- Freeze all modules except flow_refiner
- New loss term for flow_refiner
- Hybrid mode validation: compute hubert_refined, measure L1/cos against GT

### Step 4: Config + smoke test (~15 min)
- `configs/finetune_hybrid.yaml`
- 100-step smoke test

### Step 5: Full training (~2.5 hours)
- Profile batch size first
- Launch full run

### Step 6: Eval + listen (~30 min)
- Same _phase2_final_eval.py, point at hybrid checkpoint
- Focus on breathy/pharyngeal articulation — the primary metric
- Also check: does the stochastic sampling introduce variance? (run same sample twice, compare)

## Open Questions (decide before implementing)

### Q1: Refiner architecture — Conv or Conformer?
Conv (Option B) is simpler and avoids AdaLN pitfall. But if articulation needs long-range consistency (e.g., consistent vowel quality across a sustained note), Conformer might be needed. **Default: start with Conv, escalate if needed.**

### Q2: Refiner size?
8M (4 conv blocks k=7 d=384) mirrors det_head. Could go larger (12-15M) if 8M underfits. **Default: 8M, match det_head.**

### Q3: Number of inference steps?
OT-CFM on residual should be fast. **Default: 4 Euler steps.** Ablate: try 1, 2, 4, 8 and listen.

### Q4: Should we keep tech_proj?
Phase 3 tech embedding provides *some* signal (tech_proj learned from zero). Keeping it costs nothing (it's already in the encoder). The refiner will also benefit from it because frame_hidden carries the tech signal. **Default: keep.**

### Q5: Existing flow_decoder (45M) — reuse or ignore?
The frozen 45M flow_decoder in state_dict has DiTBlock with the AdaLN shortcut problem. Reusing it as-is would repeat Plan A's failure. **Default: ignore it, build fresh FlowRefiner.** Could strip it from state_dict to save file size, but not urgent.

### Q6: Temperature / cfg_scale at inference? — ANSWERED 2026-04-10

**Temperature must be near 0, NOT 1.0.** Diagnostic proved:

Residual std = 0.18, but OT-CFM noise std = 1.0 → **5.7x noise/signal mismatch**.

| temp | l1 | vs coarse |
|------|-------|-----------|
| 0.00 | 0.1198 | -4.2% (BETTER!) |
| 0.05 | 0.1234 | -1.4% |
| 0.10 | 0.1331 | +6.4% (worse) |
| 1.00 | 0.5647 | +352% (disaster) |

**Conclusion**: Use temperature=0.0 (deterministic residual corrector). The refiner learns E[residual|cond] which improves upon det_head's coarse, giving a cascaded deterministic model: det_head → refiner correction → refined output.

For true stochastic sampling (future): retrain with x_0~N(0, 0.18) instead of N(0,1) to match the residual scale. Or use DDPM with appropriate noise schedule.

## Fallback plan

If hybrid doesn't improve articulation either (unlikely but possible):
1. The residual distribution might be too complex for a small refiner → try larger refiner (15-20M)
2. Or: the residual is not well-modeled by OT-CFM → try diffusion (DDPM-style) on residual
3. Or: fundamentally, HuBERT is too averaged a representation → need a finer target (e.g., discrete HuBERT codes instead of continuous features)

## Side quests (defer until after Phase 4)

- **Speech data augmentation**: LibriSpeech (en), AISHELL (zh), CommonVoice/JVS (ja) for speech prosody
- **Targeted f0 fine-tune**: separate from hybrid, addresses pitch drift
- **Rap mode**: pitch-zeroing fix is inference-only; might benefit from training with mixed singing_ratio data
