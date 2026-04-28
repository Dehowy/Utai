# Phase 3: Technique Embedding — Design Doc + Audit Checklist

Engineered 2026-04-09 evening. Awaiting audit, then training launch.

## Motivation

Phase 2 fix is production-quality on ACE-Opencpop-style Chinese singing — listening comparison
confirms it matches or slightly exceeds Plan C. BUT on GTSinger singing samples (human performers
with natural expression), the user reports **fuzzy articulation** when listening carefully.

**Root cause**: det_head does deterministic L1 regression — it converges to
`E[HuBERT | score]` = the conditional mean. When the target HuBERT has high intra-phoneme
variance (same phoneme sung with different techniques: breathy /a/ vs pharyngeal /a/),
averaging across all variants produces a "fuzzy" prediction. ACE-Opencpop was machine-synthesized
(low intra-phoneme variance) so this averaging was invisible; GTSinger's human performers
expose it.

**Key insight**: GTSinger explicitly provides **per-phoneme technique flags** in its metadata
(6 binary arrays: mix_tech, falsetto_tech, breathy_tech, pharyngeal_tech, vibrato_tech,
glissando_tech). The dataset was designed for "technique-controllable SVS" (GTSinger paper
section 4.1). We had been **ignoring these flags entirely** — the model saw only phonemes + MIDI
and had to average across every technique variant.

**Fix**: Feed the 6 flags into the encoder. Condition the model on the observed variance source.
Deterministic regression on the conditioned distribution has **much lower intra-phoneme variance**
→ the averaging loses less detail → clearer articulation.

## Why not hybrid architecture?

The Plan C decisions doc originally listed "hybrid (det_head + flow refiner on residual)" as the
fallback if det_head proved insufficient. Phase 3 technique embedding is **cheaper and more
controllable** than hybrid:

| Aspect | Hybrid (det + flow refiner) | Technique embedding |
|---|---|---|
| Engineering effort | 1-2 day (new flow module, loss, training recipe) | ~0.5 day (one Linear layer, data pipeline fix) |
| Training cost | ~4h full retrain | ~1.5h resume from Phase 2 fix |
| Produces | Stochastic output (random sample of variance) | Deterministic output conditioned on tech flags |
| User control | None — model picks a "mode" randomly | Explicit per-phoneme technique selection |
| DAW integration | User writes score → random variant each render | User writes score + ticks technique boxes → deterministic render |
| Handles what variance | Any residual (even unmeasured, like singer idiosyncrasy) | Only the technique-caused variance (the main observable one) |

**Decision: technique embedding first.** It directly addresses the variance source we CAN
observe. Hybrid remains the fallback if tech flags aren't enough (e.g., singer-style variance
beyond techniques). For the product use case (composer writing a song in a DAW), deterministic
+ controllable is strictly better than stochastic.

## Architecture change

### Encoder (`src/model/encoder.py`)

Add a zero-initialized `tech_proj` linear layer:

```python
# In __init__:
self.tech_proj = nn.Linear(n_techniques, d_model)  # n_techniques = 6
nn.init.zeros_(self.tech_proj.weight)
nn.init.zeros_(self.tech_proj.bias)
```

In `forward`, add a new optional kwarg and use it after `input_proj` + `lang_embed`,
before positional encoding:

```python
def forward(self, phonemes, pitch, duration, singing_ratio,
            lang_id=None, phone_mask=None, tech=None):
    # ... existing phone_emb, pitch_emb, dur_emb, sr_emb, input_proj, lang_embed ...
    if tech is not None:
        x = x + self.tech_proj(tech)  # [B, T_phone, d_model]
    # ... pos_enc, conformer blocks ...
```

**Why zero init?** Makes the new architecture **perfectly backwards compatible**. When loading
Phase 2 fix checkpoint via `--init-from`, the 2 new keys (`encoder.tech_proj.weight`,
`encoder.tech_proj.bias`) start at zero. Any `tech` input multiplied by zero weights produces
zero contribution → encoder output at step 0 is IDENTICAL to Phase 2 fix. As training proceeds,
gradients push tech_proj away from zero to exploit the newly-available signal.

Param count increase: `6 * 512 + 512 = 3584` (verified in smoke test).

### Score2HuBERT (`src/model/score2hubert.py`)

Both `forward` (training) and `synthesize` (inference) accept an optional `tech` kwarg and
pass it through to `self.encoder(...)`. No other changes needed.

### Trainer (`src/training/trainer.py`)

Training step and validation step pass `batch.get("tech")` to `self.model(...)`. Using `.get`
means older datasets without tech still work (returns None → encoder skips the add).

### Dataset (`src/data/dataset.py`)

Load `tech/{utt_id}.pt` if it exists, else return a zero tensor of shape `[T_phone, 6]`:

```python
tech_path = self.processed_dir / "tech" / f"{sid}.pt"
if tech_path.exists():
    tech = torch.load(tech_path, weights_only=True).float()
    # Defensive length alignment to phoneme count
    if tech.size(0) != phonemes.size(0):
        ...
else:
    tech = torch.zeros(phonemes.size(0), 6, dtype=torch.float32)
sample["tech"] = tech
```

This means:
- `gtsinger_*/tech/*.pt` exists → real flags get loaded
- `ace_opencpop/tech/` doesn't exist → dataset returns zeros → `tech_proj(zeros) = 0`
  → ace samples train exactly as before

### Collate (`src/data/collate.py`)

Pad tech to `[B, T_phone_max, 6]`, defaulting to 6 technique dims if not present in first
sample. Always includes tech in the output dict.

## Data pipeline change

### `prepare_dataset.py` additions

New constants + helper:

```python
GTSINGER_TECH_KEYS = [
    "mix_tech",        # index 0
    "falsetto_tech",   # index 1
    "breathy_tech",    # index 2
    "pharyngeal_tech", # index 3
    "vibrato_tech",    # index 4
    "glissando_tech",  # index 5
]
N_TECHNIQUES = 6

def _expand_tech_via_gtsinger_g2p(raw_phs, tech_matrix, language):
    """Expand per-phoneme tech flags parallel to IPA decomposition.
    Same pattern as _expand_pitches_via_gtsinger_g2p — each decomposed IPA segment
    inherits the parent raw phoneme's flags."""
    ...
```

`_process_gtsinger_singing` now:
1. Stacks 6 tech arrays from the entry into `[N_raw, 6]`
2. Expands parallel to IPA decomposition via `_expand_tech_via_gtsinger_g2p`
3. Aligns length via `min_len` (same as other per-phoneme tensors)
4. Saves `tech/{utt_id}.pt` as `[T_phone, 6]` float32

`_process_gtsinger_speech` saves all-zeros of shape `[T_phone, 6]` (speech has no technique
annotation in GTSinger).

### `_add_tech_flags_to_existing_gtsinger.py` (one-off targeted script)

Populated tech files for existing `data/processed/gtsinger_{zh,ja,en}/` WITHOUT re-running
HuBERT/F0 extraction. Walks our metadata, looks up original GTSinger source metadata by
`item_name`, builds tech matrix, expands, saves. Handles speech as all-zeros. Took ~30 sec total.

Run output verified: 10188 zh + 2831 ja + 6891 en, all 0 failures, 0 missing, 0
length-adjusted.

### Merge pipeline (`scripts/preprocess.py`)

`merge_datasets` subdir list extended with `"tech"`. After populating the source tech dirs,
we deleted `multi_lang_4/tech` and re-ran merge — took a few seconds via `os.link()` hardlinks.

`multi_lang_4/tech/` contains only the 19910 gtsinger files (ace_opencpop doesn't have
`tech/`). Dataset class falls back to zeros for ace samples.

## Training config

`configs/finetune_multi_det_tech.yaml`:

- `model.encoder.n_techniques: 6` (new field, matches encoder's default)
- `training.batch_size: 48` (bumped from Phase 2's 32 — profile showed safe on 12 GB RTX 3080 Ti)
- `training.num_workers: 8` (bumped from 4)
- `training.optimizer.lr: 5.0e-5` (lower than Phase 2 fix's 7e-5 because model is well-trained)
- `training.max_steps: 8000`
- `training.scheduler.warmup_steps: 300`, `total_steps: 8000`
- `training.val_every: 250`, `save_every: 1000`
- `paths.checkpoint_dir: checkpoints/finetune_multi_det_tech`
- `paths.log_dir: logs/finetune_multi_det_tech`

All other fields match Phase 2 config (det_head shape, EMA off, variance adaptor, etc.).

## VRAM profile (batch 48 at multi_lang_4)

Measured 2026-04-09 evening via `scripts/profile_step.py`:

| Batch | Peak VRAM | Reserved VRAM | Samples/sec | Step time (median) |
|---|---|---|---|---|
| 48 | 7.46 GB | 9.86 GB | 130 | 369 ms |
| 56 | 7.66 GB | 10.36 GB | 137 | 409 ms |
| 64 | **10.02 GB** | **12.40 GB** | 131 | 487 ms + outliers |

Batch 64 spills into Windows shared memory (reserved > 12 GB physical on RTX 3080 Ti).
Sweet spot = batch 48 (peak well under limit, stable step time, good throughput).

## Expected training trajectory

Starting point (Phase 2 fix):
- val_recon_l1 = 0.1259
- val_recon_cos = 0.826
- val_f0_mae = 61.8 cents

After ~8000 steps:
- val_l1: small improvement (maybe 0.123-0.125 range — det_head now has the extra signal)
- val_cos: small improvement (0.83-0.84)
- val_f0_mae: essentially unchanged (f0 doesn't depend on tech)

On gtsinger audio:
- **Breathy_Group / Vibrato_Group / Pharyngeal_Group samples** (where tech flags are densely
  1 for 50-90% of phonemes) → expect NOTICEABLY clearer articulation, because the model can
  now produce a "breathy /a/" instead of averaging across all /a/ variants
- **Control_Group samples** (tech flags mostly 0) → similar to Phase 2 fix (model falls back
  to the "natural" distribution, which is what it learned before)

On ace_opencpop audio:
- **Unchanged** (tech_proj(zeros) = 0 → encoder output identical to Phase 2 fix)

## AUDIT CHECKLIST

Before launching full training, verify:

### 1. Encoder tech_proj zero init
```
venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'D:/MyDev/MSSTRVC/score2hubert')
from src.model.encoder import ScoreEncoder
e = ScoreEncoder()
import torch
print('weight std:', e.tech_proj.weight.std().item(), '(should be 0.0)')
print('bias mean: ', e.tech_proj.bias.mean().item(), '(should be 0.0)')
"
```

### 2. Spot-check a tech file is semantically correct
A Breathy_Group sample should have `breathy_tech` (index 2) = 1 for most phonemes.
A Control_Group for the same song should have `breathy_tech` = 0.
A speech sample should have all zeros.
An ace_opencpop sample file shouldn't exist (fallback to zero in dataset class).

### 3. Encoder param count delta
Phase 2 fix encoder: 37,029,376 params.
Phase 3 encoder: 37,032,960 params.
Delta: +3584 = 6*512 (weight) + 512 (bias) = tech_proj exactly.

### 4. Init-from loads cleanly
Run 10-step smoke train with `--init-from checkpoints/finetune_multi_det_fix/best.pt`.
Verify log shows:
- `Init checkpoint missing 2 unexpected keys` with keys `encoder.tech_proj.{weight,bias}`
- `loaded 441 keys, 2 fresh`
- Step 0 val metrics ≈ Phase 2 fix baseline (l1 ~0.126, cos ~0.825)

### 5. Dataset returns tech for BOTH gtsinger and ace samples
```
from src.data.dataset import Score2HuBERTDataset
ds = Score2HuBERTDataset("data/processed/multi_lang_4", split="test")
for s in [ds[i] for i in range(5)]:
    print(s["id"], "tech shape:", s["tech"].shape, "sum:", s["tech"].sum().item())
```
All samples should have `tech.shape = [T_phone, 6]`. Gtsinger technique-group samples should
have non-zero sums; ace_opencpop_ and speech samples should have zero sums.

### 6. Collate batches tech correctly
Build a batch of mixed samples, verify `batch["tech"].shape = [B, max_phones, 6]` and
padding rows are zeros.

### 7. Forward pass doesn't error and gives similar loss to Phase 2 fix baseline
Smoke test with 100 steps should give val l1 ~0.126, not wildly off.

## LAUNCH COMMAND

Full Phase 3 training:

```bash
cd D:/MyDev/MSSTRVC/score2hubert
venv/Scripts/python.exe scripts/train.py \
  --config configs/finetune_multi_det_tech.yaml \
  --init-from checkpoints/finetune_multi_det_fix/best.pt
```

Expected wall time: ~90 min (8000 steps × ~400ms + 32 vals × ~70s each).

Output:
- `checkpoints/finetune_multi_det_tech/best.pt` — rolling best val_l1
- `checkpoints/finetune_multi_det_tech/step_{1000,2000,...}.pt` — periodic saves
- `logs/finetune_multi_det_tech/train.log` — full training log
- `logs/finetune_multi_det_tech/events.out.tfevents.*` — TensorBoard

## Post-training eval

Point `_phase2_final_eval.py` at the new checkpoint:

```python
# Edit:
CKPT_PATH = PROJECT / "checkpoints/finetune_multi_det_tech/best.pt"
OUT_ROOT = PROJECT / "test_outputs_phase3_final"
```

Run. Listen to `test_outputs_phase3_final/zh_singing/`, `ja_singing/`, `en_singing/`,
compare to `test_outputs_phase2_fix_final/` (same test sample IDs where possible because
random seed is fixed).

Focus on gtsinger test samples from Breathy_Group, Vibrato_Group, etc. where tech flags
are densely active — these should show the biggest articulation clarity improvement.

## Fallback plan if Phase 3 doesn't improve articulation enough

1. Verify tech flags are actually being used by checking if `encoder.tech_proj.weight` has
   non-zero std after training (it should — gradients will push it away from zero).
2. If tech_proj learned but articulation is still fuzzy, it means the variance is NOT fully
   captured by technique flags — there's additional singer-style variance we can't observe.
3. Next fallback = hybrid architecture (det + small flow refiner on residual). Re-read
   `project_score2hubert_decisions.md` section "[HISTORICAL] Plan C deterministic head"
   for the hybrid design sketch. flow_decoder weights are preserved in state_dict already.
