# Phase 5: Discrete HuBERT Token Prediction

Written 2026-04-10 ~12:30. **Live plan** — update as design evolves.

## Why

Phases 3-4 proved that **continuous regression (L1/L2 loss on 768-dim vectors) fundamentally
produces averaged, "slurred" output** regardless of architecture:
- Phase 3 (tech embedding): +0.6% L1, barely audible articulation change
- Phase 4 (flow refiner): +4% L1 at temp=0, still "大舌头"
- Phase 4 stochastic: no perceptual benefit over deterministic

The root cause is mathematical: L1 loss converges to `E[HuBERT | score]` — the conditional mean.
When the target distribution is multi-modal (same phoneme has many valid realizations), the mean
is "fuzzy" and sounds slurred.

**Discrete tokens eliminate averaging by construction.** Cross-entropy classification picks ONE
specific codebook entry per frame — there is no "average of token #437 and token #892".

## Architecture

### Current pipeline (continuous regression)
```
Score → Encoder (Conformer 4L) → frame_hidden → DetHubertHead → HuBERT [B,T,768] → RVC → audio
```

### New pipeline (discrete classification)
```
Score → Encoder (Conformer 4L) → frame_hidden → DiscreteHubertHead → logits [B,T,1024]
                                                                        ↓
                                                              argmax → token_ids [B,T]
                                                                        ↓
                                                           codebook lookup → HuBERT [B,T,768]
                                                                        ↓
                                                                   RVC → audio
```

### Codebook: K-means on existing Layer 12 features

**Why NOT HuBERT's built-in quantizer**: HuBERT-Base's `final_proj` (768→256) + `label_embs_concat`
(504 entries, 256-dim) operates in a projected space. RVC uses raw Layer 12 output (768-dim).
Training our own K-means on 768-dim features ensures the codebook precisely matches what RVC expects.

**Codebook size**: 1024 tokens (standard in HuBERT-based discrete speech work: VALL-E, SoundStorm).
- ~120K entries at 50fps ≈ 30M frames → ~30K frames/centroid (well sampled)
- 512 may lose fine detail; 2048 is richer but increases head size
- **Start with 1024, ablate 512/2048 if needed**

**Storage**: `data/processed/multi_lang_4/kmeans_1024.pt` (centroids [1024, 768]) +
`hubert_tokens/{id}.pt` (int16 [T], indices 0-1023). Token dir ≈ 60 MB vs 35 GB for continuous.

### DiscreteHubertHead

Same conv architecture as DetHubertHead (reuse weights for warm start), only the output layer changes:

```python
class DiscreteHubertHead(nn.Module):
    """Discrete token classification head: [frame_hidden; f0_emb] -> token logits."""
    # in_proj:   Linear(2*d_cond, d_hidden)       # SAME as DetHubertHead (reuse weights)
    # N x Conv1d blocks with residual              # SAME (reuse weights)
    # logit_proj: Linear(d_hidden, n_codes)        # NEW (random init)
    # codebook:  Embedding(n_codes, d_hubert)      # FROZEN (from K-means)
```

Param delta vs DetHubertHead:
- Removed: `out_proj Linear(512, 768)` = 393K
- Added: `logit_proj Linear(512, 1024)` = 525K + `codebook Embedding(1024, 768)` = 786K (frozen)
- Net: +918K params, ~8.9M trainable total

### RVC compatibility

RVC's `net_g.infer()` expects `[B, T, 768]` float features. The codebook lookup
`self.codebook(token_ids)` produces exactly this. Since K-means centroids ARE points from the
original HuBERT distribution, they are by construction in-distribution for RVC.

## Training

### Loss
```python
loss = F.cross_entropy(
    logits.reshape(-1, n_codes),       # [B*T, 1024]
    tokens.reshape(-1),                 # [B*T]
    label_smoothing=0.1,
    ignore_index=-100,                  # padding
)
```

- **Label smoothing 0.1**: prevents overconfident logits, improves generalization
- **ignore_index=-100**: for padded frames (collate pads tokens to max length)
- **No temperature during training** (standard CE)

### Metrics
- `token_accuracy`: % frames with argmax == gt_token (expected 20-40% converged — codebook
  has many semantically similar entries, so exact match won't be 100%)
- `top5_accuracy`: % frames with gt_token in top-5 logits
- `recon_l1`, `recon_cos_sim`: computed on `codebook[argmax(logits)]` vs `codebook[gt_token]`
  for comparability with Phase 3/4 metrics

### Freeze strategy
| Module | Freeze? |
|--------|---------|
| Encoder | **Yes** (well trained through Phase 3) |
| DiscreteHubertHead conv blocks | **No** (warm-init from det_head, fine-tune) |
| DiscreteHubertHead logit_proj | **No** (random init, primary training target) |
| Codebook | **Yes** (frozen K-means centroids) |
| f0_predictor | **Yes** (separate issue, Phase 6) |
| variance_adaptor | **Yes** |
| flow_decoder | Ignore (legacy) |
| flow_refiner | Ignore (Phase 4, superseded) |

### Init strategy
- `--init-from checkpoints/finetune_multi_det_tech/best.pt` (Phase 3 best)
- Encoder, variance, f0, f0_cond_proj: loaded from checkpoint
- DiscreteHubertHead conv blocks: **copied from det_head weights** (same architecture, strong warm start)
- DiscreteHubertHead logit_proj: random init
- EMA: **disabled** (same fresh-module pitfall)

### Inference
- **Default**: argmax(logits) → deterministic, reproducible
- **Optional**: top-k sampling with temperature for diversity (future, not V1)
- **Three-tier DAW inference** (from Phase 4 plan, still applies):
  - Fast (preview): old det_head → continuous (skip discrete entirely)
  - Standard: discrete head → argmax → codebook → RVC
  - High quality: discrete head → top-k sampling → codebook → RVC

### Config sketch
```yaml
model:
  mode: discrete
  discrete_head:
    d_hidden: 512           # Match det_head for weight reuse
    n_blocks: 4
    kernel_size: 7
    dropout: 0.1
    n_codes: 1024
    codebook_path: "data/processed/multi_lang_4/kmeans_1024.pt"
    init_from_det_head: true  # Copy conv block weights from det_head
training:
  batch_size: 48
  lr: 1.0e-4               # Moderate — conv blocks warm, logit_proj cold
  max_steps: 15000
  loss:
    discrete: 1.0
    f0: 0.0                 # frozen
```

## Implementation Steps

| Step | Work | Est. time | Files touched |
|------|------|-----------|---------------|
| 0 | **K-means training + quantization** | 30 min | 1 new script |
| 1 | **DiscreteHubertHead module** | 30 min | 1 new file |
| 2 | **Dataset + collate** (load hubert_tokens) | 20 min | 2 existing |
| 3 | **Score2HuBERT mode='discrete'** (forward + synthesize) | 40 min | 1 existing |
| 4 | **Trainer freeze + loss** | 15 min | 1 existing |
| 5 | **Config** | 10 min | 1 new file |
| 6 | **Smoke test** | 15 min | — |
| 7 | **Full training** | 2-3 hr | — |
| 8 | **Eval + listen** | 30 min | — |
| **Total** | | **~5-6 hr** (incl. training) | |

## Quantization Quality Pre-check

Before training, measure the K-means reconstruction quality:
```python
# For each frame: error = hubert_feature - codebook[nearest_token]
# Report: mean L1, mean cosine sim, worst-case L1
```

If L1 > 0.15 or cos < 0.9, the codebook is too small → increase to 2048 or 4096.
This is a cheap check (minutes) that gates the entire approach.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Quantization too lossy (1024 insufficient) | Low | High | Pre-check, increase to 2048 |
| Classification harder to learn than regression | Medium | Medium | Warm-start conv blocks from det_head |
| Dead codebook entries (unused tokens) | Low | Low | Monitor histogram, use K-means++ init |
| Frame-level CE ignores temporal structure | Low | Medium | Encoder already provides temporal context |
| RVC sensitive to discrete jumps | Low | Medium | K-means centroids are in-distribution by construction |
| Token accuracy plateaus low | Medium | Medium | Increase head capacity (more blocks/wider) |

## Fallback

If discrete 1024 doesn't improve articulation:
1. Try 2048/4096 tokens (finer codebook)
2. Try Residual VQ (multi-codebook, captures more detail per frame)
3. Try autoregressive token prediction (model temporal dependencies explicitly)
4. Accept that HuBERT-Base representation is fundamentally limited → switch to a finer
   representation (e.g., EnCodec tokens, Wav2Vec2 features)

## F0 pitch drift issue (deferred to Phase 6)

Phase 3/4 listening revealed per-language pitch quality correlation:
- en singing ceiling (GTSinger): good pitch → model output: least pitch drift
- zh singing ceiling (GTSinger): slightly off → model output: most pitch drift
- ACE-Opencpop (machine-synthesized): clean pitch → model output: ~45c MAE (Plan C level)

Root cause: f0_predictor faithfully learns GT F0 patterns, including GT's pitch imprecision.
Fix (Phase 6): fine-tune f0_predictor on cleaner data:
- zh: ACE-Opencpop only (machine-synthesized, clean) or pitch-corrected GTSinger_zh
- en: GTSinger_en as-is (already good)
- ja: evaluate, possibly pitch-correct
- speech: add LibriSpeech (en), AISHELL (zh), CommonVoice/JVS (ja)
