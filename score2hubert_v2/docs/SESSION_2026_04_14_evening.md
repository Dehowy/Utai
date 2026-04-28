# Session 2026-04-14 evening: Run 5 audio A/B → Contamination audit → Kiritan-only tests → HuBERT cluster insight → SOFA+K-means plan

## TL;DR

1. **Run 5 audio A/B** vs Run 4: ZH flat, JA slight improvement, JVS went from "啊啊啊" to "articulation attempts" (encoder health validated).
2. **Phoneme contamination audit** (scripts/diag_phoneme_contamination.py): discovered that most consonants have cross-dataset centroid cos 0.2–0.5 (severe disagreement), SP silence token 175K frames with cos 0.23 between ACE and JA datasets (silence is the single worst contamination source).
3. **Kiritan-only hypothesis test**: 
   - Run 6 (54M Kiritan-only, 20K steps budget, stopped at 7.5K): val cos 0.8043 vs Run 5's 0.8233 — slightly worse numerically, audio has similar messiness.
   - Run 7 (7.7M tiny Kiritan-only, 5K steps): val cos 0.7959 — comparable to Run 6 on numbers but audio more garbled.
4. **HuBERT is cluster space insight**: Mel is continuous (interpolatable); HuBERT is high-dim clustered. Small models L1-average between clusters → RVC decodes garbage. Explains why DiffSinger-scale shrinkage doesn't work.
5. **Ofuton 0-shot > Kiritan trained** on tiny model (surprising — suggests Ofuton is more "canonical" HuBERT).
6. **Next plan**: big model (~54M) + SOFA force-alignment + K-means discrete tokens + SP=0.

---

## Run 5 audio listening (user) — Run 4 → Run 5 deltas

Listened to samples in `test_outputs/run5_best/`:

| Dataset | Result | vs Run 4 |
|---|---|---|
| ZH ACE | Long notes still occasionally split into 2 phonemes | Unchanged (Kaiming LR init didn't fix) |
| JA Kiritan | Noise + occasional garbled chars | Similar to Run 4 |
| JA Oniku | ACE-ZH quality | Maintained (was already good) |
| **JVS (untrained)** | **"Articulation attempts" instead of "啊啊啊"** | **Major improvement** |

Also noted: **空拍艺品** (rest frames have residue, not clean silence) — similar to long-note splits. Root cause: model has no concept of temporal position within phoneme (LR's `pos_in_phone` features didn't load-bear even with Kaiming init).

User hypothesis: "only a few phonemes are severely contaminated" — led to phoneme-level audit.

---

## Phoneme contamination audit

**Script**: `scripts/diag_phoneme_contamination.py` (new, 2026-04-14 evening)
**Output**: `docs/phoneme_contamination_audit.json`

For each phoneme id, computes per-dataset HuBERT centroid, then all-pairs cross-dataset cosine similarity. Low min-pair cos = contamination candidate.

### Worst-10 phonemes by min cross-dataset cos (≥4 datasets, ≥1000 frames)

| Phoneme | min_cos | Frames | Worst pair |
|---|:-:|:-:|---|
| /z/ | 0.20 | 7K | oniku vs jvs_music |
| /ʔ/ | 0.21 | 7K | oniku vs jvs_music |
| **/SP/** | **0.23** | **175K** | **ACE vs oniku** |
| /h/ | 0.25 | 7K | oniku vs jvs_music |
| /g/ | 0.26 | 12K | oniku vs jvs_music |
| /t/ | 0.27 | 37K | itako vs jvs_music |
| /b/ | 0.30 | 9K | oniku vs jvs_music |
| /s/ | 0.30 | 31K | oniku vs jvs_music |
| /ɸ/ | 0.30 | 3K | oniku vs jvs_music |
| /dʑ/ | 0.31 | 4K | kiritan vs jvs_music |

**Vowels** significantly better: /a/ 0.65, /i/ 0.70, /o/ 0.69, /e/ 0.68.

### SP silence token — single biggest mess

| Dataset | n_frames | mean ‖HuBERT‖ |
|---|:-:|:-:|
| ace_opencpop | 167,721 | 9.043 |
| kiritan | 158 | 9.272 |
| itako | 732 | 9.426 |
| oniku | 1,619 | 9.393 |
| ofuton | 1,131 | 9.400 |
| natsume | 3,652 | 9.398 |

**Clean silence should be near zero; ours is 9+** across all datasets. Different datasets label different residue ("breath carryover", "pitch-corrected artifacts", etc) as SP. Cross-dataset cos between ACE-SP and any JA-SP is 0.23–0.34 — essentially disagreement.

**Implication**: model is forced to learn the "average SP" of 6+ datasets, which doesn't match any of them. Explains "rest frame artifacts" user heard.

### Who's the biggest outlier
- **JVS-MuSiC** appears in 16/22 worst-pair entries. Confirmed main contaminator from Run 5 analysis.
- **oniku** appears in many worst pairs too (was pre-denoised + pitch-corrected by author → systematically different HuBERT distribution).

---

## Kiritan-only hypothesis tests

Hypothesis: single-clean-dataset sidesteps all contamination findings at once (no cross-dataset disagreement if there's only one dataset).

### Run 6: 54M full arch, Kiritan-only

Config: `configs/train_ja_kiritan_only.yaml`
- Identical to Run 5 minus all datasets except Kiritan
- 20K steps target, stopped at 7.5K after plateau confirmed

Per-dataset eval trajectory (Kiritan val):
| Step | Kiritan cos |
|:-:|:-:|
| 1000 | 0.7877 |
| 2500 | 0.8017 |
| 5000 | 0.8041 |
| **7500** | **0.8043** |

**Plateaued at 0.80**, below Run 5's multi-dataset Kiritan 0.8233 (-0.02).

Audio A/B (user listen, sample 1/2 of kiritan):
- Run 6 ≈ Run 5 — similar messiness, some noise, occasional garbled phonemes
- Not clearly better despite contamination-theory prediction
- Generalization to untrained Itako/Ofuton: reasonable (cos 0.73) but not magical

### Run 7: 7.7M tiny arch, Kiritan-only

Motivation: user observed Run 6 still "messy" → hypothesis model too big for 0.96h data → shrink to DiffSinger-scale.

Config: `configs/train_ja_kiritan_tiny.yaml`
- n_enc_layers: 2 (from 5)
- hidden_dim: 256 (from 512)
- n_heads: 4 (from 8)
- n_det_blocks: 3 (from 6)
- Total 7.7M params (~7x reduction)

Stopped at step 5500 after plateau.

Val trajectory (Kiritan):
| Step | Kiritan cos |
|:-:|:-:|
| 1000 | ~0.78 |
| 2500 | 0.7958 |
| 5000 | 0.7959 |

**Plateaued at 0.80**, ~same as Run 6 big model.

Audio: **worse than Run 6**. More garbled. Verdict: tiny model can't hit HuBERT clusters with enough precision.

---

## HuBERT is cluster space — key insight

User's insight after hearing Run 7 tiny:

| | Mel spectrogram | HuBERT 768-dim |
|---|---|---|
| Space | Continuous, interpolatable | High-dim cluster (discrete-like regions) |
| L1 output between classes | Blurry but decodable (low-res image) | **Garbage — RVC cannot decode "between-cluster" point** |
| Small model enough? | Yes (DiffSinger ~10M works) | **No — need precision to hit clusters** |

**Consequence**: Can't just shrink model because training budget is small. Must keep big model for cluster precision, and reduce overfit via other means (data volume, regularization, different loss).

**Phase 5 (v1 discrete K-means tokens)** was actually the right idea conceptually — CE loss enforces landing on a specific codebook entry, not averaging. But v1 was contaminated by GTSinger so results were ambiguous. **Worth re-trying on clean data**.

### Ofuton > Kiritan on tiny model — secondary observation

Run 7 tiny (Kiritan-only trained) listen result:
- Kiritan (trained): still messy
- Ofuton (never trained, 0-shot): **cleaner than Kiritan**!

Hypotheses:
1. Ofuton's phoneme-pitch distributions are more central/canonical → small model's "averaged HuBERT" lands near Ofuton's clusters by chance
2. Ofuton was pitch-corrected + denoised by original author → HuBERT targets are more textbook/canonical
3. Kiritan has more corner cases (long-tail vibrato, complex Japanese phonology edge cases) the tiny model can't fit

Takeaway: model capacity matters; data diversity matters differently than we thought.

---

## Next plan (post-restart)

User is restarting computer (GPU VRAM pressure). Post-restart plan:

1. **Back to big model** (~54M, Run 5-style arch) — HuBERT cluster precision requires it.

2. **SOFA force-alignment** for all JA datasets (SOFA = Singing-Oriented Forced Aligner, openVPI community tool). Replaces HTK/CTC/oto.ini mix with unified singing-specific alignment.
   - Why SOFA not MFA: MFA trained on speech, misaligns sustained singing. SOFA is the standard in the DiffSinger/OpenUTAU community.

3. **K-means discretization** of HuBERT features + CE loss on token indices.
   - Retrain Phase 5 approach on clean (no-GTSinger) data.
   - Core motivation: HuBERT cluster structure aligns naturally with discrete token prediction.
   - Downstream: `codebook[argmax(logits)]` → continuous HuBERT → RVC (unchanged).

4. **SP=0 force** in training loss.
   - Currently: SP frames have GT HuBERT with ||H||≈9 residue, each dataset different.
   - Fix: either (a) mask SP frames out of HuBERT loss, (b) force GT SP HuBERT to zero vector.
   - Expected to fix "rest artifacts" user heard in Run 4/5/6/7.

5. **NOT a single-dataset test anymore**. After SOFA aligns consistently and K-means discretizes away the averaging problem, merged JA data should work (the reasons cross-dataset failed disappear).
   - Ofuton-only would be under-data.
   - Kiritan + Ofuton + others MFA-aligned → plenty of data, consistent alignment.

---

## Archived checkpoints and outputs

- `archive/run6_ja_kiritan_only/` — Run 6 big Kiritan-only (best.pt + step 2500/5000/7500)
- `archive/run7_ja_kiritan_tiny/` — Run 7 tiny (best.pt + step 2500/5000)
- `test_outputs/run5_best/` — Run 5 audio (ACE/oniku/kiritan/itako/jvs_music/ofuton all)
- `test_outputs/run6_kiritan/` — Run 6 audio (kiritan/itako/ofuton)
- `test_outputs/run7_kiritan_tiny/` — Run 7 audio (kiritan/itako/ofuton)
- `docs/phoneme_contamination_audit.json` — full per-phoneme cross-dataset audit JSON

## Open pragmatic notes

- **Tiny model VRAM is 3.75GB vs big model 9.8GB** — tiny could run with much larger batch (256-384) for 2-3x speed, but didn't bother retuning since one-shot test.
- **torch.manual_seed cuDNN slowdown bug** (documented earlier in pipeline.md) — already fixed (removed from train()).
- **keep_last=3 auto-deletes checkpoints** — had a scare losing pre-overfit checkpoints in Run 5. For Run 6 onwards set up a shell watcher to archive any new step_*.pt to `archive/<run>/` before deletion. Manual pattern documented in `feedback_overfit_snapshots.md`.
