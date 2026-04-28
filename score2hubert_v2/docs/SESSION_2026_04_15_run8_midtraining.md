# Session 2026-04-15: Run 8 mid-training observations

Second half of 2026-04-15. Continues from `SESSION_2026_04_14-15_run8.md` (launch session).

## TL;DR

1. Training running smoothly. @ step ~30K (40%) as of last check ~14:00. No OOM in 30K steps. Speed 0.6-0.7 step/s (below the 1.5 estimated; CPU-side Python loops in trainer helpers are the bottleneck). ETA ~16h to 75K (~2026-04-16 06:00 completion).
2. Claude Code session interrupted ~10:00 due to Anthropic-side outage. User manually snapshotted step_5K/10K/15K/20K.pt to `archive/run8_sofa_mixed/` during downtime (proactive pre-JA-drift preservation per `feedback_overfit_snapshots.md`). Training process itself was unaffected. Session resumed on Anthropic-side recovery; Claude-side context was reloaded from memory docs (which proved crucial — cf. 2026-04-14-15 journal practice of updating memory eagerly).
3. Mid-training CPU audio eval @ step 25000 across 6 datasets × 2 samples = 48 wavs. 8845HS delivered 23 seconds per sample-batch despite 6 parallel eval processes. Training continued unaffected (briefly dropped to 0.3 step/s during CPU contention, bounced back).
4. User listening test identified 3 residual issues: rest-frame artifacts (primary), articulation 10-20% unclear (minor), JA F0 drift (minor). All other Run 5 problems (long-tail 口胡, cross-lang accent, etc.) appear resolved. See `memory/project_score2hubert_run8_listening.md` for full breakdown.
5. New design artifacts: **dual-path deployment (Path A: user F0, Path B: model F0)** + **WORLD post-hoc F0 rescue fallback**. See `memory/project_score2hubert_deployment_plan.md`. EN data (SingStyle111) dropped — dataset 404'd after Dr Dai's thesis defense.

---

## Training trajectory

### JA cos_sim — plateau since step 7500

| step | JA val_cos | JA gap | JA train_cos |
|---|---|---|---|
| 2500 | 0.7731 | +0.028 | 0.8014 |
| 5000 | 0.7918 | +0.059 | 0.8504 |
| 7500 | **0.7947** ← peak | +0.077 | 0.8718 |
| 10000 | 0.7949 | +0.090 | 0.8846 |
| 12500 | 0.7938 | +0.093 | 0.8871 |
| 15000 | 0.7947 | +0.101 ⚠️ | 0.8954 |
| 17500 | 0.7942 | +0.107 | 0.9010 |
| 20000 | 0.7945 | +0.109 | 0.9030 |
| 22500 | 0.7962 | +0.109 | 0.9056 |
| 25000 | 0.7944 | +0.115 | 0.9094 |
| 27500 | 0.7950 | +0.118 | 0.9127 |
| 30000 | 0.7935 | +0.119 | 0.9122 |

**Interpretation:** JA val stuck at 0.794±0.002 for 22.5K steps (10x val interval plateau!). Train_cos continuing to climb means model is memorizing aug variants, but val set (orig samples only) hit capacity. Gap widens monotonically. This is CLASSICAL early overfit pattern, but the CRITICAL feature is **val doesn't drop** — just plateau. So not "pre-drift snapshot critical yet," more like "JA has maxed out its data" by step 7500.

**Why 7500 and not later:** model has 8.5K JA train samples (3.2K orig × 3 aug). With aug_p+2 / aug_p-2 sharing phoneme sequences with orig, unique phoneme×pitch combinations in JA training are actually closer to ~3.5K (the aug only diversifies pitch). Step 7500 × batch 112 × 22% JA share ≈ 185 effective JA exposures per sample → saturation. Aug did delay vs raw 3.2K, but ceiling still shows.

### ZH cos_sim — still climbing

| step | ZH val_cos | ZH gap |
|---|---|---|
| 2500 | 0.8003 | +0.003 |
| 5000 | 0.8343 | +0.002 |
| 7500 | 0.8446 | +0.005 |
| 10000 | 0.8499 | +0.003 |
| 12500 | 0.8524 | +0.003 |
| 15000 | 0.8552 | +0.004 |
| 17500 | 0.8569 | +0.004 |
| 20000 | 0.8588 | +0.004 |
| 22500 | 0.8599 | +0.005 |
| 25000 | 0.8611 | +0.005 |
| 27500 | 0.8620 | +0.004 |
| 30000 | 0.8627 | +0.004 |

**Interpretation:** ZH gap essentially 0 — 81K samples of ACE is enough volume for 54M model to generalize. +0.001 per 2.5K steps at step 30K → extrapolates to ~0.88 by step 75K (matches/exceeds Run 5's 0.873 @ step 62500).

### F0 trajectories

| step | ZH f0_mae | JA f0_mae |
|---|---|---|
| 2500 | 138.4c | 51.9c |
| 10000 | 91.8c | 49.2c |
| 20000 | 76.2c | 49.6c |
| 30000 | 66.1c | 49.6c |

ZH F0 dropping steadily (138 → 66c, better than Run 5's 58c on ACE given F0 quality from model, will likely hit 50-55c by 75K). JA F0 plateau at ~50c but Run 5 was 53-55c — Run 8 JA F0 is numerically BETTER than Run 5 but qualitatively user hears "偶尔跑调" on a few samples = worst-case perceptible drift still there.

### Checkpoint trajectory

| File | Step at save | Notes |
|---|---|---|
| `archive/run8_sofa_mixed/step_5000.pt` | 5000 | User snapshot, pre-plateau ingress |
| `archive/run8_sofa_mixed/step_10000.pt` | 10000 | User snapshot, early plateau |
| `archive/run8_sofa_mixed/step_15000.pt` | 15000 | User snapshot, mid plateau |
| `archive/run8_sofa_mixed/step_20000.pt` | 20000 | User snapshot |
| `checkpoints/best.pt` | step 30000 @ last check | Latest best by total val loss |
| `checkpoints/latest.pt` | step 30000 @ last save | Latest checkpoint |
| `checkpoints/step_5000.pt` | 5000 | Live copy (keep_last=15 retained) |
| `checkpoints/step_{10K,15K,20K,25K}.pt` | ... | Live copies |

All 4 user-archived snapshots have live mirrors in checkpoints/ still (keep_last=15 handles this).

---

## Claude outage handling

Anthropic service went down ~10:00 for an unknown duration. User couldn't access Claude Code. Training wasn't Claude-dependent — it's just a Python process running under Windows Python 3.10 via venv's launcher.

During the outage:
- User remembered the `feedback_overfit_snapshots.md` + `next_session.md` protocol
- Manually copied `checkpoints/step_{5000,10000,15000,20000}.pt` to `archive/run8_sofa_mixed/`
- Training continued to step 25K by the time Claude returned

On resume, Claude re-read memory (project_score2hubert.md + run8_design.md + next_session.md) to restore context. Journey narrative in session docs + memory journal format proved useful for rapid context rebuild after multi-hour gap.

**Lesson: memory journaling strategy is load-bearing for long runs.** The practice of keeping memory + docs current before-the-fact pays off on unplanned interruptions.

---

## Mid-training listening test

Triggered by user curiosity after JA plateau was confirmed to be actual plateau (not drift). Wanted to hear what step 25K actually sounds like.

### Logistics

- GPU busy with training (98% VRAM, 0.6-0.7 step/s)
- Can't afford to kill training to free GPU for eval
- `eval_audio.py` supports `--device cpu`
- 8845HS = 8 physical cores / 16 threads, Zen4 arch — CPU fallback was cheap
- Tried 1 sample smoke test first — 23s wall time → totally practical
- Scaled to 6 datasets × 2 samples each (12 samples × 4 variants = 48 wavs) + later added ACE sample 3-6 (4 more samples)
- All parallel CPU jobs complete in ~3 min total wall time
- Training briefly contested CPU (speed dropped 0.7 → 0.3 step/s during eval), recovered to 0.6-0.7 after eval finished

### User listening findings (summary; full in memory)

Verdict in user's words: "效果还不错 现在刚25000步就已经接近 run5best 的水平了 ...现在的症状表现和之前小模型测试时候的欠训练效果类似"

Translation: @ 33% training, Run 8 is already roughly at Run 5 best's perceived quality. Residual issues look like "undertrained" symptoms, not fundamental failures.

**Residual issues identified:**
1. Rest-frame artifacts (ZH 电流, JA 乱码) — primary
2. Articulation 10-20% unclear — minor, already better than Run 5
3. JA F0 drift on some samples — minor, sample 5 "first long note" is canonical hardest case

**Confirmed resolved from Run 5:**
- Long-tail 口胡 — gone ⭐
- Cross-lang accent — gone (JA IPA transfer works)
- Phoneme garbling — gone

**User's self-caveats on JA > ZH articulation observation**:
- User not JA native speaker
- RVC model trained on JA → produces 微口音 on ZH (orthogonal to Score2HuBERT)
- Combined bias likely explains perceived JA > ZH delta

### Convergence predictions (my estimates, memorized)

- Rest artifacts: 75%+ resolve in remaining 50K steps (SP=0 penalty linear, SP pred → 0 is achievable)
- Articulation: 50-60% improve (pos_aux already converged, remaining is encoder slow refinement)
- JA F0 drift: 30-40% improve (no targeted mechanism, relies on natural training)

---

## Design artifacts from discussion

### Dual-path deployment

User pointed out: real usage can provide F0 curves (either hand-drawn or MIDI-derived). Model doesn't HAVE to predict F0 itself.

**Path A — user provides F0** (for experienced调教手 users who can draw curves):
- `h_pred_f0gt` variant, user F0 replaces GT F0
- Bypasses F0Predictor → no F0 drift risk
- Architecture already supports this (score2hubert.py forward takes `f0_target: torch.Tensor`)

**Path B — model predicts F0** (for casual users):
- `pred_full` variant, model generates F0
- F0Predictor natively learns singing style (vibrato, portamento, slides) because training target F0 is real sung F0 from RMVPE, not quantized MIDI
- Required for "users who can't draw F0 curves well"

Single model, one UI toggle. See `memory/project_score2hubert_deployment_plan.md`.

### WORLD post-hoc F0 rescue

User's "古典鬼畜调教" idea: pyworld decompose/synth to correct F0 post-hoc without retraining. Same pipeline as Run 8's augmentation — proven reliable (cos 0.95+ preserved under pitch shift).

Two variants:
- Hard snap: Replace drift with exact MIDI + vibrato amplitude
- Soft anchor: `corrected_f0 = pred_f0 - low_pass(pred_f0 - midi_hz) * alpha` — alpha in [0, 1] as DAW slider

Implementation cost: ~100 lines in render pipeline. No model changes. See `memory/project_score2hubert_deployment_plan.md` for code sketch.

### EN data dropped

SingStyle111 (Dr Dai's thesis dataset) was the intended EN source. It was removed from her GitHub + personal website after thesis defense. Email to author unanswered. User decision: drop EN. Users who want English can roman/katakana-spell to a JA voice model (acceptable workaround given product scope).

---

## Run 9 contingency tree (if Run 8 final listening surfaces issues)

Documented in `memory/project_score2hubert_next_session.md` section "Decide launch vs Run 9":
- Rest artifacts NOT resolved → stronger SP=0 force / lr tweak
- JA F0 drift persists → Path A deploy OR WORLD rescue OR fine-tune F0Predictor with anchor loss (4h train, freeze rest)
- Articulation degrades → unexpected, needs diagnosis
- Otherwise → LAUNCH

Expected most-likely outcome based on mid-training trajectory: **LAUNCH** with dual-path UI + optional WORLD rescue.

---

## Operational updates for memory

- `scripts/train.py` has `os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')` set at top (added 2026-04-15). Must be before `import torch`. Reduces CUDA fragmentation for future runs.
- VRAM rule-of-thumb updated in `project_score2hubert_pipeline.md`: target peak ≤ 11.0 GB (1.2 GB headroom). Run 8 sat at 10.3 GB peak / 11.4 GB reserved → 300-800 MB headroom depending on background apps.
- `memory/project_score2hubert_run8_listening.md` (new, 2026-04-15) — user listening feedback + residual issues + convergence predictions
- `memory/project_score2hubert_deployment_plan.md` (new, 2026-04-15) — dual-path + WORLD rescue plan
- `memory/project_score2hubert.md` — updated with mid-training state
- `memory/project_score2hubert_next_session.md` — updated with mid-training resume procedure + Run 9 contingency tree

## Backend process state (for restart recovery)

- Training process: `D:\MyDev\MSSTRVC\score2hubert_v2\venv\Scripts\python.exe scripts/train.py --config configs/train_run8.yaml`
- Wrapper pid 13844 (venv Python launcher) + actual pid 5264 (system Python) + 8 DataLoader worker subprocesses
- Launched: 2026-04-15 00:27:21
- Log file: `D:\MyDev\MSSTRVC\score2hubert_v2\logs\train.log`
- Output file (Claude-side background): tasks `baqvx2f7q.output` under `%TEMP%\claude\D--MyDev-MSSTRVC-score2hubert-v2\9cb4fe01.../tasks/`

If process dies mid-run:
```
cd D:/MyDev/MSSTRVC/score2hubert_v2
py -3.10 scripts/train.py --config configs/train_run8.yaml --resume checkpoints/latest.pt
```
Resume works thanks to trainer's `load_checkpoint()` restoring optimizer + scheduler + scaler + global_step.
