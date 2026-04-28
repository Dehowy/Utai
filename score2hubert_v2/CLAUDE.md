# Score2HuBERT v2 — Claude Code instructions

This file is auto-loaded into any Claude Code session in this repo. Contains
project state, key paths, don'ts, and working conventions based on:

- **v1** (`D:/MyDev/MSSTRVC/score2hubert/`, 2026-04-07 → 2026-04-11) = 5 phases / 4 days
- **v2** (this repo, 2026-04-11 → 2026-04-17) = **9 training runs / 6 days** (under 1 week)

Total: 10 days, 14 training attempts. v2 Run 9 shipped 2026-04-17.

## Project state (2026-04-17)

**Run 9 complete, model shipped to `archive/run9_final/best.pt`** (val_loss 0.1887 at step 50K).

- JA val_cos_sim = **0.8159** (+0.022 over Run 8's 0.794)
- ZH val_cos_sim = 0.8675
- JA f0_mae = 45.4c
- **User listening verdict: "非常接近 ceiling, 满意"** 🎉

Run 9 solved Run 8's 3 remaining issues:
1. **ACE labeling inconsistency** → SOFA-Mandarin realigned 100K ACE samples (100% OK rate)
2. **Pitch[SP] MIDI leakage (electric noise root cause)** → fixed at source via `scripts/fix_sp_pitch.py`
3. **Over-smoothing from consist + sustained weight** → removed sustained, consist 0.02→0.005 + d<5 exempt

See `docs/SESSION_2026_04_16-17_run9.md` for full narrative.

## Default next actions (if new session)

If user says "continue" or asks what's next without specifics:
1. **DAW integration priority** — deploy Run 9 model into Utai SynthPage
2. **Default mute strategy**: post OR both (see section below)
3. **Not retraining** unless user asks — plateau is real, need new data not more steps

## Deployment mute strategy (user directive 2026-04-17)

Default: **post or both** (post mute via `apply_midi_mute.py` with current settings
— only 音尾 fade-out, NO 音头 fade).

**DAW integration**: expose as advanced option in UI so users with high-quality backends
can override to `raw` or `inline` (no mute at all, or inline-only). Defaults stay
conservative (post/both) because most users will benefit from SP cleanup.

**Never apply 音头 fade** — produces audible GLITCH (not softening, actual transient
artifact). See `~/.claude/projects/D--MyDev-MSSTRVC/memory/feedback_post_mute_audio_head.md`.

## Key paths

```
D:\MyDev\MSSTRVC\score2hubert_v2\            # project root
├── archive/run9_final/                       # ⭐ shipped model
│   ├── best.pt                               # main ckpt for deployment
│   ├── train.yaml                            # Run 9 config
│   └── snapshots/                            # 6 ckpts across training
├── configs/train_run9.yaml                   # active config
├── scripts/
│   ├── train.py                              # launcher (--config --resume)
│   ├── eval_audio.py                         # 4 internal variants per sample
│   ├── apply_midi_mute.py                    # post mute (音尾 only)
│   ├── preprocess_sofa_realign_ace.py        # ACE SOFA realign
│   ├── preprocess_sofa_realign.py            # JA 5 singer datasets
│   ├── fix_sp_pitch.py                       # parallel SP pitch=0 fix
│   ├── diag_phoneme_contamination_v2.py      # 5-angle audit (cos + variance)
│   ├── run9_final_eval.sh                    # 4-way mute A/B
│   └── run9_mid_eval.sh                      # CPU mid-training audio eval
├── src/training/trainer.py                   # OOM handler, per-lang val, gap warn
├── data/processed/                           # train tensors (IPA phonemes)
└── data/processed_pre_sofa/                  # pre-SOFA backups (JA + ACE)

D:\MyDev\MSSTRVC\SOFA\                        # forced aligner (separate venv)
├── ckpt/JPN_Test2_Plus/                      # Greenleaf JA
├── ckpt/Mandarin/pretrained_mandarin_singing # qiuqiao ZH
└── segments/sofa_run/<dataset>/htk/phones/   # aligned output

D:\MyDev\MSSTRVC\RVC\RVC20240604Nvidia\
└── assets/weights/
    ├── yachiyo_runami_e350_s10500.pth        # main JA singer
    └── (ACE-singer1 for ZH — in archive/run8_sofa_aug/rvc_ace_singer1/)
```

## Don'ts (painful lessons 2026-04-09 → 2026-04-17)

- **Don't reintroduce GTSinger** — audio ceiling itself was muddy. See `memory/project_score2hubert_gtsinger_dropped.md`.
- **Don't add JVS to train** — 100-singer blended centroid catastrophically contaminates shared encoder (val-only is fine, we removed even val in Run 9).
- **Don't shrink model below 54M** — Run 7 showed 7.7M → HuBERT cluster-miss → garbled audio.
- **Don't re-add `torch.manual_seed()` in `trainer.train()`** — triggers cuDNN RNN slow path 18× on BiLSTM backward.
- **Don't optimize per-sample Python loops in trainer without a numerical equivalence test** — silent bug risk (wrong frame↔phoneme mapping → SP/pos targets misaligned → training looks fine but model learns wrong).
- **Don't trust aggregate val loss alone** — always cross-check per-lang + `eval_per_lang_v2.py`.
- **Don't mute 音头** in `apply_midi_mute.py` — rising fade or note-onset ramp = audible GLITCH.
- **Don't reshape data assumptions mid-run** — editing trainer.py during training has no effect (Python import cache), and any bug found mid-run means stop + restart, not warm-edit.
- **Don't add ACE arrow rows with `len(audio_bytes) < 100`** — they're intentionally invalid segments (2026001XXX range has 510 zero-byte WAVs).

## Dos (validated working patterns)

- **Snapshot proactively when plateau detected** — `archive/run9_snapshots/` pattern, in addition to auto-saved step_N.pt.
- **Per-dataset min-frames in eval_audio.py** — ACE use 200 (most <400 frames), JA singer 400-1000.
- **Length-deterministic sample selection** — eval_audio picks longest-N deterministic (same samples across runs = direct A/B).
- **Backup tensors via `data/processed_pre_sofa/` before realigning** — rollback cheap.
- **Run CPU eval during training via `--device cpu`** — minimal GPU contention with concurrent training.
- **Mid-training listening around step ~15-25% trained** — predictive of final quality (Run 9 @ step 10K already exceeded Run 8 final).
- **Kill python via `taskkill //F //PID <pid>` in git-bash** (use double-slash for flags).

## User collaboration style (per prior memory)

- **Honest tradeoff tables, not hype**. User catches flawed reasoning; walk it back directly when called.
- **Terse responses**. No trailing summary unless requested.
- **Push machine hard** (no VRAM buffer); user closes GUI before training. If spike causes OOM, fix the spike don't leave headroom.
- **Per-lang metrics > aggregate** — aggregate hides minority-lang regression.
- **Verify subset before scale** — esp. data preprocessing (dry-run with --limit before full pass).
- **Long audio samples** (12-20s) for evaluation listening. Save to per-domain subdirs.
- **Chinese conversation, English technical terms comfortable**.

## Standard commands

```bash
# Launch training
cd D:/MyDev/MSSTRVC/score2hubert_v2 && \
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 venv/Scripts/python.exe -u \
  scripts/train.py --config configs/train_run9.yaml

# Final eval with 4-way mute
bash scripts/run9_final_eval.sh

# Re-apply post mute (if CHANGE to apply_midi_mute.py)
bash scripts/run9_remute.sh

# Per-lang eval on a checkpoint
PYTHONIOENCODING=utf-8 venv/Scripts/python.exe scripts/eval_per_lang_v2.py \
  --ckpt archive/run9_final/best.pt --config configs/train_run9.yaml \
  --out-json archive/run9_final/eval_best.json

# Contamination audit
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 venv/Scripts/python.exe -u \
  scripts/diag_phoneme_contamination_v2.py --tag <your_tag>

# Launch SOFA (if re-aligning ACE)
bash scripts/run_sofa_mandarin_ace.sh

# Realign (ACE or JA)
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 venv/Scripts/python.exe -u \
  scripts/preprocess_sofa_realign_ace.py --workers 8
```

## Memory system

User-level memory lives at `C:\Users\admin\.claude\projects\D--MyDev-MSSTRVC\memory\`.
Always check `MEMORY.md` (auto-loaded index) for pointers to:
- `project_score2hubert.md` — project overview + runs table
- `project_score2hubert_run9_plan.md` — Run 9 plan + results
- `project_score2hubert_next_session.md` — current handoff state
- `project_score2hubert_deployment_plan.md` — mute strategy + path A/B
- `feedback_*.md` — user preferences / don'ts

If memory contradicts code: trust code (memory can go stale). Update stale memory when noticed.
