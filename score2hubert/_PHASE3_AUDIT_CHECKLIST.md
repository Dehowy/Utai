# PHASE 3 AUDIT CHECKLIST

**Status (2026-04-09 evening)**: All engineering done. Awaiting your audit.
After you OK, run the LAUNCH COMMAND at the bottom.

## What was done this session (summary)

Phase 2 was production-ready but had:
1. An F0 truncation bug on GTSinger data (pred and ceiling both off-key) — **fixed**, re-extracted F0, resume trained for 5000 steps → `checkpoints/finetune_multi_det_fix/best.pt` with val l1=0.1259/cos=0.826/f0=61.8c
2. Phase 2 fix is confirmed ≥ Plan C on ACE-Opencpop samples (better or equal) while adding ja/en/speech → Phase 2 fix is the new production baseline.
3. On GTSinger singing samples, det_head averaging produces audible "fuzzy articulation". Not a model regression — ACE-Opencpop just hid the problem.

Phase 3 engineering:
- Add `encoder.tech_proj = nn.Linear(6, d_model)` zero-init → backwards compatible with Phase 2 fix checkpoint
- Feed GTSinger's 6 per-phoneme technique flags (mix_voice, falsetto, breathy, pharyngeal, vibrato, glissando) through the encoder
- Write tech tensors for existing gtsinger_{zh,ja,en}/ via `_add_tech_flags_to_existing_gtsinger.py` (no HuBERT re-extraction needed)
- Re-merge multi_lang_4 with tech subdir
- New training config `configs/finetune_multi_det_tech.yaml` (batch 48, 8k steps, lr 5e-5, resumes from Phase 2 fix)

## Files modified

**Core model**:
- `src/model/encoder.py` — `tech_proj` linear, zero-init, forward accepts `tech` kwarg
- `src/model/score2hubert.py` — both `forward` and `synthesize` accept+pass `tech`
- `src/training/trainer.py` — both train and val steps pass `batch.get("tech")`
- `src/data/dataset.py` — loads `tech/{id}.pt` or falls back to zeros
- `src/data/collate.py` — pads `tech` to `[B, T_phone_max, 6]`

**Data pipeline**:
- `src/preprocessing/prepare_dataset.py` — new `GTSINGER_TECH_KEYS`, `N_TECHNIQUES=6`, `_expand_tech_via_gtsinger_g2p`, `_process_gtsinger_singing/speech` save `tech/*.pt`, AND F0 truncation bug fix (interpolate instead of truncate)
- `scripts/preprocess.py` — merge includes `"tech"` subdir, `merge_datasets` uses `encoding="utf-8"` for metadata.json loading, AND `shutil.copy2` replaced by `os.link` (hardlinks, fast)

**Eval/profile scripts**:
- `scripts/eval_audio.py` — loads + passes tech
- `scripts/profile_step.py` — passes `batch.get("tech")`
- `_phase2_final_eval.py` — loads + passes tech, supports `tech_override` for rap-in-song
- `_diag_phase2fix_on_acopencpop.py` — loads + passes tech (zeros for ace)

**Config**:
- `configs/finetune_multi_det_tech.yaml` — NEW Phase 3 production config

**Docs**:
- `docs/PHASE3_TECH_EMBEDDING.md` — full design doc + rationale
- `docs/RESULTS.md` — Phase 2 section added

**Temp scripts in project root** (use for re-runs):
- `_reextract_f0_gtsinger.py` — F0 bug fix (already run)
- `_add_tech_flags_to_existing_gtsinger.py` — tech population (already run)

## Data state

- `data/processed/gtsinger_zh/tech/` — 10188 files (7139 singing real flags + 3049 speech zeros)
- `data/processed/gtsinger_ja/tech/` — 2831 files (1999 singing + 832 speech zeros)
- `data/processed/gtsinger_en/tech/` — 6891 files (4827 singing + 2064 speech zeros)
- `data/processed/multi_lang_4/tech/` — 19910 hardlinks (gtsinger only; ace has no tech file → dataset falls back to zeros at load time)
- `data/processed/ace_opencpop/tech/` — doesn't exist (intentional)

## Checkpoint state

- `checkpoints/finetune_multi_det_fix/best.pt` — **Phase 2 fix, step 5000**, Phase 3 init source
- `checkpoints/finetune_multi_det_fix/final.pt` — same as best for Phase 2 fix
- `checkpoints/finetune_multi_det/best.pt` — Phase 2 original (superseded, kept for reference)
- `checkpoints/pretrain_opencpop_det/best.pt` — Plan C historical baseline

## AUDIT CHECKS

### Quick verifications (~5 min)

```bash
cd D:/MyDev/MSSTRVC/score2hubert

# 1. tech_proj is zero-init
venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, '.')
from src.model.encoder import ScoreEncoder
e = ScoreEncoder()
print('tech_proj.weight std:', e.tech_proj.weight.std().item(), '(expect 0.0)')
print('tech_proj.bias max:  ', e.tech_proj.bias.abs().max().item(), '(expect 0.0)')
"

# 2. Tech file semantic spot-check
venv/Scripts/python.exe -c "
import torch, json, sys
sys.stdout.reconfigure(encoding='utf-8')
proc = 'data/processed/gtsinger_zh'
with open(proc + '/metadata.json', encoding='utf-8') as f:
    meta = json.load(f)
# Find a Breathy_Group sample, verify breathy_tech (idx 2) is mostly 1
for m in meta:
    if 'Breathy_Group' in m.get('orig_item_name', '') and 'ZH-Alto' in m.get('orig_item_name', ''):
        t = torch.load(f'{proc}/tech/{m[\"id\"]}.pt', weights_only=True)
        print(f'Breathy_Group sample: {m[\"orig_item_name\"]}')
        print(f'  tech shape: {t.shape}')
        print(f'  per-dim sum (mix/falsetto/BREATHY/pharyngeal/vibrato/glissando): {t.sum(dim=0).tolist()}')
        break
"

# 3. multi_lang_4/tech has the right file count
ls data/processed/multi_lang_4/tech/ | wc -l
# Expected: 19910

# 4. Smoke train 100 steps (~1 min)
venv/Scripts/python.exe scripts/train.py \
  --config configs/finetune_multi_det_tech.yaml \
  --init-from checkpoints/finetune_multi_det_fix/best.pt \
  --override training.max_steps=100 training.scheduler.total_steps=100 training.val_every=100 training.save_every=100 training.log_every=25 \
    paths.checkpoint_dir=checkpoints/_smoke_phase3_audit paths.log_dir=logs/_smoke_phase3_audit
# Expect in log: "loaded 441 keys, 2 fresh (0 in det_head)"
# Expect val@100: l1 ~0.126, cos ~0.825 (matches Phase 2 fix baseline)

# Cleanup after smoke
rm -rf checkpoints/_smoke_phase3_audit logs/_smoke_phase3_audit
```

## LAUNCH COMMAND

After audit passes, run this to start full Phase 3 training (~90 min):

```bash
cd D:/MyDev/MSSTRVC/score2hubert
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 venv/Scripts/python.exe scripts/train.py \
  --config configs/finetune_multi_det_tech.yaml \
  --init-from checkpoints/finetune_multi_det_fix/best.pt \
  > logs/finetune_multi_det_tech_console.log 2>&1 &
```

Or via the Bash tool's `run_in_background: true`.

## Post-training

1. Check val metrics improved: expect l1 maybe 0.123-0.125, cos maybe 0.83-0.84
2. Run eval on the new checkpoint (edit `_phase2_final_eval.py` to point CKPT_PATH at `checkpoints/finetune_multi_det_tech/best.pt` and OUT_ROOT at `test_outputs_phase3_final`)
3. User listens to gtsinger samples, especially Breathy_Group / Vibrato_Group / Pharyngeal_Group where tech flags are densely active, and compares to `test_outputs_phase2_fix_final/`
4. If success: integrate with Utai DAW (replace DiffSinger in `app/backend/wrappers/synth_wrapper.py`)
5. If not enough: fall back to hybrid architecture (det + flow refiner)
