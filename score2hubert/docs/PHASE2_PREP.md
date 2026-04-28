# Phase 2: Japanese Fine-tuning — Preparation Guide

This document covers everything you need to do **before launching Phase 2 training**. Phase 2 takes the Plan C deterministic-mode model (Chinese pre-trained) and fine-tunes it on Japanese singing data.

## What you start with

- ✅ Plan C best.pt at `checkpoints/pretrain_opencpop_det/best.pt` (val_l1=0.118, cos=0.854, f0_mae=44.6 cents — see RESULTS.md Run 3)
- ✅ Audio confirmed producing recognizable lyrics (2026-04-09 morning listening test)
- ✅ Phase 2 config drafted at `configs/finetune_ja_det.yaml`
- ❌ Japanese datasets — none downloaded yet

## What you need to decide

### Choice 1: Which Japanese datasets?

Three options, in order of effort:

| Path | Datasets | Effort | Data hours | Recommended? |
|---|---|---|---|---|
| **Fast** | GTSinger ja subset only | Auto-download (~10 GB) | ~10-15h | ✅ Best first try |
| **Standard** | GTSinger ja + Kiritan + Ofuton | Manual login required for Kiritan + Ofuton | ~12h | If quality from Fast isn't enough |
| **Maximum** | All Japanese + keep Chinese | Adds ACE-Opencpop in mix | Both | If you want bilingual support, but more compute |

**Recommendation: start with Fast.** Phase 2 is iterative — 15k steps takes ~1.5h, you can listen and decide whether to escalate.

### Choice 2: Bilingual or Japanese-only?

Plan C is currently a Chinese model (best at Chinese). When you fine-tune on Japanese:

- **Japanese-only**: model becomes a Japanese model. **It will likely forget Chinese** (catastrophic forgetting is real). The Utai DAW use case is "any voice via RVC", so language matters less than you think — but if you want both languages from one checkpoint, this isn't the right approach.
- **Bilingual** (recommended): mix Chinese (ACE-Opencpop) and Japanese data during fine-tuning. Costs more compute but preserves Chinese capability. The IPA shared vocabulary makes this seamless — same model serves both.

For the **Fast** path described above, just doing GTSinger ja gets you the Japanese-only model. If you want bilingual, after preprocessing GTSinger you'd merge with the existing `data/processed/ace_opencpop` dataset (the merge tooling exists).

## Phase 2 Path A: Fast (GTSinger ja only)

### Step 1: Download GTSinger

GTSinger is on HuggingFace. Auto-download via the script:

```bash
cd D:/MyDev/MSSTRVC/score2hubert
venv/Scripts/python.exe scripts/download_datasets.py --dataset gtsinger
```

This calls `huggingface_hub.snapshot_download` with patterns filtering Japanese files. **You may need to accept the dataset's terms on HuggingFace first** (visit https://huggingface.co/datasets/GTSinger/GTSinger and click "Agree" if logged in).

Expected disk usage: ~10 GB filtered to Japanese subset (full dataset is ~29 GB).

If the auto-download fails (network issues, license issues), the fallback is git lfs clone — see the dataset description in `download_datasets.py`.

### Step 2: Preprocess GTSinger to Score2HuBERT format

```bash
venv/Scripts/python.exe scripts/preprocess.py \
  --dataset gtsinger \
  --raw-dir data/raw/gtsinger \
  --out-dir data/processed/gtsinger_ja \
  --language ja
```

This will:
1. Parse GTSinger's TextGrid alignments and metadata
2. Convert phonemes to IPA via the Japanese G2P table
3. Extract HuBERT features per audio file (slow — ~30-60 min depending on dataset size)
4. Extract F0 via RMVPE
5. Compute durations from TextGrid alignments
6. Generate train/val/test splits
7. Save `metadata.json` and per-sample `.pt` files

Expected wall time: **30-60 minutes** for ~10h of Japanese data on RTX 3080 Ti.

### Step 3: Verify preprocessing succeeded

```bash
ls data/processed/gtsinger_ja/
# expected: hubert/ f0/ phonemes/ pitch/ duration/ energy/ metadata.json vocab.yaml hubert_stats.pt

cat data/processed/gtsinger_ja/metadata.json | venv/Scripts/python.exe -c "import json,sys; d=json.load(sys.stdin); print(f'samples: {len(d)}'); print(f'splits: {set(m[\"split\"] for m in d)}')"
```

### Step 4: Sanity check the config matches

Verify `configs/finetune_ja_det.yaml`:
- `data.processed_dir` should be `"data/processed/gtsinger_ja"` (matches what you preprocessed)
- `model.encoder.n_layers` should be `4` (matches Plan C)
- `model.det_head` should be present (4 blocks, d_hidden=512)
- `training.ema.enabled` should be `false`
- `training.optimizer.lr` should be `7.0e-5`

### Step 5: Launch Phase 2 training

```bash
venv/Scripts/python.exe scripts/train.py \
  --config configs/finetune_ja_det.yaml \
  --init-from checkpoints/pretrain_opencpop_det/best.pt
```

**Important**: use `--init-from`, NOT `--resume`. They differ:
- `--init-from`: loads model weights only, fresh optimizer/scheduler/step. This is the right call when switching datasets / training phases.
- `--resume`: loads everything including optimizer state, continues exactly where left off. Wrong for Phase 2 because the LR schedule and optimizer state belong to Phase 1.

Expected wall time: **~1.5 hours** for 15k steps at batch=32. Plus ~15 min total for periodic validation.

### Step 6: Eval audio

After training completes:

```bash
venv/Scripts/python.exe scripts/eval_audio.py \
  --model checkpoints/finetune_ja_det/best.pt \
  --rvc-model RVC/RVC20240604Nvidia/assets/weights/yachiyo_runami_e350_s10500.pth \
  --output-dir test_outputs_phase2_ja \
  --num-samples 5 \
  --min-frames 350
```

Note: the test split is from the Japanese data now, so samples will be Japanese songs. Listen to `pred_full` and `h_pred_f0gt` for content quality, `gt_ceiling` for the RVC ceiling, `hgt_f0pred` for F0 quality.

## Phase 2 Path B: Bilingual (recommended for production)

If you want the model to serve both Chinese AND Japanese:

### Steps 1-3: Same as Path A, plus preprocessing is reused

The existing `data/processed/ace_opencpop` (from Phase 1) is already preprocessed in the same format. We can merge it with the new Japanese dataset.

### Step 3.5: Merge Chinese + Japanese

```bash
venv/Scripts/python.exe scripts/preprocess.py \
  --merge \
  --out-dir data/processed/multi_lang \
  --sources data/processed/ace_opencpop data/processed/gtsinger_ja
```

This creates a merged dataset under `data/processed/multi_lang/` with the union of all samples and re-randomized train/val/test splits.

### Step 4: Update config

Edit `configs/finetune_ja_det.yaml`:
```yaml
data:
  processed_dir: "data/processed/multi_lang"   # was: "data/processed/gtsinger_ja"
```

### Step 5: Launch (and bump steps for the larger dataset)

The merged dataset is ~10× larger than GTSinger ja alone (because ACE-Opencpop is 100k samples, GTSinger ja is ~10k). You'll want more training steps to see all the data.

Edit `configs/finetune_ja_det.yaml`:
```yaml
training:
  total_steps: 25000      # was 15000
  max_steps: 25000        # was 15000
```

Then launch as in Path A Step 5.

Expected wall time: **~2.5 hours** for 25k steps.

## Phase 2 Path C: Add Kiritan + Ofuton (manual download)

GTSinger covers ~10h of Japanese. Adding Kiritan (~57 min) + Ofuton (~46 min) gives more singer diversity, especially male voice (Ofuton is male, GTSinger ja is mixed but mostly female).

The catch: both require **manual download with login**.

### Kiritan

1. Audio: visit https://zunko.jp/kiridev/login.php — requires Twitter/X account login. 50 songs, ~57 minutes.
2. Extract to `data/raw/kiritan/`. Expected structure:
   ```
   kiritan/
   ├── wav/
   ├── mono_label/      # HTS phoneme labels
   ├── midi_label/
   └── musicxml/
   ```
3. Labels (MIDI + phonemes) — auto-clone:
   ```bash
   venv/Scripts/python.exe scripts/download_datasets.py --dataset kiritan
   ```
4. Preprocess:
   ```bash
   venv/Scripts/python.exe scripts/preprocess.py \
     --dataset kiritan \
     --raw-dir data/raw/kiritan \
     --out-dir data/processed/kiritan
   ```

### Ofuton

1. Audio: visit https://sites.google.com/view/oftn-utagoedb — fill out the form, wait for email with download link. 46 songs, ~46 min, male voice.
2. Extract to `data/raw/ofuton/`. Expected structure:
   ```
   ofuton/
   ├── wav/
   ├── lab/             # phoneme labels
   ├── mid/             # MIDI files
   └── musicxml/
   ```
3. Preprocess:
   ```bash
   venv/Scripts/python.exe scripts/preprocess.py \
     --dataset ofuton \
     --raw-dir data/raw/ofuton \
     --out-dir data/processed/ofuton
   ```

### Merge all three Japanese sources

```bash
venv/Scripts/python.exe scripts/preprocess.py \
  --merge \
  --out-dir data/processed/ja_merged \
  --sources data/processed/gtsinger_ja data/processed/kiritan data/processed/ofuton
```

Then point `configs/finetune_ja_det.yaml`'s `data.processed_dir` to `"data/processed/ja_merged"`.

## Hyperparameters worth tuning if Path A doesn't work

If the initial Phase 2 run doesn't produce good Japanese audio:

| Issue | Suggested adjustment |
|---|---|
| Catastrophic forgetting (Chinese got wiped) | Switch to Path B (bilingual) |
| Underfit (recon_l1 stays high, train ≈ val both bad) | Bump `total_steps` to 25-30k, maybe LR up to 1e-4 |
| Overfit (train recon_l1 drops fast, val plateaus or rises) | Lower LR to 5e-5, enable more aggressive augmentation |
| Audio is muddy / blurry but content is right | Already at det head capacity ceiling — try Phase 2 hybrid (det warm-start + flow refiner). See RESULTS.md for the pattern. |
| Japanese phoneme accuracy is bad | Verify phoneme G2P mapping in `src/data/phoneme.py:openjtalk_to_ipa()` is correct for the dataset. May need dataset-specific tweaks. |

## Pre-flight checklist

Before running Step 5 of any path, verify:

- [ ] `data/processed/<your_choice>/metadata.json` exists and has > 100 samples
- [ ] `data/processed/<your_choice>/hubert/*.pt` files are present (a few MB each)
- [ ] `configs/finetune_ja_det.yaml`'s `data.processed_dir` matches the path above
- [ ] `checkpoints/pretrain_opencpop_det/best.pt` exists (~1.4 GB)
- [ ] GPU has ~3 GB free (training will use ~2 GB)
- [ ] `nvidia-smi` shows no other processes hogging the GPU

## Things to watch during Phase 2 training

The same things as Plan C, plus:

- **First val should immediately give train-aligned numbers** (val_l1 ≈ train recon_l1). If they diverge, you may have re-introduced an EMA-style issue or some other initialization bug. Stop and diagnose.
- **f0_mae should start near 50 cents** (where Plan C left off, or slightly higher because val data is now Japanese, not Chinese) and improve over training.
- **Step time should be ~280-340 ms** (same as Plan C — we're using the same architecture).
- **VRAM should be ~2-3 GB** (similar to Plan C, maybe slightly less because det_head is no longer fresh).

If anything looks off, refer to `scripts/diag_live_vs_ema.py` and `scripts/diag_trainer_validate.py` — they're general-purpose diagnostics that work for any training run on this codebase.
