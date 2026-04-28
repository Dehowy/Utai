# Score2HuBERT Training Guide

## Prerequisites

### Hardware
- GPU: NVIDIA RTX 3080 Ti (12GB VRAM) or better
- RAM: 32GB+ recommended
- Storage: ~50GB free (datasets + checkpoints)

### Software Environment
```bash
# Python 3.10 (match main project)
# CUDA 12.1 + PyTorch 2.5.1 (match main project)

# Create isolated environment (recommended)
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash

# Install dependencies
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# fairseq (for HuBERT loading, install separately)
pip install fairseq --no-deps

# Montreal Forced Aligner (optional, for phoneme alignment)
# conda install -c conda-forge montreal-forced-aligner
```

### Environment Dependencies Installed
| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.5.1+cu121 | Core deep learning framework |
| torchaudio | 2.5.1 | Audio processing |
| fairseq | ≥0.12.2 | HuBERT model loading |
| torchdiffeq | ≥0.2.3 | ODE solvers for flow matching |
| omegaconf | ≥2.3.0 | Configuration management |
| librosa | ≥0.10.0 | Audio feature extraction |
| soundfile | ≥0.12.0 | Audio I/O |
| pyworld | ≥0.3.2 | F0 extraction (Harvest) |
| praat-parselmouth | ≥0.4.2 | F0 extraction (Praat) |
| pyopenjtalk | ≥0.3.0 | Japanese G2P |
| pypinyin | ≥0.49.0 | Chinese pinyin conversion |
| lightning | ≥2.2.0 | Training utilities |
| tensorboard | ≥2.16.0 | Training visualization |
| matplotlib | ≥3.8.0 | Plotting |

---

## Step 1: Download Datasets

```bash
# View available datasets and instructions
python scripts/download_datasets.py --info

# Auto-download GTSinger (from HuggingFace)
python scripts/download_datasets.py --dataset gtsinger

# Opencpop, Kiritan, Ofuton-P require manual download
# Follow the instructions printed by:
python scripts/download_datasets.py --dataset opencpop
python scripts/download_datasets.py --dataset kiritan
python scripts/download_datasets.py --dataset ofuton
```

After downloading, verify directory structure:
```
data/raw/
├── opencpop/
│   ├── wavs/               # 3756 WAV files
│   └── segments/transcriptions.txt
├── kiritan/
│   ├── wav/                # WAV files
│   └── mono_label/         # .lab phoneme labels
├── ofuton/
│   ├── wav/
│   └── lab/
└── gtsinger/
    └── {singer}/{song}/audio.wav + annotation.json
```

---

## Step 2: Preprocess Datasets

### 2.1 Process Each Dataset

```bash
# Opencpop (Chinese, for pre-training)
python scripts/preprocess.py --dataset opencpop \
    --raw-dir data/raw/opencpop \
    --out-dir data/processed/opencpop

# Kiritan (Japanese)
python scripts/preprocess.py --dataset kiritan \
    --raw-dir data/raw/kiritan \
    --out-dir data/processed/kiritan

# GTSinger Japanese
python scripts/preprocess.py --dataset gtsinger \
    --raw-dir data/raw/gtsinger \
    --out-dir data/processed/gtsinger \
    --language ja
```

Each command will:
1. Parse dataset annotations
2. Extract HuBERT features (Layer 12, 768-dim, 100fps) using RVC's hubert_base.pt
3. Extract F0 using RMVPE (matching RVC)
4. Convert phonemes to unified vocabulary
5. Compute durations in frames
6. Assign train/val/test splits
7. Save metadata.json + per-sample .pt files

### 2.2 Merge Datasets

```bash
# Merge all Japanese datasets for fine-tuning
python scripts/preprocess.py --merge \
    --out-dir data/processed/ja_merged \
    --sources data/processed/kiritan data/processed/gtsinger

# (Opencpop is used separately for pre-training, no need to merge)
```

### 2.3 RVC Compensation (Optional but Recommended)

```bash
# Analyze RVC's feature transformation and create compensated targets
python scripts/preprocess.py --rvc-compensation \
    --hubert-dir data/processed/ja_merged/hubert \
    --f0-dir data/processed/ja_merged/f0 \
    --rvc-model RVC/RVC20240604Nvidia/assets/weights/Iroha_e175_s6300.pth \
    --out-dir data/processed/ja_merged/hubert_compensated \
    --alpha 0.35
```

This will:
1. For each sample, feed HuBERT features + F0 into RVC's decoder
2. Re-extract HuBERT from the reconstructed audio
3. Compute the offset (what net_g loses)
4. Save adjusted features: `H_target = H_gt + 0.35 × (H_gt - H_roundtrip)`
5. Output compensation statistics for analysis

After compensation, update the config to use compensated features:
```yaml
# In configs/finetune_ja.yaml
data:
  processed_dir: "data/processed/ja_merged"
  # Swap hubert directory to compensated version
  # (handled automatically if compensation dir is named "hubert_compensated")
```

---

## Step 3: Train

### 3.1 Phase 1: Chinese Pre-training

```bash
python scripts/train.py --config configs/pretrain_opencpop.yaml
```

Expected: ~4-6 hours on RTX 3080 Ti for 100k steps.

Monitor training:
```bash
tensorboard --logdir logs/
```

Key metrics to watch:
- `train/flow_loss`: should decrease steadily
- `train/f0_loss`: should converge within ~10k steps
- `val/total_loss`: should track training loss without divergence
- `val/f0_mae_cents`: F0 accuracy in cents (target: <50 cents)

### 3.2 Phase 2: Japanese Fine-tuning

```bash
python scripts/train.py \
    --config configs/finetune_ja.yaml \
    --resume checkpoints/pretrain_best.pt
```

Expected: ~2-3 hours for 50k steps.

### 3.3 Quick Test Run

To verify everything works before long training:
```bash
python scripts/train.py --config configs/base.yaml \
    --override training.max_steps=100 training.val_every=50 training.log_every=10
```

---

## Step 4: Evaluate & Inference

### 4.1 Generate Test Samples

```bash
# Basic synthesis (HuBERT features only)
python scripts/inference.py \
    --model checkpoints/best.pt \
    --phonemes "ja_k ja_o ja_n ja_n ja_i ja_ch ja_i ja_w ja_a" \
    --pitches "60 60 62 62 64 64 64 62 62" \
    --durations "30 30 30 30 30 30 30 30 60" \
    --output test_output.wav

# Full pipeline with RVC
python scripts/inference.py \
    --model checkpoints/best.pt \
    --rvc-model RVC/RVC20240604Nvidia/assets/weights/Iroha_e175_s6300.pth \
    --phonemes "ja_k ja_o ja_n ja_n ja_i ja_ch ja_i ja_w ja_a" \
    --pitches "60 60 62 62 64 64 64 62 62" \
    --durations "30 30 30 30 30 30 30 30 60" \
    --steps 10 --solver midpoint --cfg-scale 2.0 \
    --output test_with_rvc.wav

# Rap mode test
python scripts/inference.py \
    --model checkpoints/best.pt \
    --rvc-model RVC/RVC20240604Nvidia/assets/weights/Iroha_e175_s6300.pth \
    --singing-ratio 0.0 \
    --phonemes "ja_k ja_o ja_n ja_n ja_i ja_ch ja_i ja_w ja_a" \
    --pitches "0 0 0 0 0 0 0 0 0" \
    --durations "20 20 20 20 20 20 20 20 40" \
    --output test_rap.wav
```

### 4.2 Inference Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--steps` | 10 | 4-50 | More steps = higher quality, slower |
| `--solver` | midpoint | euler/midpoint/rk4 | Higher order = better quality per step |
| `--cfg-scale` | 2.0 | 1.0-5.0 | Higher = stronger condition adherence |
| `--temperature` | 1.0 | 0.5-1.5 | Lower = less variation, higher = more expressive |
| `--pitch-shift` | 0 | -24 to +24 | RVC output pitch shift (semitones) |

Recommended for production: `--steps 10 --solver midpoint --cfg-scale 2.0`
Recommended for speed: `--steps 4 --solver euler --cfg-scale 1.5`
Recommended for quality: `--steps 20 --solver rk4 --cfg-scale 2.5`

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `training.batch_size` (try 8 or 4)
- Enable gradient accumulation: `training.accumulate_grad: 4`
- Reduce `data.max_frames` (try 800)

### F0 Loss Not Converging
- Check F0 extraction quality: visualize extracted F0 vs. audio
- Verify frame alignment between HuBERT and F0 (must be same length)
- Try different F0 extraction method (harvest instead of rmvpe)

### Flow Loss Plateaus
- Increase `model.flow_matching.n_layers` (try 8)
- Increase `training.optimizer.lr` slightly (try 3e-4)
- Reduce `model.flow_matching.cfg.drop_prob` (try 0.1)

### Generated Features Sound Robotic/Metallic
- Apply RVC compensation (Section 2.3)
- Increase inference steps (try 20)
- Check HuBERT feature statistics match real audio features
- Verify using correct HuBERT layer (must be Layer 12 for V2 models)

### Cross-Lingual Transfer Not Working
- Ensure unified phoneme vocabulary is used
- Verify language IDs are correct
- Try unfreezing more encoder layers during fine-tuning
- Increase Japanese dataset size (more augmentation)
