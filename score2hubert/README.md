# Score2HuBERT

**Musical Score to HuBERT Feature Synthesis via Optimal Transport Conditional Flow Matching**

A neural singing voice synthesis frontend that generates HuBERT features + F0 directly from musical scores, designed for direct injection into RVC's decoder — bypassing the traditional audio intermediate entirely.

## Architecture

```
Score (phonemes + pitch + duration + singing_ratio)
  → Score Encoder (6-layer Conformer, 512-dim)
  → Length Regulator (phoneme → frame expansion)
  → Variance Adaptor (energy, breathiness, voicing)
  → F0 Predictor (dual-mode: singing + rap)
  → Flow Matching Decoder (6-layer DiT, OT-CFM → 768-dim HuBERT)
  → RVC Decoder (net_g, frozen) → Audio
```

**Key innovations:**
- **No audio intermediate**: Score → HuBERT features → RVC decoder (no vocoder, no re-extraction)
- **OT-CFM flow matching**: 4-10 step inference vs. 50-100 for diffusion
- **Singing/rap continuum**: Continuous `singing_ratio` parameter, not binary
- **RVC compensation**: Pre-compensate for decoder smoothing during training
- **Classifier-free guidance**: Improved generation quality at inference

**Model size**: ~62M parameters | **Inference**: ~70ms for 3s audio (RTX 3080 Ti)

## Quick Start

```bash
# 1. Install dependencies
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Download datasets
python scripts/download_datasets.py --info

# 3. Preprocess
python scripts/preprocess.py --dataset ace_opencpop --raw-dir data/raw --out-dir data/processed/ace_opencpop

# 4. Train (Phase 1: Chinese pre-training on ACE-Opencpop ~130h)
python scripts/train.py --config configs/pretrain_opencpop.yaml

# 5. Fine-tune (Phase 2: Japanese)
python scripts/train.py --config configs/finetune_ja.yaml --resume checkpoints/pretrain_opencpop/best.pt

# 6. Inference
python scripts/inference.py --model checkpoints/finetune_ja/best.pt \
    --rvc-model RVC/RVC20240604Nvidia/assets/weights/Iroha_e175_s6300.pth \
    --phonemes "ja_k ja_o ja_n ja_n ja_i ja_ch ja_i ja_w ja_a" \
    --pitches "60 60 62 62 64 64 64 62 62" \
    --durations "30 30 30 30 30 30 30 30 60" \
    --output output.wav
```

## Project Structure

```
score2hubert/
├── configs/                    # YAML configurations
│   ├── base.yaml              # Base hyperparameters
│   ├── pretrain_opencpop.yaml # Chinese pre-training config
│   ├── finetune_ja.yaml       # Japanese fine-tuning config
│   └── phoneme_vocab.yaml     # Phoneme vocabulary definition
├── src/
│   ├── model/                 # Neural network architecture
│   │   ├── modules.py         # Shared modules (Conformer, DiT, AdaLN)
│   │   ├── encoder.py         # Score encoder (Conformer)
│   │   ├── flow_matching.py   # OT-CFM decoder
│   │   ├── f0_predictor.py    # Dual-mode F0 predictor
│   │   ├── variance.py        # Variance adaptor
│   │   └── score2hubert.py    # Full model assembly
│   ├── data/                  # Data loading
│   │   ├── dataset.py         # PyTorch dataset
│   │   ├── collate.py         # Batch collation with padding
│   │   └── phoneme.py         # Phoneme vocabulary management
│   ├── preprocessing/         # Feature extraction pipeline
│   │   ├── extract_hubert.py  # HuBERT feature extraction (RVC-compatible)
│   │   ├── extract_f0.py      # F0 extraction (RMVPE/Harvest/Praat)
│   │   ├── rvc_compensation.py # RVC roundtrip compensation analysis
│   │   └── prepare_dataset.py  # Dataset-specific parsers
│   ├── training/              # Training infrastructure
│   │   ├── trainer.py         # Training loop (FP16, EMA, TensorBoard)
│   │   └── losses.py          # Multi-objective loss functions
│   └── inference/             # Inference pipeline
│       └── synthesizer.py     # Score → HuBERT → RVC decoder → Audio
├── scripts/                   # Entry point scripts
│   ├── download_datasets.py   # Dataset acquisition
│   ├── preprocess.py          # Feature extraction
│   ├── train.py               # Training
│   └── inference.py           # Synthesis
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # Detailed technical architecture
│   └── TRAINING_GUIDE.md      # Step-by-step training instructions
├── data/raw/                  # Downloaded datasets
├── data/processed/            # Preprocessed features
├── checkpoints/               # Model checkpoints
├── logs/                      # TensorBoard logs
└── requirements.txt           # Python dependencies
```

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)**: Model design, feature space alignment, training strategy, theoretical analysis
- **[Training Guide](docs/TRAINING_GUIDE.md)**: Step-by-step training instructions, troubleshooting

## Training Data

| Dataset | Language | Duration | Purpose |
|---------|----------|----------|---------|
| ACE-Opencpop | Chinese | ~130h (30 singers) | Pre-training |
| GTSinger-ja | Japanese | ~80h singing + 16h speech | Fine-tuning |
| Kiritan | Japanese | ~57min | Fine-tuning |
| Ofuton-P | Japanese | ~46min | Fine-tuning |
| Opencpop | Chinese | ~5.2h (1 singer) | Alternative pre-training |

## RVC Model Compatibility

Uses the same HuBERT model as RVC (`hubert_base.pt`, Layer 12, 768-dim, 100fps) to ensure generated features are in-distribution for RVC V2 decoders. Compatible with all V2 RVC models.

## License

Research use. Datasets have their own licenses — check individual dataset terms.
