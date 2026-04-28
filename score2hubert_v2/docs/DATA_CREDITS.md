# Data Source Credits & Licenses

This file lists all training data sources used by Score2HuBERT v2,
their authors, and any attribution requirements.

## Active Datasets (used in training)

### ACE-Opencpop (Chinese)
- **Source**: HuggingFace dataset `espnet/ace-opencpop-segments`
- **Origin**: ACE Studio (Chinese SVS commercial product) + Opencpop singing corpus
- **License**: Research / non-commercial use
- **Singers**: 30+ synthesized voices
- **Used as**: Primary ZH training data (~90K segments, ~100h)

### UTAU Voice Banks (Japanese)
8 banks, all derived from non-commercial UTAU community voice banks:
- ChugokuUsagi (中国うさぎ) — public release
- ずんだもん 連続音 — zunko.jp
- つくよみちゃんUTAU音源 — https://tyc.rei-yumesaki.net/material/utau/ — AI training permitted
- 九州そら — zunko.jp
- 北海道めろん — zunko.jp
- 四国めたん — zunko.jp
- 大江戸ちゃんこ 連続音 — zunko.jp
- 沖縄あわも 連続音 — zunko.jp
- **License**: Non-commercial. zunko.jp (東北ずん子) requires character/source credit per the official guidelines.
- **Used as**: Primary JA training data (~18.7K segments, ~3h)

### Kiritan Singing Database (Japanese)
- **Source**: 東北きりたん歌唱データベース
- **Author**: zunko.jp (東北ずん子)
- **URL**: https://zunko.jp/
- **License**: Non-commercial research use, character attribution required.
- **Used as**: JA continuous singing supplement (~730 segments, ~0.96h post-segmentation)

### Itako Singing Database (Japanese)
- **Source**: 東北イタコ歌唱データベース
- **Author**: zunko.jp (東北ずん子)
- **URL**: https://zunko.jp/
- **License**: Non-commercial research use, character attribution required.
- **Used as**: JA continuous singing supplement (~605 segments, ~0.80h post-segmentation)

### OFUTON_P_UTAGOE_DB (Japanese) — added 2026-04-13
- **Source**: おふとんP歌声DB (same author family as ONIKU)
- **License**: Use permitted with **author attribution required**
- **Specifications**:
  - 46 songs (49 melodies, 57 audio files)
  - ~46 min 30 sec voiced content
  - HTK 100ns labels, 96kHz/24bit audio
  - **Single male singer**
  - Pre-pitch-corrected and noise-removed (same caveat as ONIKU)
- **Used as**: JA male voice supplement (~571 segments, ~0.77h)

### Natsume_Singing_DB (Japanese) — added 2026-04-13
- **Source**: https://ksdcm1ng.wixsite.com/njksofficial/enunu-nnsvs
- **GitHub (label-only)**: Natsume_Singing_DB_0713
- **Author**: 夏目悠李 (Natsume Yuli) — male voice
- **License**: 利用規約 https://ksdcm1ng.wixsite.com/njksofficial/規約-rules
- **Specifications**:
  - 52 songs, modeled after 東北きりたんDB
  - Float-seconds OpenJTalk labels (Kiritan-style)
- **Used as**: JA male voice supplement (~565 segments, ~0.85h)

### JVS-MuSiC (Japanese) — added 2026-04-13
- **Source**: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music
- **Authors**: Hiroki Tamaru, Shinnosuke Takamichi, Naoko Tanji, Hiroshi Saruwatari (UTokyo)
- **Citation**: Tamaru et al., "JVS-MuSiC: free Japanese multispeaker singing-voice corpus", arXiv:2001.07044, 2020.
- **License (audio)**: Non-commercial research/personal use. Tags CC-BY-SA 4.0.
- **Specifications**:
  - 100 singers (49 male + 51 female), professional
  - Each sings 2 songs:
    - Common: Katatsumuri (~30s each, 100 versions)
    - Unique: 26 different children's songs (each sung by 3-4 singers)
  - 24 kHz, 16-bit RIFF WAV
  - **NO phoneme alignment provided** — we did CTC alignment using
    `vumichien/wav2vec2-large-xlsr-japanese-hiragana` + `ctc-segmentation`
- **Caveats**:
  - Raw recordings have natural pitch/tempo variation (each singer in own key)
  - 100-singer diversity → multi-distribution data (good for breaking averaging)
- **Used as**: JA multi-singer diversity supplement (~181 segments, ~2.05h)
- **Pipeline**: `scripts/preprocess_jvs_music.py` (uses lyrics from `src/data/japanese_song_lyrics.py`)

### ONIKU_KURUMI_UTAGOE_DB (Japanese) — **NEW in v2 Run 4+**
- **Source**: 御丹宮くるみ歌声データベース
- **Author**: 御丹宮くるみ (Oniku Kurumi)
- **URL**: https://onikuru.info/db-download/
- **License**: Use permitted with **author attribution required**
- **Specifications**:
  - 56 songs (all from public-domain sources, mainly 童謡)
  - ~61 minutes of voiced content
  - HTK 100ns labels, 96kHz/24bit audio
  - Single female singer
  - Pitch-corrected and noise-removed by author
  - CVVC-coverage spell-singing included
- **Used as**: JA clean singing supplement (~738 segments, ~1.02h)

## Models / Pretrained Components

### HuBERT-Base
- **Source**: Facebook/Meta AI
- **Used as**: Frozen feature extractor (Layer 12, 768-dim)
- **Path in repo**: `RVC/RVC20240604Nvidia/assets/hubert/hubert_base.pt`

### RMVPE
- **Used as**: F0 extractor for training targets
- **Path in repo**: `RVC/RVC20240604Nvidia/assets/rmvpe/rmvpe.pt`

### RVC (Retrieval-based Voice Conversion)
- **Used as**: Inference-time decoder (HuBERT + F0 → audio)
- Voice models in `RVC/RVC20240604Nvidia/assets/weights/`

## Datasets Considered but Dropped

### GTSinger (zh/ja/en) — **DROPPED 2026-04-12**
- Empirically caused training degradation across multiple runs
- See `docs/SESSION_2026_04_12.md` for the diagnostic evidence
- All preprocessed data archived to `archive/gtsinger_processed_dropped/`

## Distribution & Sharing Notes

The trained Score2HuBERT v2 model is **non-commercial only** due to the
licensing constraints of:
- ACE-Opencpop (research)
- All zunko.jp data (Kiritan/Itako/UTAU banks)
- ONIKU (use permitted but attribution required)

When sharing or publishing results, **always include attribution**:
- "Trained on Kiritan/Itako/UTAU voice banks from 東北ずん子プロジェクト (zunko.jp)"
- "Trained on 御丹宮くるみ歌声データベース (https://onikuru.info/)"
- "Trained on ACE-Opencpop dataset"
