# Architecture Reset: Back to Plan C + Minimal IPA Extension

Written 2026-04-10 ~21:30. This document records the full reasoning journey and the
final architecture decision.

## The Journey (2026-04-08 → 2026-04-10)

### What we tried and what we learned

| Phase | What | Result | Key Lesson |
|-------|------|--------|------------|
| 1 (Plan A) | Flow matching on HuBERT | F0 perfect, HuBERT = "啊啊啊" | DiTBlock AdaLN ignores phoneme condition |
| 1 (Plan C) | **Deterministic det_head** | **l1=0.1183, cos=0.854, clear zh audio** | HuBERT favors regression; 80% deterministic |
| 2 | +GTSinger multilingual | l1=0.1259, cos=0.826, fuzzy + 跑调 | Human singing variance → det_head averaging exposed |
| 2 fix | F0 interpolation bug fix | Fixed pitch, articulation still fuzzy | F0 100fps vs HuBERT 50fps alignment |
| 3 | Tech embedding (6 flags) | +0.6%, barely audible | Conditioning helps marginally, not enough |
| 3 | Rap mode bug | Fixed (pitch zeroing) | Rap head never saw non-zero MIDI pitch |
| 4 v1 | Hybrid flow refiner | OT-CFM noise 5.7x too high | Noise scale must match residual std |
| 4 v2 | Hybrid (noise=0.18) | +4% det, stochastic useless | Stochastic doesn't help this task |
| 5 | Discrete 4096 tokens | **zh "能听了"**, ja 乱码 | Correct tokens are sharp; wrong tokens are worse than blur |
| 5b | Balanced training | No improvement | Frozen encoder is the ja/en bottleneck |
| 6 | F0 fine-tune | 60.2→60.0c, no change | Same data → same result |
| 6 | Pitch correction post-proc | ±100 cents clamp | Effective for preventing drift |

### The three critical diagnostics

1. **Per-phoneme singing vs speech** (same language):
   - Feature cos = 0.794, token overlap = 13.4%
   - Speech and singing are different domains in HuBERT space

2. **Per-phoneme cross-language** (singing vs singing):
   - Feature cos = **0.876**, token overlap = **6.5%**
   - Same IPA phoneme across languages is VERY similar in features
   - But 4096 tokens over-resolve the tiny differences → different tokens

3. **Quantization quality vs det_head**:
   - 4096 K-means cos ceiling = 0.834
   - Det_head cos = 0.828 (on ACE-Opencpop subset: 0.854)
   - The codebook wasn't even better than regression!

### The fundamental insight

**ACE-Opencpop det_head (Plan C) was the optimal architecture all along.**

Everything after Phase 1 was solving problems that GTSinger's human-singing variance
introduced. ACE-Opencpop is machine-synthesized with low intra-phoneme variance →
det_head's conditional mean ≈ the true value → no fuzziness.

The entire Phase 2→5 arc can be summarized as:
1. Added hard data (GTSinger) → quality degraded
2. Tried to fix degradation with fancy architectures → marginal improvements
3. Discovered the fix was to not add the hard data in the first place

This is NOT a failure — the experiments proved:
- L1 regression averaging is fundamental and can't be architected away
- Discrete tokens work for HIGH-ACCURACY languages but catastrophically fail for low-accuracy ones
- OT-CFM noise scale must match target distribution
- Cross-language IPA phonemes share HuBERT space (cos 0.88)
- User control > model prediction for expressiveness

## The New Architecture

### Pipeline

```
Score (USTX)
  → phonemes (IPA), pitch (MIDI), duration
  → User-drawn: pitch curve (with vibrato) + loudness curve
  → Score2HuBERT (Plan C det_head, ~53M trainable)
      → continuous HuBERT [T, 768] + F0 from pitch curve
  → RVC (voice conversion)
      → loudness → natural timbre variation (RVC's rms_mix_rate)
  → Final audio
```

### What stays from Phase 1

- **Encoder**: Conformer 4L, d_model=512, 37M params — unchanged
- **DetHubertHead**: 4 conv blocks, 8.3M params — unchanged
- **f0_cond_proj**: projects pitch curve to conditioning — unchanged
- **Checkpoint**: `checkpoints/pretrain_opencpop_det/best.pt` (Plan C, step 30000)
- **IPA vocabulary**: 80 tokens, shared across zh/ja/en

### What gets removed

- ~~FlowDecoder (45M)~~ — never worked for this task
- ~~FlowRefiner (5.5M)~~ — marginal improvement, not worth complexity
- ~~DiscreteHubertHead~~ — wrong approach for multi-language
- ~~Tech embedding (tech_proj)~~ — user controls vibrato/energy manually
- ~~VarianceAdaptor (energy/breathiness/voicing predictors)~~ — user draws loudness curve
- ~~f0_predictor for singing mode~~ — user draws pitch curve (optional fallback for auto mode)
- ~~GTSinger dependency~~ — ACE-Opencpop is sufficient for base model

### What's new

1. **Minimal ja/en phoneme extension**: Fine-tune ONLY the ~10 phoneme embeddings that
   are unique to ja/en (ɾ, ɴ, ɸ, ç, ɯ, θ, ð, ɹ, etc.). A few hundred samples each.
   Everything else (encoder weights, det_head weights) stays frozen.

2. **User-drawn pitch curve** (frontend):
   - Already supported via `synthesize(pitch_curve=...)` parameter
   - Vibrato = sinusoidal pitch modulation drawn by user
   - Portamento = smooth pitch transitions drawn by user
   - Eliminates f0_predictor inaccuracy entirely

3. **User-drawn loudness curve** (frontend → RVC):
   - Input loudness modulates RVC's output timbre
   - High loudness → tenser voice, more pharyngeal quality (learned from RVC training data)
   - `rms_mix_rate` parameter smooths transitions
   - Replaces: tech embedding, energy predictor, breathiness predictor

4. **Pitch correction** (optional, inference-only):
   - Already implemented: clamp predicted F0 to ±100 cents of MIDI note
   - Only needed if user doesn't draw pitch curve (auto mode)

### What remains to do

| Task | Work | Est. |
|------|------|------|
| Test Plan C on ja/en phoneme sequences directly | Zero-shot test, no training | 30 min |
| If needed: fine-tune ~10 new phoneme embeddings | Tiny dataset + freeze everything else | 1-2h |
| Wire pitch curve input in DAW frontend | Frontend JS, already has piano roll | 2-3h |
| Wire loudness curve input → RVC rms_mix_rate | Frontend + backend API | 2-3h |
| Strip unused modules from inference path | Code cleanup | 1h |
| Integration test with Utai DAW | End-to-end test | 2-3h |

Total estimated: **~1-2 days** to production, vs weeks of iterating on architectures.

## Data Plan

### Chinese (zh): ACE-Opencpop — unchanged
- 100K segments, ~130h, 30 singers, machine-synthesized
- Production quality, no changes needed

### Japanese (ja): Kiritan + Itako + UTAU voice banks
- **Kiritan singing DB**: 50 songs (3.5h), single singer, HTK mono labels + MIDI
  - Skip song 08 (MIDI issues per README)
  - Use master labels (kiritan_singing-master, community-corrected)
- **Itako singing DB**: 50 songs (1.2h), single singer, same format
  - Song 50 = phoneme supplement (covers rare phonemes: ギュ、ジェ、ツァ etc.)
  - Use master labels (itako_singing-main)
- **UTAU voice banks**: 9 singers, 5387 files (10.6h), 連続音 format
  - Multi-pitch variants (A4, B3, C5, etc.) per singer
  - Filenames encode phoneme transition chains (hiragana → IPA mapping needed)
  - ChugokuUsagi, ずんだもん, 中部つるぎ, 九州そら, 北海道めろん, 四国めたん,
    大江戸ちゃんこ, 沖縄あわも, 関西しのび
- **Total ja: ~15h** — comparable quality to ACE (low variance, controlled recording)
- All phonemes map to existing IPA vocab (no new IDs needed)

### English (en): GTSinger Control_Group only
- Filter existing gtsinger_en for `#Control_Group#` entries (~2700 samples)
- Natural singing, no extreme techniques → lower variance
- Already preprocessed, no new extraction needed

## License

**IMPORTANT — Training data usage restrictions:**

The Kiritan/Itako singing databases and associated UTAU voice banks are distributed under
Japan's amended Copyright Act Article 30-4 (改正著作権法30条の4). This permits use for
machine learning / AI training but imposes restrictions:

- **Commercial use**: Restricted to Japan's Tohoku region (東北地方) only
- **Non-commercial use**: Unrestricted
- **VTuber / virtual streamer use**: Unrestricted
- **Model distribution**: The trained Score2HuBERT model is open-source under Apache 2.0
  terms EXCEPT that **commercial use is prohibited** due to the training data restrictions

Citation for Kiritan:
> I. Ogawa and M. Morise: Tohoku Kiritan singing database: A singing database for
> statistical parametric singing synthesis using Japanese pop songs, Acoustical Science
> and Technology, vol. 42, no. 3, pp. 140-145, May 2021.

Citation for Itako:
> https://zunko.jp/itadev/login.php

## Why this works

1. **ACE-Opencpop has low intra-phoneme variance** → det_head's mean prediction is sharp
2. **IPA phonemes transfer cross-language** (cos 0.88) → existing embeddings mostly work
3. **RVC handles perceptual quality** → model only needs phonemic accuracy, not expressiveness
4. **User control is more precise than model prediction** → vibrato/energy are better drawn than predicted
5. **Simplicity is a feature** → fewer moving parts = more reliable, faster inference, easier to debug

## Lessons for future projects

1. **Start with the simplest baseline that works, then add complexity ONLY if the baseline
   fails on the ACTUAL use case** — not on harder data you added.
2. **Don't optimize for harder data if the easy data already serves your product** — ACE-Opencpop
   was sufficient for a DAW singing synthesis tool.
3. **User control > model prediction** for creative tools — users WANT to control vibrato
   and dynamics, not have an AI guess them.
4. **Discrete tokens are a double-edged sword** — correct tokens are perfect, wrong tokens
   are worse than blurry regression.
5. **Cross-language phoneme sharing works in HuBERT space** — IPA is the right abstraction
   for multilingual singing synthesis.
6. **Noise scale matters for flow matching** — always match x_0 distribution to target distribution.
7. **Every experiment that "fails" teaches something** — the Phase 3→5 arc proved fundamental
   limitations of each approach that would have been unknowable without trying.
