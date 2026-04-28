# Score2HuBERT Documentation Index

Score2HuBERT is the neural SVS frontend for Utai: musical score → HuBERT[768] + F0 → RVC → audio, replacing DiffSinger in the DAW workflow.

The project has two distinct phases, separated by the 2026-04-11 architecture reset:

---

## 📘 Phase 2 — v2 era (2026-04-11 onwards, current)

Clean rewrite. Minimal Conformer encoder + det_head. Focus on IPA phonemes, deterministic regression, cross-lingual sharing.

### Session log (chronological)

- [**SESSION_2026_04_11.md**](./SESSION_2026_04_11.md) — Run 1 baseline (ACE only, val 0.8944)
- [**SESSION_2026_04_12.md**](./SESSION_2026_04_12.md) — Run 2 (+GTSinger), Run 3 (+UTAU/Kiritan/Itako)
- [**SESSION_2026_04_13.md**](./SESSION_2026_04_13.md) — Data audit + GTSinger drop + 4 new JA datasets (OFUTON/Natsume/ONIKU/JVS-MuSiC)
- [**SESSION_2026_04_14.md**](./SESSION_2026_04_14.md) — Run 4 (pos+dur+filter) + Run 5 (JVS drop, Kaiming init)
- [**SESSION_2026_04_14_evening.md**](./SESSION_2026_04_14_evening.md) — Run 6+7 (Kiritan-only) + phoneme contamination audit + HuBERT-is-cluster insight
- [**SESSION_2026_04_14-15_run8.md**](./SESSION_2026_04_14-15_run8.md) — Run 8 full design stack + SOFA + augmentation + launch
- [**SESSION_2026_04_15_run8_midtraining.md**](./SESSION_2026_04_15_run8_midtraining.md) — Mid-training observations @ step 30K; CPU listening eval; residual issues; dual-path deployment design; WORLD F0 rescue plan; EN data drop

### Reference

- [**ARCHITECTURE.md**](./ARCHITECTURE.md) — v2 architecture spec (ScoreEncoder + LR + F0Predictor + DetHubertHead)
- [**DATA_CREDITS.md**](./DATA_CREDITS.md) — attribution for ONIKU/Kiritan/Itako/OFUTON/Natsume/JVS-MuSiC/ACE (**critical for non-commercial usage**)

### Audit artifacts

- `phoneme_contamination_audit.json` — pre-SOFA baseline (2026-04-14)
- `phoneme_contamination_audit_post_sofa.json` — after SOFA realign (2026-04-15)
- `phoneme_contamination_audit_post_aug.json` — after pyworld aug (2026-04-15)
- `audit_post_sofa.log`, `audit_post_aug.log` — pretty-printed audit reports
- `dataset_audit_report.json` — 5-step 2026-04-13 dataset audit

### Runs at a glance (v2)

| Run | Date | Data | Arch | Steps | Best Val | Key lesson / outcome |
|-----|------|------|-----|-------|----------|---|
| 1 | 04-11 | ACE only | 54M | 50K | 0.8944 | baseline |
| 2 | 04-11/12 | +GT en/ja | 54M | 75K | 0.8684 | ZH fine, EN/JA bad |
| 3 | 04-12/13 | +UTAU/Kiritan/Itako | 54M | 75K | 0.7416 | GT confirmed problematic → drop |
| 4 | 04-14 | 8 datasets, pos+filter | 54M | 75K | 0.2330 | per-sample metric bug fixed |
| **5** | **04-14** | **7 JA (JVS val-only), Kaiming LR** | **54M** | **75K** | **0.2366** | **per-dataset table; audio still messy** |
| 6 | 04-14 | Kiritan only | 54M | 7.5K | 0.3860 | single-clean not a silver bullet |
| 7 | 04-14 | Kiritan only tiny | 7.7M | 5.5K | 0.3781 | **tiny model can't hit clusters** |
| **8** | **04-15 →** | **SOFA+aug, 5 JA + ACE** | **54.6M** | **75K** | **_running @ 30K/40%_** | **9-mech stack. Step 25K listening: rest artifacts + minor articulation + JA F0 drift remain. Other Run 5 issues resolved. Expected launch-ready by step 75K if SP=0 converges.** |

---

## 📗 Phase 1 — v1 era (pre-2026-04-11, archived)

Experimental phases A/B/C exploring flow matching, tech embedding, discrete tokens. Accumulated learnings but ceiling issues led to the v2 reset.

All v1 docs archived at [`archive/docs_v1_reference/`](../archive/docs_v1_reference/) — see [INDEX.md](../archive/docs_v1_reference/INDEX.md) there for navigation.

Key findings that carried into v2:
- **IPA unifies cross-lingual phonemes** (cos 0.876 cross-lang) — kept
- **HuBERT regression > flow matching** (flow produces "generic voice") — kept
- **EMA pathological for fresh-head regimes** — kept (disabled in v2)
- **Discrete K-means retrospectively looked viable but was contaminated by GTSinger** — informed 2026-04-14 K-means revisit plan; ultimately dropped for Run 8 after SOFA unified alignment

### Notable v1 artifacts

- v1 project root: `D:/MyDev/MSSTRVC/score2hubert/` (legacy code + weights still present)
- v1 test outputs: `score2hubert/test_outputs_phase3_final/`, `test_outputs_phase4_hybrid/`, `test_outputs_phase4_v2_det/`, `test_outputs_archive/`

---

## When to read what

- **Just joined / resuming after a break?** Start with `memory/project_score2hubert.md` (project_score2hubert in memory index) — current state + recent runs.
- **Need to understand Run 8 design?** `memory/project_score2hubert_run8_design.md` + latest session log.
- **Debugging a preprocessing / training gotcha?** `memory/project_score2hubert_pipeline.md`.
- **Why was GTSinger dropped / JVS excluded / UTAU excluded?** `memory/project_score2hubert_gtsinger_dropped.md` + `_contamination_theory.md`.
- **Contemplating architectural change?** Check `_cluster_insight.md` + `_decisions.md` first.
- **v1 legacy questions?** `archive/docs_v1_reference/INDEX.md`.
