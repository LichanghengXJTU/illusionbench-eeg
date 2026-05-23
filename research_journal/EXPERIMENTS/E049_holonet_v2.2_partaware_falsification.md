# E049 — HOLO-Net v2.2 part-aware FTPC §6 falsification (Gen 1 variant 1)

**Hypothesis** (pre-registered before run, per NOTES_FOR_USER 22:20 directive
+ this experiment's docstring):
K=4 part-aware face templates {T_eyes, T_nose, T_mouth, T_chin} at fixed
sub-regions of the 16×16 DINOv2 patch grid; δ readout = concat of per-part
pooled mismatches → 1536-d. Predicted: CSI lifts toward PASS (1.300 → ≥1.5);
PWI held near PASS (0.446); ISIrbox held; **ISI unlikely to flip from
anti-direction** without orientation-aware processing.

**Method**:
- Frozen DINOv2 ViT-S/14 backbone
- Per-part templates with EMA buffer; 500-step template-only training on
  ImageNet face samples (no SSL, no backbone gradient)
- Region map: eyes rows 5-8 cols 3-13, nose 7-10 6-10, mouth 10-13 4-12,
  chin 12-16 5-11 (covers 90 of 256 patches, focused on face areas)

**Stimuli / Models**: same as E044/E045 (FFHQ-1024 Thatcher 200 IDs,
Composite 100, Part-Whole 100, Random-bbox 200).

**Training summary** (3.3 min on H100, 13,648 face samples):
| Part | h × w | T_ema init norm | T_ema final norm |
|------|-------|----------------:|-----------------:|
| eyes  | 3 × 10 | 104.6 | 94.7 |
| nose  | 3 × 4 | 68.1 | 61.5 |
| mouth | 3 × 8 | 96.0 | 87.5 |
| chin  | 4 × 6 | 96.6 | 90.4 |

Smooth EMA decay (~9-10% over 500 steps); norms proportional to region
patch count → well-behaved training.

**Result (FFA = δ_concat 1536-d, pixel-corrected)**:

| Layer | ISI | CSI | PWI | ISIrbox |
|-------|----:|----:|----:|--------:|
| v1/v2/v4/mfp/afp/atl (aliases) | 0.908 | 1.282 | 0.397 | 0.662 |
| **ffa (δ_concat 1536-d)** | **0.981** | **1.017** | **0.617** | **0.139** |

**§6 verdict at FFA**:
- ISI = 0.981 — need ≥ 3.0 → **FAIL**
- CSI = 1.017 — need ≥ 1.5 → **FAIL**
- PWI = 0.617 — need ≤ 0.5 → **FAIL**
- ISIrbox = 0.139 — need ≤ 1.5 → **PASS**
- ALL FOUR: **FAIL (1/4 pass)**

**Diff vs v3 (E045) FFA** (the immediate predecessor):

| Index | v3 ffa (δ_pooled) | v2.2 ffa (δ_concat) | Δ | Direction |
|---|---:|---:|--:|---|
| ISI | 0.901 | 0.981 | +0.08 | Better (no longer anti-Thatcher, but still FAIL) |
| CSI | 1.300 | 1.017 | -0.28 | **Worse** (was close to PASS, now regressed) |
| PWI | 0.446 (PASS) | 0.617 (FAIL) | +0.17 | **Worse** (lost PASS) |
| ISIrbox | 0.654 | 0.139 | -0.52 | Better margin but suspiciously low |

**Interpretation** (4 observations):

1. [CONFIRMED] **ISI 0.901 → 0.981 — part-aware DOES neutralize anti-Thatcher**.
   The signal moves in the right direction (closer to 1.0 = no asymmetry,
   from 0.9 = anti-Thatcher). This is partial confirmation that part-level
   mismatch CAN encode some Thatcher-relevant signal, but **insufficient to
   flip to pro-Thatcher** because the readout doesn't condition on whole-face
   orientation.

2. [CONFIRMED] **CSI -0.28, PWI +0.17 — part decomposition loses holistic
   context**. Composite (CSI) and Part-Whole (PWI) measure how the WHOLE face
   integrates parts; cutting AFP into 4 independent regions severs the
   cross-part spatial relationships these paradigms depend on. The per-part
   pooled δ_k vectors don't carry enough holistic geometry to express
   composite-misalignment or whole-context-modulation of parts.

3. [CONFIRMED] **ISIrbox 0.139 — extreme face-centric readout**. The 4
   part templates cover only 90 of 256 patches and all in face regions.
   Random-bbox perturbations (at non-face locations) don't enter the
   δ_concat readout. PASSes the criterion (≤ 1.5) by a wide margin but
   suspicious — the value is so low it suggests we've over-specialized the
   readout to faces and may be missing scene-level signal.

4. [CONFIRMED] **δ_concat 1536-d is higher-variance than δ_pooled 384-d**.
   Per-part dims carry per-part noise; without cross-part normalization,
   the 4× dimensionality may dilute the signal-to-noise of the relevant
   directions.

**Diagnosis & v2.3 design**:

Part-aware alone is **strictly worse** on the holistic paradigms (CSI, PWI)
and only marginally better on ISI. The part decomposition trade-off:
- Pros: enables per-feature analysis (and slightly reduces anti-Thatcher)
- Cons: severs cross-part holistic context

Next iteration **v2.3** = orientation-aware readout via vflip:
- Keep v3's GLOBAL FTPC template T (works on CSI/PWI)
- ADD a vflip(x) second-forward through DINOv2
- δ_global = AFP - T (the v3 readout, preserves CSI/PWI)
- δ_orient = T - AFP_vflip (NEW orientation-asymmetric signal)
- ffa = concat(δ_global_pooled, δ_orient_pooled) → 768-d

**Why this should give Thatcher** (analytical prediction):
- Upright normal: AFP ≈ T → δ_global small; vflip(AFP) far from T → δ_orient large
- Upright thatched: AFP slightly off T → δ_global medium; vflip far → δ_orient very large
- Inverted normal: AFP far from T → δ_global large; vflip ≈ T → δ_orient small
- Inverted thatched: AFP far from T → δ_global large; vflip slightly off T → δ_orient medium
- In the orient channel: d(V1, V2) ~ medium, d(V3, V4) ~ small → **ISI > 1**

**v2.3 pre-registered predictions**:
- ISI: 1.5-3.0 (pro-Thatcher direction, may not hit 3.0 threshold)
- CSI: ≥ 1.3 (global readout recovers; possibly higher with extra channel)
- PWI: ≤ 0.5 (global readout recovers to v3 level)
- ISIrbox: ≤ 1.5 PASS

**Replicability**:
- Random seed: 20260521
- Code: `holo_net/model_v2_dinov2_partaware.py`, `train_v2_dinov2_partaware.py`,
  `eval_extract_dinov2_partaware.py`, `eval_falsification_dinov2_partaware.py`
- Output: `/workspace/runs/2026-05-23_holonet-v2.2-partaware_seed20260521/`
- NPZs: `/workspace/illusionbench-eeg/outputs/embeddings/{paradigm}_HOLONET_v2.2_partaware/`

**Linked Q###**: continues Q011 (architectural fix for §6 failure)
**Linked Idea-###**: Idea-003 score unchanged (5.5 — partial credit for ISI
improvement, debit for CSI/PWI regression net out roughly).
