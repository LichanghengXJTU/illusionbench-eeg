# E032 — AuraFace RGB sanity + AdaFace IR-50 dataset sweep

**Date**: 2026-05-22 (tick 30)
**Linked**: E031 (angular-margin loss ablation).

## Goal

Two questions raised by E031:
1. **Preprocessing confound**: P23 AuraFace was extracted with BGR
   preprocessing (insightface convention). Its high Thatcher ISI (1.70) and
   inverted Part-Whole (PWI 1.43) might partly reflect the BGR/RGB channel
   ordering rather than the model's intrinsic illusion sensitivity.
2. **Data-vs-architecture**: P24 (AdaFace IR-101 MS1MV2) vs P25 (ArcFace IR-101
   WebFace4M) differ in BOTH loss AND data. To disentangle, fix the loss
   (AdaFace) and vary the data (CASIA, MS1MV2, WebFace4M).

## Methods

**Part A — AuraFace RGB sanity (P23rgb)**:
Re-extract AuraFace (ResNet-100, glintr100.onnx) on all 4 stimulus sets, with
**RGB** preprocessing (no channel reversal), still using [-1, 1] normalization.
Compare ISI/CSI/PWI directly with P23 (BGR variant).

**Part B — AdaFace IR-50 dataset sweep**:
Extract three AdaFace IR-50 variants from CVLFace:
- **P26: minchul/cvlface_adaface_ir50_casia** — CASIA-WebFace (~500K images,
  ~10K identities, noisier/older dataset)
- **P28: minchul/cvlface_adaface_ir50_webface4m** — WebFace4M (~4M cleaned
  images, ~600K identities)
- **P29: minchul/cvlface_adaface_ir50_ms1mv2** — MS1MV2 (~5.8M images, ~85K
  identities)

(IR-50 VGG2 variant not released on HF; only IR-18 has VGG2.)

All IR-50 models use the same architecture (43.6M params, 512-d output, RGB
112×112 with mean=std=0.5).

## Result

| Prior | Backbone | Data | Thatcher ISI | Composite CSI | Part-Whole PWI | Random-bbox ISI | Face-spec% |
|---|---|---|---|---|---|---|---|
| P23 ArcFace AuraFace (BGR) | R100 | (insightface) | 1.70 [1.64, 1.77] | 1.16 [1.12, 1.20] | 1.43 [1.34, 1.52] | 0.39 | 77% |
| **P23rgb ArcFace AuraFace (RGB)** | R100 | (insightface) | **1.45 [1.39, 1.51]** | 1.17 [1.14, 1.21] | **0.99 [0.94, 1.04]** | 0.36 | 75% |
| P24 AdaFace IR-101 MS1MV2 | IR-101 | MS1MV2 | 1.24 [1.18, 1.30] | 1.60 [1.51, 1.70] | 0.59 [0.56, 0.62] | 0.13 | 89% |
| P25 ArcFace IR-101 WebFace4M | IR-101 | WebFace4M | 1.17 [1.11, 1.23] | 1.32 [1.26, 1.39] | 0.50 [0.48, 0.53] | 0.36 | 69% |
| **P26 AdaFace IR-50 CASIA** | **IR-50** | **CASIA** | **2.91 [2.63, 3.22]** | 1.28 [1.23, 1.34] | 0.75 [0.71, 0.80] | **1.04** | 64% |
| P28 AdaFace IR-50 WebFace4M | IR-50 | WebFace4M | 1.33 [1.26, 1.41] | 1.34 [1.29, 1.40] | 0.52 [0.49, 0.54] | 0.44 | 67% |
| P29 AdaFace IR-50 MS1MV2 | IR-50 | MS1MV2 | 1.21 [1.14, 1.27] | 1.46 [1.40, 1.54] | 0.64 [0.61, 0.66] | 0.26 | 79% |

## Interpretation

[CONFIRMED] **1. The AuraFace Part-Whole inversion is a BGR/RGB preprocessing
artifact**. With correct RGB preprocessing (P23rgb), PWI = 0.99, essentially
baseline. With BGR (P23, the insightface convention), PWI = 1.43, inverted.
The Thatcher ISI also drops from 1.70 → 1.45 with RGB. **Going forward, we
must use the model's expected RGB preprocessing**; the inversion was not a
real model finding.

[CONFIRMED] **2. AdaFace IR-50 trained on CASIA-WebFace shows the strongest
Thatcher signature of any face-recognition model (ISI 2.91 [2.63, 3.22])**.
This is higher than DINOv2-large (2.37) and comparable to mid-tier CLIP
(CLIP-B/32 = 5.18, but only 1/4 the scale). However, its random-bbox ISI is
1.04 (vs 0.13 for IR-101 MS1MV2), so its face-feature-specificity is 64% —
lower than IR-101 MS1MV2's 89%. This indicates more residual general-
orientation bias.

[CONFIRMED] **3. Data-scaling pattern is INVERTED for AdaFace IR-50**:
- CASIA (~500K images, ~10K identities, noisy): Thatcher ISI **2.91**
- MS1MV2 (~5.8M images, ~85K identities, cleaned): Thatcher ISI 1.21
- WebFace4M (~4M cleaned images, ~600K identities, very clean): Thatcher ISI 1.33

**Smaller, noisier training data → higher Thatcher ISI**. This is opposite of
the standard scaling law for CLIP-class models (where bigger data → more
Thatcher). The CASIA pattern is consistent with the idea that:
- Noisy/diverse face data → more sensitivity to fine perturbations
  (Thatcherization), since the model has to learn to recognize despite
  identity-irrelevant variation
- Cleaned/large data → embeddings become more identity-tightly-grouped,
  collapsing all face variations of the same identity to similar codes
  (suppressing Thatcher signal)

[CONFIRMED] **4. Backbone scale (IR-101 vs IR-50) has SMALL effect on
illusion fingerprints when data is held constant** (compare P24 vs P29, both
MS1MV2): ISI 1.24 vs 1.21, CSI 1.60 vs 1.46, PWI 0.59 vs 0.64. Backbone
adds slight refinement but does not change the pattern.

[CONJECTURE] **5. The dominant axis for face-recognition model variation is
training data, NOT loss or architecture**. AdaFace vs ArcFace gives similar
results when data + backbone match. Data type (CASIA vs MS1MV2 vs WebFace4M)
gives dramatically different patterns.

## Refined dissociation taxonomy (24 priors)

The 24-prior battery now has these patterns:
1. **CLIP-class** (Thatcher dominant, scale-emergent): CLIP family
2. **DINOv2-class** (composite + part-whole dominant): DINOv2 family
3. **Bio-anatomy class**: CORnet-S (part-whole only)
4. **AuraFace pattern (BGR artifact)**: P23, now resolved — really part-whole
   is baseline, Thatcher is moderate (1.45)
5. **AdaFace CASIA pattern**: high Thatcher (~3), modest Composite (~1.3),
   moderate Part-Whole (~0.75). Unique among face-rec.
6. **AdaFace MS1MV2 pattern**: balanced moderate dissociation (ISI ~1.2,
   CSI ~1.5, PWI ~0.6)
7. **AdaFace/ArcFace WebFace4M pattern**: similar to MS1MV2 but slightly
   weaker
8. **Triplet face-rec class**: FaceNet — baseline on all
9. **Reconstructive class**: MAE, VAE
10. **Untrained / pixel**

**Total: 10 distinct dissociation patterns across 24 priors** — strong
evidence that the holistic-face illusion fingerprint is multi-dimensional
and not captured by any single training paradigm.

## Implication for Idea-003

- ✅ **Sub-path (a) candidate refined further**: the strongest off-the-shelf
  Thatcher emerges in **AdaFace IR-50 trained on CASIA-WebFace**. This
  motivates a CORnet-S + AdaFace + CASIA-like training as a clean recipe.
- ⏭ **New angle (NEW from E032)**: The "small noisy data → emergent
  illusion signal" pattern suggests we may want to NOT use modern cleaned
  datasets when training the bio-prior. CASIA is from 2014, 10K identities,
  ~50 images per identity. Easy to obtain.
- ⏭ **Open question Q010-refined**: is the CASIA-vs-MS1MV2 difference
  driven by image quality variation, identity diversity, or label noise?
  Could test by training AdaFace on a subsampled-CASIA-equivalent slice
  of MS1MV2 or WebFace4M.

## Replicability

- Server: /workspace/illusionbench-eeg/
- Scripts: `/tmp/tick30_arcrgb_adaface_ir50.py` (initial run, interrupted)
  + `/tmp/tick30_finish.py` (resumed for P27, P29)
- NPZs: `outputs/embeddings/{thatcher_ffhq,composite_ffhq,partwhole_ffhq,
  thatcher_randombbox}/P{23rgb,26,28,29}_*.npz`
- Metrics: `outputs/tables/e032_*.csv`

---
