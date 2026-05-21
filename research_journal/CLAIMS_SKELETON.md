# 5 main claims of the paper (with supporting evidence)

Each claim is stated precisely, with supporting experiments listed, and the
minimum supporting numbers. Used to (a) anchor the writing, (b) audit consistency
between sections, (c) identify weakest claim that might need more experiments.

---

## Claim 1 — Holistic-face illusion sensitivity in visual priors is paradigm-specific, not unified

**Statement**: Different holistic-face illusion paradigms are captured by different
classes of modern visual priors. No single model class dominates all three.

**Evidence**:
- E002 + E005: image-text contrastive models (CLIP, SigLIP, MetaCLIP) ISI 3.5-6.8 on Thatcher
- E023: same models give CSI 1.10-1.21 (pixel-corrected) on composite-face — weaker
- E024: DINOv2 family lowest PWI 0.31-0.40 on part-whole — strongest spatial-holistic effect
- E023 + E024: DINOv2 family wins composite + part-whole but lower on Thatcher

**Strength**: STRONG. Three independent paradigms × consistent pattern across model classes.

**Counter-claims considered**:
- Could the dissociation be a metric artifact? No — pixel baseline = 1.000 on Thatcher rules out trivial reasons; pixel-corrected values on composite/part-whole preserve qualitative ordering.

---

## Claim 2 — Face-identification-trained models show no holistic illusion effect on any paradigm

**Statement**: FaceNet (VGGFace2-trained and CASIA-Webface-trained Inception-Resnet-V1)
shows ISI/CSI/PWI essentially equal to baseline (within ±0.1 of 1.0) on all
three paradigms despite having explicit face-identity supervision.

**Evidence**:
- E004 + E023 + E024:
  - FaceNet Thatcher ISI 1.026 - 1.117
  - FaceNet composite CSI 1.111 - 1.154 (raw) ≈ baseline-equivalent (corrected)
  - FaceNet part-whole PWI 0.711 - 0.953
- d_up and d_inv values show FaceNet IS responsive to perturbations (large d~0.4
  on Thatcher upright), but the orientation × perturbation interaction is absent.

**Strength**: STRONG. Two FaceNet variants on three paradigms, consistent baseline result.

**Interpretation**: pose/orientation invariance objective of face-identification
training actively suppresses orientation-dependent configural features.

**Counter-claims considered**:
- Could it be that FaceNet doesn't see faces well enough? No — d_up ≈ 0.38 confirms
  FaceNet DOES respond strongly to feature perturbations.

---

## Claim 3 — The face-feature-specific Thatcher signal in image-text contrastive priors is ~70% face-feature-localized

**Statement**: The Thatcher ISI in CLIP-class priors (4-7) is dominated (60-80%)
by face-feature-region perturbation specifically, not by general upright-orientation
bias. Random-location bbox controls reduce CLIP ISI by 59-74%.

**Evidence**:
- E003: 200 FFHQ identities with random non-feature bboxes (same Thatcherize algorithm)
  - CLIP-bigG14: 6.763 → 1.972 (-71%)
  - All CLIP variants: 59-74% reduction
  - DINOv2-large/giant: PWI even below 1 (0.85-0.89) on random-bbox

**Strength**: MODERATE-STRONG. Single discriminating experiment. Could be reinforced by
gaze-tracking or human Bubbles psychophysics in future work.

**Counter-claims considered**:
- Could it be measurement artifact? No — same algorithm, same identities, only bbox
  location varies. Pixel-baseline still = 1.000.

---

## Claim 4 — EEG bottleneck imposes uniform per-CLIP-dim attenuation, not face-specific destruction

**Statement**: ATM-S decoded EEG embeddings preserve CLIP-H/14 target space with
mean per-dim Pearson r ≈ 0.158 across 10 THINGS-EEG2 subjects, with NO
dimension-specific structure correlating with Thatcher loading.

**Evidence**:
- E020: Spearman ρ(preservation, Thatcher loading) = -0.029, permutation p = 0.35
- E020: High-loading-quartile preservation 0.154 vs low-loading-quartile 0.165 (~identical)
- E021: Face-related test concepts preservation 0.150 vs random-subset 0.145, perm p = 0.37

**Strength**: STRONG. Two independent NULL tests (per-dim and per-category).

**Interpretation**: Current EEG visual decoding is non-specific low-pass. Predicted
EEG-side ISI on Thatcher: ~1.5-2 (collapse from 5.5 image-side).

**Counter-claims considered**:
- Could ATM be uniquely bad? Other EEG decoders not testable yet (no pre-computed
  embeddings on HF for AVDE/ENIGMA). Limitation, not refutation.
- Could the result reflect THINGS-EEG2 train data limitations? Possibly, but face-category
  subgroup analysis already rules out a major training-data face-vs-object split.

---

## Claim 5 — Current EEG-to-image decoders cannot inherit any holistic-face illusion effect

**Statement (composite)**: Given (a) only image-text contrastive priors carry the
Thatcher signature (Claim 1+2), (b) the EEG bottleneck is uniform low-pass with
r=0.158 per dim (Claim 4), and (c) no current EEG decoder uses multiple
diverse-paradigm visual anchors, current EEG decoders cannot reproduce human-aligned
illusion sensitivity on any of the three paradigms.

**Evidence**: Combination of Claims 1+4 plus the structural observation that
ATM/AVDE/ENIGMA/HVF/ViEEG all anchor exclusively to image-text contrastive priors
(documented in literature review).

**Strength**: STRONG combination claim. The individual building blocks are strong;
the combination is logically valid.

**Counter-claims considered**:
- "EEG decoder + non-CLIP anchor would work" — direct Route A analysis on AVDE / ENIGMA
  would either support or refute. Not yet possible.
- "EEG signal carries info CLIP doesn't" — possible but unsupported. The per-dim
  analysis shows existing decoders don't recover even what CLIP has.

---

## Audit: which claim is weakest?

**Claim 3** is the weakest because it rests on a single discriminating experiment
(random-bbox control). Strengthening options: human gaze-tracking on Thatcher stimuli,
Bubbles-style classification-image psychophysics. None are in scope for this paper.
We can hedge in the discussion: "primarily face-feature-localized, with a small
residual general orientation component (random-bbox CLIP ISI 1.3-2.0, well above 1)."

**Claim 4** has the highest counter-claim risk: limited to ATM. Worth attempting
a Route A on another EEG decoder before final submission if pre-computed embeddings
become available.
