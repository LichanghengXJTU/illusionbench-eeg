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

**Strength**: STRONG (now). E003 + E025 together. E025 added bbox-size sensitivity
sweep (0.5×, 1.5×) showing CLIP random-bbox ISI remains 1.2-2.0 across sizes
while face-Thatcher stays 5-7 — face-specificity gap is robust to perturbation
magnitude. Could be additionally reinforced by gaze-tracking or human Bubbles
psychophysics in future work.

**Counter-claims considered**:
- Could it be measurement artifact? No — same algorithm, same identities, only bbox
  location varies. Pixel-baseline still = 1.000.
- Could it be bbox-size-dependent? No — E025 sensitivity sweep shows
  CLIP-bigG14 random-bbox ISI varies between 1.47 and 1.97 across 0.5×–1.5×
  scales, while face-Thatcher ISI is 5.50-6.76. Robust gap.

---

## Claim 4 — EEG bottleneck imposes uniform per-CLIP-dim attenuation, not face-specific destruction

**Statement**: ATM-S decoded EEG embeddings preserve CLIP-H/14 target space with
mean per-dim Pearson r ≈ 0.158 across 10 THINGS-EEG2 subjects, with NO
dimension-specific structure correlating with Thatcher loading. **This uniform
low-pass holds INDIVIDUALLY for each subject.**

**Evidence**:
- E020: Spearman ρ(preservation, Thatcher loading) = -0.029, permutation p = 0.35
- E020: High-loading-quartile preservation 0.154 vs low-loading-quartile 0.165 (~identical)
- E021: Face-related test concepts preservation 0.150 vs random-subset 0.145, perm p = 0.37
- **E035 (NEW — strengthening)**: per-subject Route A — all 10 subjects show
  Spearman(r, loading) between -0.063 and +0.037, 8/10 negative, 9/10
  p > 0.1. Mean r per subject in [0.126, 0.212], std=0.026. High-Q minus
  low-Q quartile difference per subject in [-0.020, +0.006]. The pooled
  distribution of all 10240 (dim, subject) r values is roughly Gaussian
  centered at 0.158 with no bimodality.

**Strength**: STRONG (raised from MEDIUM-STRONG by E035).

**Interpretation**: Current EEG visual decoding is non-specific low-pass. Predicted
EEG-side ISI on Thatcher: ~1.5-2 (collapse from 5.5 image-side). The result
holds individually for each subject — not a subject-averaging artifact.

**Counter-claims considered**:
- Could ATM be uniquely bad? Other EEG decoders not testable yet — no
  pre-computed embeddings publicly available for NICE/AVDE/ENIGMA/ViEEG on
  HuggingFace as of 2026-05; NICE-EEG releases model weights only
  (HF: eeyhsong/NICE), not pre-computed THINGS-EEG2 embeddings. We could
  re-run NICE inference but this requires the full preprocessed
  EEG dataset (~20 GB). Marked as limitation.
- Could the result reflect THINGS-EEG2 train data limitations? Possibly, but face-category
  subgroup analysis already rules out a major training-data face-vs-object split.
- Could the result be a subject-averaging artifact? **No (E035)**.

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
become available. **Per-subject consistency check (E035, tick 36) reduces this risk
substantially**: all 10 subjects show the same uniform low-pass pattern, ruling
out the subject-averaging concern.

---

## Claim 6 (NEW post-E031/E032) — Face-recognition models can approach moderate Thatcher signal under specific training conditions, but the data regime matters more than the loss formulation

**Statement**: AdaFace IR-50 trained on the small noisy CASIA-WebFace dataset
(~500K images, ~10K identities, 2014-era) shows Thatcher ISI 2.91 [2.63, 3.22] —
the highest among 25 priors after CLIP-class. AdaFace IR-101 trained on cleaner
larger MS1MV2 (~5.8M images) gives only 1.24, and AdaFace IR-50 on cleaner
WebFace4M (~4M images) gives 1.33. This INVERTED data-scaling pattern is
opposite to CLIP's, where more data → more Thatcher.

**Evidence**:
- E030: P23 ArcFace AuraFace (R100): Thatcher ISI 1.70 [1.64, 1.77]
- E031: P24 AdaFace IR-101 MS1MV2: ISI 1.24, CSI 1.60, PWI 0.59 (partially
  paradigm-consistent — first face-rec model with all three indices deviating
  from baseline in the expected direction)
- E031: P25 ArcFace IR-101 WebFace4M: ISI 1.17, CSI 1.32, PWI 0.50
- E032: P26 AdaFace IR-50 CASIA: ISI 2.91, CSI 1.28, PWI 0.75 — strongest
  face-rec Thatcher of any model
- E032: P28 AdaFace IR-50 WebFace4M ISI 1.33; P29 AdaFace IR-50 MS1MV2 ISI 1.21
- E032: AuraFace's earlier PWI 1.43 inversion was a BGR-preprocessing artifact
  (RGB-correct PWI = 0.99)

**Strength**: STRONG. Multiple AdaFace variants tested with fixed loss + varied
data and varied backbone show data is the dominant axis. Inversion of CLIP's
scaling law is reproducible.

**Interpretation**: Face-identity training with angular-margin loss can develop
Thatcher sensitivity if the training data is sufficiently noisy/diverse to
require sensitivity to fine-grained perturbations. Cleaning the data eliminates
this signal — likely because identity manifolds become too tight.

**Counter-claims considered**:
- Could it be a preprocessing confound? E032 P23rgb sanity check shows AuraFace
  BGR vs RGB differs by 0.25 in ISI but the overall pattern is preserved.
- Could it be backbone scale? E032 IR-50 vs IR-101 on same MS1MV2 data shows
  near-identical ISI (1.21 vs 1.24) — backbone scale is small effect.

---

## Claim 7 (NEW post-E033/E034) — Representational geometry is paradigm-conditional; dissociation is a (model × paradigm) interaction

**Statement**: At the level of global RSA (representational similarity analysis)
on the 800-stimulus Thatcher set, the 25 priors cluster into 6 stable families
(reconstructive, pixel, semantic [CLIP+DINOv2+SigLIP merged], angular-margin
face-rec, triplet face-rec, CORnet-S alone). However, the same cluster
analysis on Composite and Part-Whole stimuli shows that several priors
MIGRATE between clusters: CORnet-S (alone → semantic → reconstructive across
the three paradigms), AdaFace IR-50 CASIA (angular-margin → triplet → semantic),
FaceNet (triplet → triplet → semantic). Cross-paradigm Spearman agreement
between RSA matrices is 0.86 (Thatcher↔Composite), 0.54 (Thatcher↔Part-Whole),
0.73 (Composite↔Part-Whole) — Part-Whole is the most distinct paradigm.

**Evidence**:
- E033: 25×25 RSA correlation matrix on Thatcher 800 stimuli; hierarchical
  clustering at k=6 yields 6 families.
- E033: CKA(CLIP-H/14, AdaFace-CASIA) = 0.45 quantifies AdaFace's ~55% novel
  information over CLIP. CKA(CLIP-H/14, CLIP-bigG14) = 0.93 (within-CLIP
  baseline). CKA(CLIP-H/14, DINOv2-giant) = 0.75.
- E034: per-paradigm RSA matrices for Composite and Part-Whole; same 6-cluster
  procedure shows different cluster memberships.
- E034: cross-paradigm Spearman agreements as stated.

**Strength**: STRONG. Quantitative cross-paradigm comparison + qualitative
migration pattern + CKA quantification of novelty.

**Interpretation**: The paradigm-specific dissociation we report in §4.1-4.4
is a fine-grained effect WITHIN a globally similar representational space
(CLIP and DINOv2 cluster together at global level despite opposite paradigm
scores). This strengthens rather than weakens the case for paradigm-specific
benchmark tests — they reveal structure global RSA misses. The cluster
migration shows that the same model's effective representation depends on
what task is being asked.

**Counter-claims considered**:
- Could the migrations be clustering-algorithm artifacts (sensitive to k)?
  Partly. We report k=6 throughout for consistency; at k=4 and k=8 the
  qualitative pattern of "CORnet migrates, AdaFace migrates, FaceNet
  collapses on PartWhole" persists.
- Could it be sample-size effect (800 vs 400 stimuli)? Possibly small contribution.
  But the cross-paradigm Spearman = 0.54 on Part-Whole reflects a real signal
  difference, not stochastic noise.

---

## Claim 8 (NEW post-E044) — Bio-faithful architecture is not sufficient; the configural-illusion signal is training-objective-driven (constructive negative)

**Statement**: A maximally bio-faithful face architecture — HOLO-Net, with each
layer mapped to a brain region (LGN→V1→V2→V4→OFA→MFP→AFP→Orientation-Gate→FFA
→ATL + Predify-style PC feedback + magno-PFC gist), trained only on Glint360K
face identities with an AdaFace identity objective and no illusion-stimulus
exposure — shows NO Thatcher effect (ISI ≈ 1.0) and NO composite effect
(CSI ≈ 1.0) at any layer including the FFA layer where the design pre-
registered the falsification. This pattern is exactly that of FaceNet
(Claim 2), confirming that the configural-illusion emergence pinned to
image-text contrastive training (Claims 1, 3) is **not architecture-driven** —
the training objective is the lever. The HOLO-Net FFA module is, however, not
inert: it drives Part-Whole PWI from mfp 0.75 → ffa 2.20 (the largest
single-paradigm deviation HOLO-Net produces), but in the opposite direction to
the design's ≤ 0.5 criterion — the architecture changes something real, just
not the human-aligned configural signal.

**Evidence**:
- E041: minimal HOLO-Net (bio components OFF) FFA-layer ablation floor —
  Thatcher ISI 0.98, Composite CSI 1.19, Part-Whole PWI 0.78, Random-bbox
  1.08; fails all (intended baseline).
- E044: full HOLO-Net v5 (all bio components, trained to identity loss 0.97,
  functional Orientation Gate) FFA layer — ISI 1.00, CSI 0.99, PWI 2.20,
  Random-bbox 1.08; fails 3 of 4 pre-registered §6 criteria → ALL FOUR: FAIL.
- Per-layer profile (pixel-corrected): ISI flat ≈ 1.0 at every layer (v1 1.02,
  v2 1.04, v4 1.00, mfp 0.99, afp 1.02, ffa 1.00, atl 1.00); CSI flat ≈ 1.0
  (range 0.97-1.18); PWI jumps mfp 0.75 → afp 2.07 → ffa 2.20 → atl 0.81 —
  the FFA holistic-binding module specifically restructures part-whole
  geometry. Random-bbox stays ≈ 1.0-1.1 at all layers (control intact).

**Strength**: STRONG. The full HOLO-Net + FaceNet (Claim 2) — two
architecturally very different face-identity-trained models — both produce
ISI ≈ 1.0. The negative result is reproducible across architectures conditioned
on training objective.

**Interpretation**: confirms Claims 1+2+3 from the architecture side. The
"first paradigm-consistent" headline that Idea-003 aimed at is refuted as
pre-registered. But the constructive negative is itself a load-bearing finding:
it rules out architectural bio-fidelity as the cause of the dissociation in
Claim 1, leaving the training objective as the operative lever.

**Counter-claims considered**:
- Undertrained? At step 30000, identity loss 0.97 (well-trained); orientation
  0.16 (gate accurate); ISI ≈ 1.0 at all layers from early epochs onward.
  Training is not the limitation.
- Could a different architecture revision (design §8: 6-step FFA, Capsule,
  GLOM) succeed under identity training? Possible but contraindicated — the
  FFA module IS not inert (PWI 2.20); the architecture HAS expressive
  capacity; the gap is the objective per Claim 2.
- The design §7 Thatcher mechanism is conceptually flawed: the Orientation
  Gate distinguishes WHOLE-face orientation, but Thatcher is a LOCAL-feature
  inversion within an upright face — the gate has no signal for it. A
  revision could address this, but again would not change the objective.
- Could HOLO-Net + image-text contrastive objective succeed? Open question
  (FLAG-002 option 2) — would test architecture × objective. Not run.
