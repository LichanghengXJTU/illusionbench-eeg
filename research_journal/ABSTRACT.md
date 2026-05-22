# Abstract — polished v2 (tick 17), v3 (tick 37)

**Target**: ≤ 200 words for venues that cap abstracts; ≤ 250 words for longer.

---

## v3 (218 words) — current canonical version (tick 37, post-E028-E035)

Current EEG-to-image decoders (ATM, AVDE, ENIGMA, HVF, ViEEG) universally
anchor visual prediction to image-text contrastive embeddings, most commonly
CLIP-ViT-H/14. Whether such pipelines inherit human-aligned holistic face
processing has not been measured. We introduce **IllusionBench-EEG**, a
three-paradigm benchmark — Thatcher illusion, composite-face, part-whole —
constructed at FFHQ-1024 native resolution and tested on **25 visual priors**
spanning 8 training paradigms plus ATM's pre-computed EEG embeddings on all
10 THINGS-EEG2 subjects. **Image-side**: a training-objective × paradigm
dissociation emerges. Image-text contrastive priors dominate Thatcher (ISI
4-7, near-human); DINOv2 dominates composite-face + part-whole spatial
integration (CSI 1.4-1.7, PWI 0.31-0.40); face-identification networks
trained with triplet loss sit at baseline. We additionally identify two
sub-class dissociations: AdaFace IR-50 trained on the small noisy CASIA
dataset reaches ISI 2.91 — opposite of CLIP-style data-scaling — and
bio-inspired CORnet-S is the most spatially-holistic prior (PWI 0.21).
Representational similarity analyses show these dissociations are
paradigm-conditional, with priors migrating between clusters across
paradigms. **EEG-side**: across 10 individual subjects, per-CLIP-dim
preservation is uniformly r ≈ 0.158 with no Thatcher-loading structure
(Spearman ρ = -0.029, every subject |ρ| < 0.07). Current EEG decoders
cannot reproduce holistic face processing on any paradigm; we argue this
calls for multi-prior, dimension-fine alignment.

---

## v2 (190 words) — original — sharper, claim-first

Current EEG-to-image decoders (ATM, AVDE, ENIGMA, HVF, ViEEG) universally
anchor visual prediction to large-scale image-text contrastive embeddings,
most commonly CLIP-ViT-H/14. Whether the resulting pipeline inherits
human-aligned holistic face processing has not been measured. We introduce
**IllusionBench-EEG**, a three-paradigm benchmark — Thatcher illusion,
composite-face, part-whole — built at FFHQ-1024 native resolution with
pixel-baseline-verified metrics, evaluated across 17 visual priors and
ATM's pre-computed EEG embeddings on all 10 THINGS-EEG2 subjects.
**Image-side**: a clean training-objective × paradigm dissociation emerges.
Image-text contrastive models (CLIP, SigLIP, MetaCLIP) dominate Thatcher
(ISI 4-7, near-human); DINOv2 self-supervised models dominate composite and
part-whole spatial binding (CSI 1.4-1.7, PWI 0.31-0.40); face-identification
models (FaceNet) sit at baseline on all three paradigms despite explicit
identity supervision. **EEG-side**: per-CLIP-dim preservation is uniform at
r ≈ 0.158, with no dimension-specific (Spearman ρ = −0.029, p = 0.35) or
face-category-specific (face-subset 0.150 vs random 0.145, p = 0.37) structure.
Current EEG visual decoders cannot reproduce human-aligned holistic face
processing on any paradigm. We argue this calls for multi-prior decoders and
dimension-fine alignment objectives.

---

## v1 (original, 250 words — kept for comparison)

Current EEG-to-image decoders (ATM, AVDE, ENIGMA, HVF, ViEEG) all anchor
their visual prediction to large-scale image-text contrastive vision encoders
— most commonly CLIP-ViT-H/14 LAION. Whether these EEG decoders inherit
human-like configural face processing from their visual priors is unknown.
We introduce IllusionBench-EEG, a three-paradigm benchmark (Thatcher illusion,
composite-face, part-whole) constructed at FFHQ-1024 native resolution with
strictly matched pixel-baseline sanity, evaluating 17 visual priors spanning
six training paradigms. Image-side findings: a clean training-objective ×
paradigm dissociation emerges: image-text contrastive priors (CLIP, SigLIP,
MetaCLIP) dominate the Thatcher illusion (ISI 4-7, near-human magnitude) but
are weak on composite-face; DINOv2 self-supervised priors dominate composite-
face and part-whole context binding (CSI 1.4-1.7, lowest-PWI 0.31-0.40) but
are weaker on Thatcher (1.4-3.3); face-identification-trained models
(FaceNet, VGGFace2 / CASIA-Webface) sit at baseline (≈1.0) across ALL three
paradigms. EEG-side findings: direct analysis of ATM's pre-computed EEG
embeddings across 10 THINGS-EEG2 subjects (1024 CLIP-H/14 dimensions × 200
test concepts) reveals uniform per-dim attenuation (mean Pearson r ≈ 0.158)
with no dimension-specific structure: Thatcher-loaded dimensions are neither
preserved nor selectively destroyed (Spearman ρ = -0.029, permutation
p = 0.35). Face-category test concepts receive no privileged preservation
either (p = 0.37 vs random-matched controls). Together these results show
that current EEG visual decoders cannot capture any of the three holistic-
face illusions: the EEG bottleneck is a non-specific low-pass that washes
out fine perceptual structure regardless of the visual anchor's holistic
emergence. We release the 3-paradigm stimulus generators, 17-prior model zoo,
and per-CLIP-dim preservation analysis pipeline.

---

## Elevator pitch (60 words, for PI / lab conversation)

We built a three-paradigm benchmark — Thatcher, composite, part-whole — for
holistic face illusions and tested 17 visual priors plus ATM's pre-computed
EEG embeddings. CLIP dominates Thatcher, DINOv2 dominates composite + part-
whole, face-trained networks emerge nothing, and ATM's EEG bottleneck
uniformly washes out CLIP fine structure regardless of dimension.
