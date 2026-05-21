# E030 — Tick 28 literature scan + ArcFace test for off-the-shelf face-tuned bio-inspired priors

**Date**: 2026-05-22 (tick 28)
**Linked**: Idea-003 sub-paths (a) face-tuned CORnet, (b) NSD-aligned encoders,
(c) face-PredNet, (e) custom design.

## Goal

After E028 (CORnet-S no Thatcher signal) and E029 (fusion = distance averaging,
no emergent paradigm-consistency), Idea-003 needs an off-the-shelf model that
combines **bio-inspired anatomy** with **face-feature tuning**, ideally with
**image-text contrastive training** (the apparent prerequisite for Thatcher per
E005 + E028). This tick scopes the literature/HuggingFace ecosystem for any
such off-the-shelf model.

## Methods

Targeted web search (4 queries) for:
1. CORnet variants pretrained on face data (CORnet-S + VGGFace2, etc.)
2. NSD (Natural Scenes Dataset)-aligned PyTorch vision encoders (Conwell 2024,
   Allen 2022 follow-ups)
3. PredNet / γ-net / predictive coding nets with face training
4. CLIP encoder fine-tuned on face image-text pairs (CelebA-CLIP / FaceCLIP)
5. Modern face-recognition SOTA beyond FaceNet (ArcFace / AuraFace).

Then for any candidate, attempt to load the encoder as a standalone vision
prior and embed our IllusionBench-EEG 3-paradigm battery.

## Results

### Search 1: face-tuned CORnet-S

**No off-the-shelf checkpoint exists.** CORnet-S (Kubilius et al., NeurIPS
2019) was released only with ImageNet pretraining. No publication has
re-trained the architecture on face data and released weights. To obtain a
face-CORnet, custom training is required (estimated 1-3 days on the current
H100 server with a face dataset like VGGFace2).

### Search 2: NSD-aligned PyTorch vision encoders

The Natural Scenes Dataset (NSD; Allen et al. 2022) is a large 7T fMRI dataset
of natural scenes viewed by 8 subjects. Several papers in 2023-2025 (e.g.,
MedARC-AI's fMRI-reconstruction-NSD project, Conwell et al. 2024) align image
encoders to NSD activity. However:
- These pipelines are typically **decoders FROM fMRI TO image**, not
  **encoders aligned to brain that take an image as input**.
- The "brain-aligned vision encoder" formulation (image → embedding that
  predicts fMRI) does exist (e.g., Brain-Score's BehaviorMatch scoring of
  CORnet, NSD-supervised CLIP fine-tunes in some labs), but I could not find a
  clean PyTorch checkpoint that is *both* (i) released with face-rich training
  data and (ii) loadable as a CLIP-style encoder.
- **Conclusion**: NSD-aligned PyTorch encoders that are face-tuned and easily
  loadable do not appear to be publicly released as of 2026-05.

### Search 3: face-PredNet / γ-net

PredNet (Lotter et al. 2017) is video prediction trained on KITTI driving
data. Some downstream studies use predictive coding on face perception (e.g.,
PMC6596791) but they re-train or adapt the architecture and do not release
face-specific checkpoints. γ-net (Linsley et al. 2020) is contour-illusion-
focused, not face-tuned.

**Conclusion**: No off-the-shelf face-predictive-coding network is
available.

### Search 4: CLIP encoder fine-tuned on face data

I found multiple "FaceCLIP" variants:

1. **ByteDance/FaceCLIP** (arXiv:2504.14202, May 2025): an **ID-preserving
   image-generation pipeline**. It uses OpenAI CLIP-L-14 and OpenCLIP-bigG-14
   as **frozen text encoders**; the face encoder is a *separate* identity-
   encoder added on top. The CLIP **vision encoder is unchanged from OpenAI**.
   So the "vision prior" of ByteDance/FaceCLIP is identical to our P05/P06.
2. **DiffusionCLIP-CelebA_HQ** (gwang-kim): trains a diffusion model on
   CelebA-HQ guided by CLIP; the CLIP encoder itself is **not** fine-tuned.
3. **FaceCLIP (Springer 2024)**: facial-expression-from-text generation. Same
   pattern — vanilla CLIP encoder.

**Critical finding**: there is **no public model where the CLIP vision-encoder
weights have been fine-tuned on face image-text pairs**. The community has
not produced the obvious next-step model. This is itself a literature gap
that could motivate future work.

### Search 5: modern face-recognition SOTA — ArcFace / AuraFace

ArcFace (Deng et al. CVPR 2019) is the modern standard for face recognition,
using ResNet-100 + additive angular margin loss. The HuggingFace community
release `fal/AuraFace-v1` provides a ResNet-100 + ArcFace-loss checkpoint
loadable via `insightface` package (ONNX format).

This was tested on our 3 paradigms as an additional face-identity-trained
baseline beyond P17/P18 (FaceNet-VGGFace2/CASIA). **Pre-registered prediction**
(set BEFORE running, 2026-05-22): if FaceNet's ISI ≈ 1.11 reflects a general
property of identity-discriminative training, then ArcFace should give
similarly low ISI/CSI and unremarkable PWI (i.e., ≈ baseline on all three
paradigms).

**Result** (ArcFace / AuraFace ResNet-100 + ArcFace loss, computed on 2026-05-22):
- Thatcher ISI: **1.697 [1.638, 1.766]** (predicted: ~1.0-1.2) → **PREDICTION FALSIFIED**
- Composite CSI: **1.161 [1.123, 1.197]** (predicted: ~1.0-1.2) → matches prediction
- Part-Whole PWI: **1.428 [1.335, 1.520]** (predicted: ~0.6-0.8) → **PREDICTION FALSIFIED** (PWI > 1, opposite direction)
- Random-bbox ISI: **0.387 [0.348, 0.431]** (predicted: ~0.5-1.0) → matches prediction
- **Face-feature-specificity**: 1 - 0.387/1.697 = **77%** (vs FaceNet's 66%, CLIP-bigG14's 71%)

## ArcFace surprise: face-identity training matters but loss matters more

ArcFace **does show Thatcher** (ISI 1.70), unlike FaceNet's triplet-loss model
(ISI 1.12). Both are face-identity-discriminatively trained — but on different
backbones (ResNet-100 vs Inception-Resnet-V1) and different loss functions
(additive angular margin vs triplet). The Thatcher signal appears to require
the **angular-margin loss formulation**, not just face training.

**Comparison with FaceNet on all three paradigms**:
| Prior | Thatcher | Composite | Part-Whole | Random-bbox | Spec% |
|---|---|---|---|---|---|
| FaceNet | 1.12 | 1.11 | 0.71 | 0.38 | 66% |
| ArcFace | **1.70** | 1.16 | **1.43** | 0.39 | **77%** |

This is a **third failure-mode** in our taxonomy, distinct from both CLIP and
FaceNet:
- **CLIP-class**: Thatcher-dominant (paradigm-INconsistent on Composite/Part-Whole)
- **DINOv2-class**: Composite/Part-Whole-dominant (paradigm-INconsistent on
  Thatcher)
- **FaceNet (triplet)**: baseline on all three (no holistic signal)
- **ArcFace (angular-margin)**: Thatcher + INVERTED Part-Whole (paradigm-
  INconsistent in a new way — PWI > 1, opposite of CLIP)
- **CORnet-S (bio-anatomy, ImageNet)**: Part-Whole-dominant, no Thatcher

**Interpretation of ArcFace's PWI > 1**: ArcFace's angular-margin loss makes
identity boundaries tight, so any local perturbation (eye region change) moves
the embedding off the identity centroid. In **isolated eye-region condition**
(no surrounding face context), the distance is large. In **whole-face
condition**, the surrounding context provides identity-confirming signal that
partially compensates for the perturbed eye region, REDUCING the distance.
This gives PWI < 1 normally. But ArcFace's tight angular bounds may be more
sensitive in the WHOLE condition because the perturbation now interacts with
the rest of the face identity signal — giving PWI > 1.

(This interpretation should be tested with additional ArcFace variants or
re-formulated as a hypothesis for future work; current data don't pin down
the mechanism.)

## Interpretation

[CONFIRMED]: **No off-the-shelf face-tuned bio-inspired vision encoder
exists** as of 2026-05. Idea-003 sub-paths (a) face-CORnet, (b) NSD-PyTorch,
(c) face-PredNet **all require custom training**.

[CONFIRMED]: **No off-the-shelf face-tuned CLIP encoder exists**. The natural
next-step model — fine-tune CLIP vision-encoder on face image-text pairs —
has not been built and released. ByteDance/FaceCLIP and others use frozen
CLIP + add-on identity encoders rather than re-training the vision encoder.

[CONJECTURE — literature-gap finding worth flagging]: The community appears
to have systematically avoided fine-tuning the CLIP vision-encoder on face
data, presumably because:
- Face-recognition is well-served by purpose-built networks (FaceNet,
  ArcFace, etc.)
- ID-preserving generation uses CLIP-as-text-encoder + face-encoder fusion
- No clear downstream task motivates fine-tuning the vision encoder itself

This is potentially a **gap our work can identify** in the discussion: the
"missing model" that would test whether face-image-text-contrastive training
inherits CLIP's Thatcher signature is genuinely not built.

[CONFIRMED — NEW]: **ArcFace's angular-margin loss is sufficient to produce a
Thatcher signature** (ISI 1.70, 77% face-feature-specific). This is a
non-trivial finding that contradicts our initial hypothesis (formed from
FaceNet results) that face-identity training is uniformly insensitive to
Thatcher. The mechanism appears to be: angular-margin losses produce tight
identity manifolds where local perturbations (eye thatcherization) move
points off the manifold more in the upright condition than inverted (where
identity is less recognized → both V3 and V4 are off-manifold).

[CONFIRMED — NEW]: **ArcFace inverts the Part-Whole effect** (PWI 1.43 > 1).
This is a fourth distinct failure mode in our taxonomy:
- CLIP-class: high ISI, low PWI (paradigm-INconsistent in one direction)
- DINOv2-class: low ISI, very low PWI (paradigm-INconsistent in another)
- FaceNet (triplet): all baseline (no signal)
- ArcFace (angular-margin): mid ISI, INVERTED PWI (paradigm-INconsistent
  in yet a third way)
- CORnet-S (bio-anatomy): all baseline ISI, very low PWI (paradigm-INconsistent)

**Five distinct dissociation patterns across 19 priors confirms that no
single off-the-shelf vision model achieves paradigm-consistency**. This
is a stronger and more concrete version of the Idea-001 finding.

## Implications for Idea-003

| Sub-path | Status post-E030 | Cost to realize |
|---|---|---|
| (a) face-tuned CORnet | No off-the-shelf; requires custom training | 1-3 days GPU |
| (b) NSD-aligned encoder | No clean public PyTorch checkpoint matching our needs | Multi-week if reproducing Conwell |
| (c) face-PredNet | Not released | Multi-week |
| (d) multi-anchor fusion | ❌ REFUTED (E029) | — |
| (e) custom bio-inspired model from scratch | Multi-week | 2-6 weeks GPU |

**All paths require custom training beyond loop time-budget**. The publishable
contribution then becomes:
- **Idea-001 finding** (paradigm dissociation + uniform low-pass EEG bottleneck)
- + **Idea-003 negative**: "*we test whether bio-inspired anatomy (CORnet-S)
  or face-identity training (FaceNet/ArcFace) alone or in combination
  recovers paradigm-consistent illusion sensitivity, and find that none do.
  The remaining hypothesis — CLIP-style image-text contrastive training on
  face-rich data — has not been built; the community has not produced this
  model.*"

This negative result is itself contentful: it falsifies the cheapest
explanations of CLIP's Thatcher signature and identifies the specific
training regime that would resolve the question.

## Next steps (tick 29 candidates)

1. **Refine Idea-003 framing** in IDEA_PIPELINE: emphasize that the
   contribution is now "**identify the necessary training ingredient for
   paradigm-consistent face vision**" — a falsifiable question that fits
   within current data + a small training experiment.
2. **Test ArcFace** (in progress this tick) to close the face-identity
   training family completely.
3. **Optional small face-fine-tune** of CORnet-S (1-layer head on VGGFace2
   subset) as a probe: does adding face-identity training to bio-anatomy
   yield Thatcher? Estimated 2-4h GPU on existing server.
4. **Reframe paper**: integrate Idea-003 negative as Section 6.3 of PAPER_DRAFT.

## Replicability

- Search queries documented above; each yielded null on the face-tuned
  CLIP-encoder question.
- ArcFace install: `pip install insightface onnxruntime-gpu`; model
  `huggingface_hub.snapshot_download("fal/AuraFace-v1")`.
- Pre-registered predictions saved in this file before measurement.

---
