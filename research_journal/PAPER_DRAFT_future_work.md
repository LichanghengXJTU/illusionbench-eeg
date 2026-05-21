# Future Work + Open Questions (paper supplementary / Section 6 expansion)

**Audience**: paper reviewers + lab seniors. Each item is concrete enough to
become a follow-up paper or a thesis chapter.

---

## A. Cross-decoder EEG bottleneck mapping

**Question**: Is the uniform per-dim attenuation we observe in ATM (r ≈ 0.158)
characteristic of EEG-to-CLIP projection in general, or is it ATM-specific?

**Experiment proposal**:
1. Once AVDE, ENIGMA, HVF release pre-computed EEG embeddings (or once we can
   train Route A heads against their inference pipelines), repeat the per-dim
   preservation analysis.
2. Predicted outcomes:
   - If all decoders show r ≈ 0.15-0.2 → uniform low-pass is a property of
     the EEG modality and current decoder architectures generally
   - If some decoders preserve specific dimensions better → architectural
     differences matter, and we can identify what helps
3. Decoder zoo to test (priority order): ENIGMA (most likely to release),
   AVDE (NeurIPS 2026 implementation should expose intermediates),
   HVF (HKUST-GZ — internal channel available), ViEEG (no code yet).

**Estimated cost**: 0.5-1 day per decoder once embeddings are accessible.

---

## B. Multi-prior EEG decoder prototype

**Question**: Can a multi-prior EEG decoder (CLIP + DINOv2 + face-recognition
target) preserve more of the dimension-fine perceptual structure than a
single-CLIP anchor?

**Experiment proposal**:
1. Train a small EEG encoder on THINGS-EEG2 with three contrastive heads:
   one to CLIP-H/14, one to DINOv2-giant, one to FaceNet-VGGFace2.
2. Compute Route A preservation for each head.
3. On our IllusionBench-EEG stimuli, compute the resulting ISI/CSI/PWI on
   the EEG-decoded multi-prior embedding.
4. Predicted outcome: if multi-prior beats single-CLIP on at least 2 of 3
   paradigms while not destroying retrieval, this is a strong architectural
   recommendation.

**Estimated cost**: 1-2 weeks training + analysis. Compute: a few thousand
GPU-hours.

---

## C. Behavioral-anchored EEG metric: 2AFC verification framing

**Question**: Does our embedding-distance-based ISI map onto a more
behaviorally-grounded "is this the same modified vs unmodified face?" 2AFC
discrimination accuracy?

**Experiment proposal**:
1. For each model, compute a calibrated cosine-distance threshold from
   held-out within-identity pairs.
2. Predict "same person?" for each (V1, V2) pair across the 3 paradigms.
3. Compute d′ from the resulting hit/false-alarm rates per orientation.
4. Compare with our ISI_pert measure across all 17 models.

**Predicted outcome**: ISI_pert and behavioral-d′ should be monotonically
related; if they diverge, the gap is itself informative (some models may have
high embedding-distance shift but unstable verification accuracy due to
threshold sensitivity).

---

## D. Beyond face: non-face holistic / contour illusions

**Question**: Do the same model classes that dominate face Thatcher dominate
non-face holistic illusions (Mooney faces, Kanizsa contours, Navon
global-local)?

**Experiment proposal**:
1. Add Mooney-face stimuli (high-contrast threshold binarization of FFHQ faces)
2. Add Kanizsa subjective-contour stimuli (illusory triangle)
3. Add Navon hierarchical letters (large H made of small Hs / Ss)
4. Compute embedding-distance metrics on the upright vs inverted contrasts.

**Predicted outcome**: DINOv2's "spatial-holistic" property would predict
strong response on Mooney (contour completion) and Navon (global-level
perception). CLIP would predict weaker.

**Estimated cost**: 1-2 weeks for stimulus generation + extraction.

---

## E. Direct EEG of Thatcher / composite / part-whole stimuli

**Question**: When real subjects view holistic-face-illusion stimuli, does
their EEG carry the orientation × paradigm interaction that the corresponding
CLIP embeddings show?

**Experiment proposal** (BIG):
1. Replicate the THINGS-EEG2 RSVP paradigm but with our 3-paradigm IllusionBench
   stimulus battery (~3,200 trials per subject)
2. Collect EEG from N=20 subjects
3. Train decoders specifically on this paradigm
4. Measure preservation directly (not via inference from THINGS preservation)

**Estimated cost**: 3-6 months of lab work (ethics + collection + analysis).
**Lab fit**: HKUST EEG-decoding lab — this is plausibly within scope.

---

## F. Harmonized prior comparison (Q004, currently blocked)

**Question**: Do perception-aligned visual priors (Serre lab's Harmonized
models, ClickMe-trained) show higher / lower / equal Thatcher ISI than
vanilla CLIP variants?

**Experiment proposal**:
1. Resolve the TF/Keras 3 compatibility issue documented in NEED-001
   (NOTES_FOR_USER.md). Options:
   - Install TF 2.15 + Keras 2 in a separate Python 3.11 environment
   - Port Harmonized weights to PyTorch (requires architecture mapping)
   - Use DINOv2 + THINGS-similarity fine-tune as fallback
2. Add Harmonized models to the model zoo and compute all three paradigm
   indices.
3. Compare to vanilla counterparts (Harmonized ResNet50 vs vanilla
   torchvision ResNet50).

**Estimated cost**: 1-2 days if TF 2.15 venv works; 1 week if PyTorch port
required.

---

## G. CLIP-L14 outlier analysis (Q009)

**Question**: Why does OpenAI CLIP-L/14 (P03) systematically behave as an
outlier across all three paradigms? Thatcher ISI 4.0 (others 5-7), composite
CSI 1.60 corrected (others 1.10-1.20)?

**Experiment proposal**:
1. Compare CLIP-L/14's image preprocessing (Resize, CenterCrop, normalization
   means/stds) with LAION CLIP variants in the same script. Confirm they are
   identical (likely).
2. Compare training-data composition: OpenAI 400M curated pairs vs LAION-2B
   web-crawl. The OpenAI curation may have demographic / aesthetic biases
   that propagate.
3. Embed a sample of FFHQ faces with both CLIP-L/14 and CLIP-H/14, compute
   their pairwise CLIP-space distances → does CLIP-L/14 cluster differently?

**Estimated cost**: 1-2 days.

---

## H. Dimension-fine EEG architecture (the architectural prediction made real)

**Question**: We predict in §5.2 that EEG decoders preserving dimension-fine
perceptual structure are required for human-aligned holistic processing.
Can we prototype one?

**Architectural sketch**:
1. Identify the top-K CLIP-H/14 dimensions ranked by Thatcher-loading.
2. Add a per-dim loss multiplier so the EEG-to-CLIP projection objective
   weights those dimensions higher.
3. Train the resulting architecture on THINGS-EEG2.
4. Measure (i) THINGS retrieval (must not collapse), (ii) per-dim
   preservation r (should increase for the targeted dims), (iii) the
   predicted EEG-side ISI on hypothetical Thatcher EEG (via inference).

**Estimated cost**: 1 week for the prototype training + analysis.

---

## I. Scaling law for holistic emergence

**Question**: Within image-text contrastive training, how does Thatcher ISI
scale with model parameter count?

**Data already collected**:
| Model | Params | Thatcher ISI |
|---|---|---|
| CLIP-B/32 | 151M | 5.18 |
| CLIP-L/14 | 428M | 4.05 |
| CLIP-H/14 LAION | 986M | 5.50 |
| CLIP-g/14 LAION | 1.4B | 5.54 |
| CLIP-bigG/14 LAION | 2.5B | 6.76 |
| SigLIP-base | 94M | 3.51 |
| SigLIP-SO400M | 400M | 5.59 |

A clean power-law fit (ISI vs log-params, family-restricted) would yield
**emergence-prediction** behavior: at what parameter count does an image-text
model cross ISI = 4 (lower edge of human range)?

**Estimated cost**: 0.5 days (data already exists, just need fit + plot).

---

## J. ERP-style stage analysis (theoretically motivated, not yet feasible)

**Question**: Carbon et al. (2005) found N170-style ERP signatures of Thatcher
detection appear EVEN FOR INVERTED Thatcher conditions where conscious report
fails. Does the EEG preservation profile differ between early (N170 ~140-200ms)
and late (300-500ms) windows?

**Experiment proposal**:
1. Re-run Route A separately for early-time vs late-time EEG epochs.
2. Predicted outcome: early window preservation should be more
   orientation-invariant; late window should show orientation-asymmetry.
3. This would map our embedding-distance dissociation onto Carbon's
   neural-temporal dissociation, strengthening the bridge to neuroscience.

**Estimated cost**: blocked on per-time-window EEG embeddings not being
released by ATM authors. Could be done in-house with our own training.

---

## Triage

The most directly publishable in the present scope would be:
1. **(A)** cross-decoder Route A — strengthens Claim 4 substantially
2. **(I)** scaling law — uses existing data, gives a clean predictive figure
3. **(C)** behavioral verification framing — sanity check that strengthens Claim 1
4. **(F)** Harmonized comparison — completes the model-zoo

The bigger, longer-horizon items are:
5. **(B)** multi-prior decoder prototype — a separate paper
6. **(E)** direct Thatcher-EEG collection — a separate paper, lab-fit specific
7. **(D)** non-face illusion battery — extends the benchmark substantially
8. **(H)** dimension-fine EEG architecture — methods paper
9. **(J)** ERP-stage analysis — requires per-time-window EEG access
