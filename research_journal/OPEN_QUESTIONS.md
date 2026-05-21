# Open questions

Hypotheses are ranked by score = importance × tractability (0-10). All claims are
model-level (no brain inference; see `feedback-scientific-stance` memory).

---

## Q001 — Is the CLIP ISI signature face-specific or general orientation bias?

- **Status**: **ANSWERED — face-specific dominates (70-80%); small residual general bias (20-30%)**
- **Score**: 9/10 (was load-bearing; now resolved)
- **Resolution experiment**: E003 random-bbox FFHQ control. Same 200 identities, same algorithm, same bbox sizes, but bboxes placed at NON-feature locations (forehead/hair/neck/background, excluded by dilated face-landmark mask).
- **Result**: CLIP-bigG14 dropped from 6.76 (face) → 1.97 (random-bbox); −71% reduction. All 5 CLIP variants dropped 59-74%. DINOv2-large/giant fell to 0.85-0.89 (below 1, no upright bias for non-feature regions). Residual ISI > 1 in CLIP class (1.3-2.1) is real general upright bias but small.
- **Implication for project**: Idea-001 (EEG Thatcher signal preservation) now has a face-feature-specific phenomenon to track through the EEG decoder, rather than a hard-to-interpret orientation-distance ratio.
- **Linked experiment IDs**: E003 ✓
- **Notes**:
  - 2026-05-21 — created.
  - 2026-05-21 23:30 — ANSWERED via E003. Updated.

---

## Q002 — Does verification accuracy (2AFC) match the ISI_pert pattern?

- **Status**: untested
- **Score**: 7/10 (sanity for the metric; not novel by itself)
- **Prior art checked**: Carbon 2005 used 2AFC d' on humans for ISI; not on models. Jacob 2021 used identification-confidence drop.
- **Discriminating experiment**: For each (V_normal_i, V_thatched_i) pair, build a Siamese verification classifier using cosine threshold calibrated on a held-out subset. Compute discrimination d' per orientation. ISI_d' = d'_up/d'_inv.
- **Expected**: ISI_d' should correlate with ISI_pert across models. If not, ISI_pert is measuring something else.
- **Linked experiment IDs**: E004 (planned)

---

## Q003 — Do face-trained backbones (FaceNet, ArcFace) show the strongest ISI?

- **Status**: **ANSWERED — REFUTED. Face-trained models show essentially NO Thatcher effect.**
- **Score**: 8/10 (decisive negative result reshaped the project framing)
- **Resolution experiment**: E004 added P17 FaceNet-VGGFace2 + P18 FaceNet-CASIA-Webface.
- **Result**: FaceNet ISI(face Thatcher) = 1.02-1.12, compared with CLIP-bigG = 6.76.
  FaceNet's d_up=0.38 and d_inv=0.34 are BOTH large (it notices feature
  inversion) but EQUAL across orientations.
- **Implication**: The CLIP-class Thatcher signal is NOT a face-recognition
  feature — it is an orientation-emergent property absent in models trained
  for pose-invariant identity matching. Tracking the explanation now becomes Q007.
- **Linked experiment IDs**: E004 ✓

---

## Q004 — Do perception-aligned priors (Harmonized) show different ISI from vanilla CLIP/ViT?

- **Status**: untested
- **Score**: 8/10 (key control for the Serre-lab theoretical framing)
- **Prior art checked**: Fel et al. 2022 (NeurIPS) — Harmonized models more aligned with human attention on natural images. Not tested on holistic illusions.
- **Discriminating experiment**: Add Harmonized-ResNet50, Harmonized-ViT, Harmonized-EfficientNet (Serre lab releases). Extract on FFHQ Thatcher; compute ISI; compare with vanilla counterparts.
- **Expected (perceptual alignment helps)**: ISI(Harmonized) > ISI(vanilla same-arch)
- **Linked experiment IDs**: E006 (planned)

---

## Q005 — Does the EEG-to-image decoding pipeline preserve or destroy the image-side ISI signature?

- **Status**: untested
- **Score**: 10/10 (the actual EEG question; central to the project's lab fit)
- **Prior art checked**: ATM/AVDE/ENIGMA/HVF/ViEEG never tested against illusion stimuli. Phillips & White 2026 reviews face-DNN alignment but doesn't cover EEG.
- **Discriminating experiment**: For ATM (CLIP-H/14 visual anchor), AVDE (LaBraM + CLIP-H/14), ENIGMA (lightweight CLIP):
  - Extract the EEG-conditioned image embedding by running their full pipeline on THINGS-EEG2 test set
  - Compare image-side embedding ISI vs EEG-conditioned embedding ISI on the Thatcher battery
- **Expected**: Multiple scenarios — the result IS the contribution.
- **Linked experiment IDs**: E020+ (deferred; blocked on Q001 disambiguation)

---

## Q006 — Does the ISI hierarchy reflect emergent face-configural processing or training-data orientation distribution?

- **Status**: untested
- **Score**: 7/10
- **Prior art checked**: not yet
- **Discriminating experiment**: TBD; possibly compare CLIP variants trained on differently-curated subsets if available, OR compute "face-image fraction" of LAION-2B and correlate.
- **Linked experiment IDs**: E0?? (deferred)

---

## Q007 (NEW from E004) — What specifically about CLIP-class training drives the orientation-emergent face-Thatcher sensitivity?

- **Status**: untested
- **Score**: 8/10
- **Prior art checked**: pending
- **Background**: E004 showed face-recognition training does NOT produce
  Thatcher ISI; CLIP does. Three candidate mechanisms:
  (a) **scale + diverse data**: any sufficiently large image model on web data emerges this
  (b) **text-image alignment**: language captions encode upright-orientation defaults
  (c) **contrastive objective without pose-invariance**: not penalizing orientation-variance
- **Discriminating experiment** (E005): Add SigLIP, EVA-CLIP, MetaCLIP — all
  image-text contrastive but with different training data/objectives. If all
  give ISI 4-7 like CLIP → (a) or (b) generic to image-text training. If they
  diverge widely → (c) specific.
  Also: add MoCo-v3 / iBOT / CLIP-fine-tuned-on-faces if available, to
  separate text-from-scale.
- **Expected outcomes**:
  - All similar to CLIP: text-image-training-general → Q005 EEG story unchanged
  - Wide variance: training-specific → Q006 / Q007 deepen
- **Linked experiment IDs**: E005 (planned, next-tick candidate)
