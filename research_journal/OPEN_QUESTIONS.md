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

- **Status**: **SCOPED, ready for experiment in tick 5** (see E006)
- **Score**: 10/10 (the actual EEG question; central to the project's lab fit)
- **Prior art checked**: ATM/AVDE/ENIGMA/HVF/ViEEG never tested against illusion stimuli. Phillips & White 2026 reviews face-DNN alignment but doesn't cover EEG.
- **Hard problem identified (E006)**: We have NO EEG data of Thatcher stimuli. THINGS-EEG2 is natural objects, not face Thatcher. So we cannot directly "decode Thatcher EEG".
- **Refined discriminating experiment (Route A in E006)**:
  - Use ATM trained checkpoint + THINGS-EEG2 test data
  - Per CLIP-H/14 dimension d, compute "EEG-preservation per dim" = correlation(CLIP_target[:,d], ATM_decoded[:,d]) across trials
  - Per dim d, compute "Thatcher loading" = magnitude of CLIP(V1)−CLIP(V2) component in dim d, averaged across FFHQ Thatcher identities
  - Test: are Thatcher-loaded dimensions well-preserved or destroyed by the EEG bottleneck?
- **Expected** (working hypothesis):
  - ATM trained on natural objects → low EEG preservation of face-specific dimensions → predicted Thatcher signature destroyed (a clean negative result is publishable)
  - If preserved: we have strong evidence current EEG decoders ALREADY inherit the Thatcher signature even though never tested on face stimuli
- **Linked experiment IDs**: E006 (scoping done), E020 (Route A planned for tick 5)

---

## Q006 — Does the ISI hierarchy reflect emergent face-configural processing or training-data orientation distribution?

- **Status**: untested
- **Score**: 7/10
- **Prior art checked**: not yet
- **Discriminating experiment**: TBD; possibly compare CLIP variants trained on differently-curated subsets if available, OR compute "face-image fraction" of LAION-2B and correlate.
- **Linked experiment IDs**: E0?? (deferred)

---

## Q007 — What specifically about CLIP-class training drives the orientation-emergent face-Thatcher sensitivity?

- **Status**: **PARTIALLY ANSWERED — family-general (not CLIP-specific); refined into Q008**
- **Score**: 8/10
- **Resolution experiment**: E005 added SigLIP-base, SigLIP-SO400M, MetaCLIP-H/14.
- **Result**: All three give ISI 3.5-6.1 on face Thatcher, comparable to
  CLIP family. The Thatcher signature is general to image-text contrastive
  training, not OpenAI/LAION-CLIP-specific. Scaling within the family
  (SigLIP-base 3.5 → SigLIP-SO400M 5.6 → MetaCLIP-H/14 6.1) mirrors CLIP's
  own scaling (b32 5.2 → bigG14 6.8). Random-bbox ratio same as CLIP (50-75% drop).
- **Refined picture across 17 priors**:
  - No language + no scale: ISI ≈ 1
  - No language + scale (DINOv2): ISI up to 3.3
  - Language + scale (CLIP / SigLIP / MetaCLIP): ISI 3.5-6.8
  - Face-identity training: ISI ≈ 1 (active suppression)
- **Linked experiment IDs**: E005 ✓

---

## Q008 (NEW from E005) — Can we separate the SCALE contribution from the LANGUAGE contribution?

- **Status**: untested
- **Score**: 7/10
- **Discriminating experiment**: We already have DINOv2 base (86M params, ISI 1.37)
  → large (300M, 2.37) → giant (1.1B, 3.34). If we also test smaller CLIP/SigLIP
  variants and plot ISI vs param count separately for language vs no-language
  curves, we can attribute slope to language additivity.
- **Linked experiment IDs**: E008 (planned, low priority — current data may suffice)
