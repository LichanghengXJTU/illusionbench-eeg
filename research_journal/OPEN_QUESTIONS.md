# Open questions

Hypotheses are ranked by score = importance × tractability (0-10). All claims are
model-level (no brain inference; see `feedback-scientific-stance` memory).

---

## Q001 — Is the CLIP ISI signature face-specific or general orientation bias?

- **Status**: untested
- **Score**: 9/10 (load-bearing for the whole interpretation of E001/E002)
- **Prior art checked**: partial — Jacob et al. 2021 (Nat Comm) showed face-trained DNNs reproduce Thatcher; did NOT test whether non-face images also produce this asymmetry under matched perturbation. Search needed.
- **Discriminating experiment**: non-face Thatcher control (E003). Take 200 ImageNet non-face images, define 2 random local bboxes per image, apply 180° rotation. Run all 12 priors. Compute ISI.
- **Expected outcome to support "face-specific configural"**: ISI(nonface, CLIP-bigG) ≈ 1.0-1.5 (no upright-orientation effect on non-face local perturbation)
- **Expected outcome to refute (= general orientation bias)**: ISI(nonface, CLIP-bigG) ≈ 4-7 (same magnitude as face Thatcher)
- **Linked experiment IDs**: E003 (planned)
- **Notes**:
  - 2026-05-21 — created. This is the FIRST thing to resolve in tick 1.

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

- **Status**: untested
- **Score**: 8/10 (decisive for face-specificity claim)
- **Prior art checked**: Jacob 2021 reported Thatcher emergence in VGG-Face; no head-to-head with CLIP/DINOv2.
- **Discriminating experiment**: Add FaceNet (VGGFace2-trained Inception-Resnet) and ArcFace to model zoo; extract on FFHQ Thatcher; compute ISI.
- **Expected (face-specificity hypothesis)**: ISI(face-trained) > ISI(CLIP-bigG) > ISI(DINOv2-giant) > ...
- **Linked experiment IDs**: E005 (planned)

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
