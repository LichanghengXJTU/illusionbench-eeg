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

- **Status**: **ANSWERED — image-text family ≈ DINOv2 SSL in slope (1.71 vs 1.78 per log10 M params) but offset by +2.9 ISI units**
- **Score**: 7/10
- **Resolution experiment**: E026 fit per-family linear regressions on log-params (n=8 image-text, n=3 DINOv2).
- **Result**: slopes nearly identical; intercepts differ by +2.9 in favor of
  image-text family. **Language conditioning produces a constant offset on
  top of the shared scale-emergence rate.**
- **Implication**: image-text training crosses ISI=4 (lower human edge) at
  ~72M image-encoder params; DINOv2 SSL would need ~2.5B params. Both
  families exhibit the same slope ~1.75 per decade of scale.
- **Linked experiment IDs**: E026 ✓

---

## Q010 (NEW from user 2026-05-22) — Are existing bio-inspired vision priors paradigm-consistent on IllusionBench-EEG, and competitive on standard EEG-to-image retrieval?

- **Status**: **ANSWERED — CORnet-S is NOT paradigm-consistent; it captures Part-Whole but no Thatcher or Composite.**
- **Score**: 10/10 (Idea-003 hinged on this; resolved)
- **Resolution experiment (E028)**: CORnet-S (DiCarlo lab, ImageNet-trained) gave Thatcher ISI 0.99 [0.90, 1.08] (= pixel baseline), Composite CSI 1.15 (≈ pixel baseline 1.10), Part-Whole PWI 0.21 [0.19, 0.24] — the LOWEST PWI of all 25 priors in our battery.
- **Interpretation**: bio-inspired anatomy alone (V1→V2→V4→IT + recurrence) trained on ImageNet captures whole-face context dominance over local perturbations (Part-Whole) but NOT face-feature-specific orientation interaction (Thatcher) and NOT spatial-alignment integration (Composite). The full configural-illusion repertoire requires combining bio-anatomy with face-specific data and training objective. Sub-paths (b) NSD-aligned, (c) face-PredNet, (d) multi-anchor fusion all ruled out by E029-E030.
- **Linked experiment IDs**: E028 ✓ (the answer), E029 (refuted sub-path d), E030 (no off-the-shelf face-bio), E031 (refined sub-path a), E032 (AdaFace-CASIA reveals the candidate recipe).

---

## Q011 (NEW from E031/E032) — Does face-recognition with angular-margin loss + small noisy training data uniformly produce a Thatcher-like signal, or is AdaFace-CASIA an idiosyncratic case?

- **Status**: partially answered (1 angular-margin checkpoint at IR-50 + CASIA showed ISI 2.91). Need more datapoints in the (loss, backbone, data) cube to claim generality.
- **Score**: 8/10
- **Background**: E031 showed AdaFace IR-101 MS1MV2 (ISI 1.24), ArcFace IR-101 WebFace4M (ISI 1.17), AdaFace IR-50 MS1MV2 (1.21), AdaFace IR-50 WebFace4M (1.33) — all face-rec models with ISI < 1.5. Only AdaFace IR-50 CASIA (P26) gave ISI 2.91. This single datapoint is the basis for Claim 6 (data-inversion).
- **Discriminating experiments**:
  1. Add SphereFace / CosFace ResNet-50 checkpoints on CASIA-WebFace if released.
  2. Add AdaFace IR-50 on additional small noisy datasets (e.g., LFW ~5K identities, IJB-A, AgeDB-30).
  3. Train AdaFace IR-50 on a downsampled subset of MS1MV2 (matching CASIA size + identity count) — does it ALSO give ISI ~3?
- **Expected outcomes**: If multiple angular-margin face-rec models on CASIA-sized data converge on ISI ~2.5-3, the data-inversion claim is fully general. If AdaFace IR-50 CASIA is unique, the claim narrows to specific (backbone × dataset × loss) interactions.
- **Linked experiment IDs**: E031 ✓, E032 ✓. Future: E036 (planned).

---

## Q012 (NEW from E033/E034) — How robust is the paradigm-conditional cluster migration across (a) clustering algorithms, (b) k choices, (c) larger benchmarks?

- **Status**: untested for robustness; the qualitative pattern was observed at k=6 using average-linkage.
- **Score**: 6/10 (foundational for Claim 7 robustness)
- **Discriminating experiments**:
  1. Re-cluster at k=4 and k=8; check if CORnet-S, AdaFace-CASIA, FaceNet migrations persist.
  2. Use Ward linkage and complete linkage instead of average; check stability.
  3. Bootstrap the stimulus set (resample 80%) and recompute RSA — does each subject sample yield the same cluster taxonomy?
- **Linked experiment IDs**: E033 ✓, E034 ✓. Future: E037 (planned).

---

## Q013 (NEW from E035) — Are there ANY THINGS-EEG2 concepts for which the EEG-side embedding is selectively above-baseline?

- **Status**: open (single per-dim category-level analysis in E021 was NULL; not exhaustive)
- **Score**: 6/10 (sanity for Claim 4)
- **Background**: E021 took the top-30 face-related test concepts (by CLIP-text similarity) and found per-dim r = 0.150 vs random-matched 0.145 (NULL). But there could be other concept categories (e.g., animals, tools, foods) where EEG selectively preserves CLIP. If any positive subset exists, the uniform low-pass story has a wrinkle.
- **Discriminating experiment**: cluster the 200 test concepts via CLIP-text similarity to 12 category labels (food, tool, animal, vehicle, face, body part, plant, building, container, weapon, instrument, garment). For each category, compute mean r and compare with random-matched subset. Report any category with z > 2.
- **Linked experiment IDs**: planned E038.

---

## Q009 (NEW from E023) — Why is CLIP-L14 systematically outlier in opposite direction across paradigms?

- **Status**: untested
- **Score**: 6/10
- **Background**: OpenAI CLIP-L14 (P03) consistently behaves as an outlier:
  Thatcher ISI 4.0 (others 5-6.8), composite CSI 1.60 corrected (others 1.10-1.20).
  The systematic cross-paradigm reversal is suspicious — could be a real
  training-data or architectural artifact, or it could reflect that P03's
  ImageNet preprocessing differs from LAION CLIPs.
- **Discriminating experiment**: Examine P03's image preprocessing pipeline
  (Resize size, crop, normalization) vs LAION CLIPs. If preprocessing
  differs, the apparent outlier may be measurement artifact. If same,
  it's a genuine model-training-data effect worth investigating.
- **Linked experiment IDs**: E009 (planned)

---

## Q014 (NEW from E040, tick 50) — Does the FULL HOLO-Net (bio components ON) optimize at all, or is the multi-task + recurrent + predictive-coding landscape pathological?

- **Status**: **ANSWERED (tick 51, E042) — the bio components are NOT the
  blocker. The v1 failure was an AdaFace loss bug (an extra `− scale·margin`
  term, already fixed in `losses.py`). The FULL HOLO-Net with the fixed loss
  starts at identity loss 13.20 ≈ the minimal model's 13.13, and runs cleanly.
  The "minimal mode" detour was unnecessary; HOLO-Net trains.**
- **Score**: 9/10 (load-bearing — Idea-003's entire thesis depends on the full
  model training; the falsification criteria are evaluated at the FFA layer,
  which is disabled in the only configuration that currently trains).
- **Background**: E040 run v1 — full HOLO-Net (all bio components) + AdamW
  lr≈1e-4 + 360K classes — identity loss stuck 63-69 for 8750 steps, no
  decrease. Minimal mode (all bio components OFF) + SGD lr 0.1 + 10K classes
  trains fine (13→5.8). The two configs differ on THREE axes simultaneously,
  so the failure is not attributable.
- **Discriminating experiment**: from the known-trainable config (minimal,
  SGD lr 0.1, num_classes=10000), re-enable bio components ONE at a time:
  +Predify PC → +FFA recurrent attention → +Orientation Gate → +OFA branch
  → +LGN-Magno dual-stream → +PFC-Gist. Train each ~2-3K steps; record
  whether identity loss still descends. Also test: full components + SGD 0.1
  (isolates optimizer from architecture).
- **Expected outcome**:
  - If a single component flips training from "descends" to "stuck" → that
    component is the culprit; fix in isolation (re-design / detach its
    gradient path / re-weight its aux loss / `.detach()` the PC target).
  - If even one extra component breaks it → the 6-term multi-task formulation
    needs a principled re-weighting (e.g., GradNorm / uncertainty weighting)
    or staged curriculum (identity-only warmup, then add aux losses).
  - If full-components + SGD 0.1 trains → the v1 failure was just AdamW
    lr 1e-4, and HOLO-Net is salvageable cheaply.
- **Why it matters**: Idea-003 currently rests on a model that does not
  optimize. Until Q014 is answered, the headline novelty (bio-fidelity
  architecture) is untested and the current run only delivers sub-path (a).
- **Linked experiment IDs**: E040 ✓ (the observed failure), E042 ✓ (the
  controlled re-test that isolated the cause).

---

## Q015 (NEW from E042 monitoring, tick 53) — Is the HOLO-Net Orientation Gate actually learning upright-vs-inverted?

- **Status**: **ANSWERED (tick 54) — REAL BUG, fixed.** The `OrientationGate`
  global-average-pooled `afp_spatial` before its classifier; a vertical flip is
  a spatial permutation and a mean is permutation-invariant, so the gate input
  was provably orientation-blind (`GAP(t)−GAP(flip t)` = 0.000000 exact).
  Fix: pool to a coarse 4×4 grid instead of 1×1 (`pool4×4` changes 131% under
  flip). E042 run v1 killed (its gate could never learn → Thatcher mechanism
  inert); run v2 relaunched with the fix. See E042 addendum.
- **Score**: 8/10 (the Orientation Gate is load-bearing: design doc §7 makes it
  the mechanism of the Thatcher effect — "gate suppresses FFA recurrence on
  inverted faces". If it never learns, the full HOLO-Net would give Thatcher
  ISI ≈ 1, i.e. a null result.)
- **Observation**: in the E042 full run, the `orientation` CE loss is flat —
  0.695 (step 0) → 0.695 (step 1700) → 0.707 (step 3850) ≈ ln 2 (chance). The
  upright/inverted distinction (a whole-image vertical flip) should be an easy
  task; 3850 steps of no movement is suspicious.
- **Candidate explanations**: (a) genuinely slow — aux weight is only 0.1 and
  global grad-clip norm 1.0 may attenuate it; the backbone is still forming so
  `afp_spatial` may not yet support the readout; (b) a real bug — gradient to
  the OrientationGate / AFP path is blocked, or the label path is wrong.
  Code audit (tick 53) found no obvious structural bug: the orientation CE
  backprops into the gate classifier and into `afp_spatial`.
- **Discriminating check**: monitor the loss to **step ~10000**. If still ≈ 0.69
  → real bug → investigate (probe gate-classifier gradient norms; verify
  `orientation` labels in the batch; consider raising the aux weight or a warmup
  for it). If descending by step 10000 → it was just slow; no action.
- **Linked experiment IDs**: E042.

---

## Q016 (NEW from E042 runs v1-v3, tick 56) — Which bio component breaks the full HOLO-Net's identity training?

- **Status**: **SUPERSEDED / based on a misjudgment (tick 58).** The premise —
  that the full model "fails" to train — was wrong: it rested on judging
  v1/v2/v3 during steps 4000-7000, but the minimal reference trajectory (fetched
  tick 58) shows identity bounces 10-13.6 until ~step 8000 in EVERY config,
  including the minimal run that ends at loss ~1.5. No failure was established,
  so there is nothing to isolate. Replaced by the real question Q017.
- **Score**: 10/10 — strictly blocking. The minimal HOLO-Net trains identity to
  loss ~1.8; the full model has failed to descend across 3 runs (v1/v2/v3). No
  HOLO-Net falsification result is possible until the full model trains.
- **Background**: v3 identity descended 13→10.6 then rose to 14.1 (step 4550);
  orientation flat. v1/v2 also failed. The patch-and-relaunch loop did not
  converge — switched to isolating components on the known-good minimal base.
- **Discriminating experiment (E043)**: `train.py --enable <components>` —
  diag-A `+ffa`, diag-B `+pc,ofa,gist,magno`. If one group's identity descends
  like the minimal run and the other's does not, the culprit group is isolated;
  then narrow within the group, and test `+orient` separately.
- **Candidate causes**: FFA holistic term injecting noise into the identity
  representation; multi-task aux-loss gradient interference on the shared
  backbone; AdaFace margin warmup (4000 steps) too fast for the slower-learning
  full model (v3's identity blew up exactly when the margin hit full).
- **Linked experiment IDs**: E043 (the ablation), E042 (the v1-v3 runs).

---

## Q017 (NEW, tick 58) — Does the full HOLO-Net descend after the normal ~8000-step identity bounce, like the minimal run?

- **Status**: **ANSWERED — YES (tick 60, E042 v4).** v4's identity loss
  descended 13→8.47 by step 11400, tracking the minimal run almost exactly
  (minimal ≈8.5 at the same step). The full HOLO-Net trains identity at the
  minimal model's rate. The tick-55/56 "failure" calls were premature judgments
  of the normal bounce. — Follow-on issue: v4's OrientationGate stayed inert
  (orientation loss flat at ln2); fixed via `gate.detach()`, being verified by
  v5. See E042 tick-60 addendum.
- **Score**: 10/10 — blocking; this is the actual "does HOLO-Net train" question
  that ticks 51-57 never cleanly answered (every full run was killed mid-bounce).
- **Background**: the minimal HOLO-Net bounces identity 10-13.6 for ~8000 steps
  then descends monotonically to ~1.5. The full model was never run past the
  bounce. v4 = full model (OrientationGate-4×4 + FFA/ATL-additive fixes),
  num_classes 10000, batch 512, 30000 steps, seed 20260521.
- **Discriminating outcome**:
  - SUPPORT (HOLO-Net trains): by step ~15000 identity is clearly < 8 and
    descending (minimal was ~6.3 at step 15000); orientation descends below ln2.
  - REFUTE (genuine failure): identity still bouncing ≥ 10 at step 15000+ while
    the minimal reference is at ~6 — then a real failure IS established and
    component isolation (a corrected E043) becomes warranted.
- **Linked experiment IDs**: E042 (v4).
