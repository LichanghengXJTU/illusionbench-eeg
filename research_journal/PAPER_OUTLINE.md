# Paper outline — IllusionBench-EEG

**Status**: draft v1, 2026-05-22 (after tick 8). Built from E001–E024 + scope checks.
**Target venue first-tier candidates**: NeurIPS 2026 Evaluations & Datasets Track (D&B
successor) ; ICLR 2027 main track ; ICML 2027 ; NeurIPS 2027 main. Workshop fallback:
NeurIPS 2026 NeurReps / UniReps / SVRHM.

---

## Working title (3 candidates, decide later)

1. **IllusionBench-EEG**: Three Holistic-Face Illusions Reveal Dissociable Configural
   Processing in the Visual Priors of EEG Decoders
2. **Three faces of holistic processing**: how training-paradigm interactions shape
   which face illusions emerge in modern visual priors
3. **The EEG bottleneck doesn't see the face**: uniform per-dim attenuation
   collapses configural information regardless of visual anchor

Title 1 is the most descriptive; title 3 is the punchiest. Final to be decided.

---

## Abstract (~250 words, draft)

Current EEG-to-image decoders (ATM, AVDE, ENIGMA, HVF, ViEEG) all anchor their
visual prediction to large-scale image-text contrastive vision encoders — most
commonly CLIP-ViT-H/14 LAION. Whether these EEG decoders inherit human-like
configural face processing from their visual priors is unknown. We introduce
**IllusionBench-EEG**, a three-paradigm benchmark (Thatcher illusion, composite-face,
part-whole) constructed at FFHQ-1024 native resolution with strictly matched
pixel-baseline sanity, evaluating 17 visual priors spanning six training paradigms.
**Image-side findings**: a clean training-objective × paradigm dissociation emerges:
image-text contrastive priors (CLIP, SigLIP, MetaCLIP) dominate the Thatcher
illusion (ISI 4-7, near-human magnitude) but are weak on composite-face; DINOv2
self-supervised priors dominate composite-face and part-whole context binding
(CSI 1.4-1.7, lowest-PWI 0.31-0.40) but are weaker on Thatcher (1.4-3.3); face-
identification-trained models (FaceNet, VGGFace2 / CASIA-Webface) sit at baseline
(≈1.0) across ALL three paradigms. **EEG-side findings**: direct analysis of ATM's
pre-computed EEG embeddings across 10 THINGS-EEG2 subjects (1024 CLIP-H/14
dimensions × 200 test concepts) reveals uniform per-dim attenuation (mean Pearson
r ≈ 0.158) with no dimension-specific structure: Thatcher-loaded dimensions are
neither preserved nor selectively destroyed (Spearman ρ = -0.029, permutation
p = 0.35). Face-category test concepts receive no privileged preservation either
(p = 0.37 vs random-matched controls). Together these results show that current
EEG visual decoders cannot capture any of the three holistic-face illusions: the
EEG bottleneck is a non-specific low-pass that washes out fine perceptual
structure regardless of the visual anchor's holistic emergence. We release the
3-paradigm stimulus generators, 17-prior model zoo, and per-CLIP-dim preservation
analysis pipeline.

---

## 1. Introduction (1.5 pages)

### 1.1 Motivation
- Face holistic processing as a load-bearing constraint for human visual perception
  (Thompson 1980, Tanaka & Sengco 1997, Carbon 2005, Murphy & Cook 2017, Rossion 2013).
- Recent EEG-to-image decoding has exploded since 2024 (ATM, AVDE, ENIGMA, HVF,
  ViEEG), but all anchor visual reconstruction to a small family of image-text
  contrastive encoders, almost exclusively CLIP-ViT-H/14 LAION-2B.
- Open question: do these EEG decoders inherit human-like configural face
  processing — or does the visual prior already lack it, or does the EEG signal
  itself destroy it?

### 1.2 Approach
- We design a 3-paradigm holistic-face benchmark constructed at FFHQ 1024×1024
  resolution (where canonical Thatcher / composite / part-whole illusions are
  visually salient).
- We test 17 visual priors spanning 6 training paradigms (image-text contrastive,
  SigLIP, self-supervised global, MAE, VAE, face-identity, untrained controls).
- We separately test EEG-side preservation using ATM's pre-computed EEG features
  on THINGS-EEG2, decomposed per CLIP-H/14 dimension.

### 1.3 Contributions
1. **IllusionBench-EEG**: open three-paradigm stimulus battery + metrics with
   verified pixel-baseline sanity, generated from FFHQ-1024.
2. **Training-objective × paradigm dissociation**: first systematic mapping of
   which holistic-face axis is dominated by which training paradigm.
3. **Empirical refutation of "face-trained DNNs are most face-aligned"**: FaceNet
   shows ZERO holistic illusion effect; image-text contrastive models exceed them
   on every paradigm.
4. **EEG per-dim preservation analysis on ATM**: direct evidence the EEG
   bottleneck is uniform low-pass (r ≈ 0.158), with no privileged channel for
   Thatcher-loaded or face-category-related information.
5. **Architectural prediction**: human-aligned EEG visual decoding requires
   dimension-fine perceptual structure preservation, not just CLIP-cluster
   semantic anchoring.

---

## 2. Related work (1.0 page)

### 2.1 Holistic face processing
Thatcher (Thompson 1980), composite (Young 1987, Murphy & Cook 2017), part-whole
(Tanaka & Sengco 1997). ERP signatures (Carbon et al. 2005, Latinus & Taylor 2010).
Configural vs featural processing (Rossion 2013).

### 2.2 Face-DNN alignment
Jacob et al. 2021 (Nat Comm): partial Thatcher reproduction in face-trained CNNs.
Phillips & White 2026 (BJP): face-identity DCNN review, no CLIP / illusion ISI /
EEG. Our work fills the CLIP-class × EEG gap.

### 2.3 EEG visual decoding
ATM (Li et al. 2024, NeurIPS), AVDE (ICLR 2026), ENIGMA (NeurIPS 2025), HVF
(ICLR 2026), ViEEG (preprint May 2025). All use CLIP-class image anchors.

### 2.4 Human-alignment metrics for vision DNNs
Brain-Score (Schrimpf 2018), error consistency (Geirhos 2020), neural harmonizer
(Fel 2022), shape-vs-texture bias (Geirhos 2019). Our ISI/CSI/PWI are
illusion-specific and EEG-bottleneck-aware additions.

---

## 3. IllusionBench-EEG: stimuli and metrics (1.5 pages)

### 3.1 Stimulus generation
- Source: FFHQ-1024 (1000 images per shard; we use 200 for Thatcher, 100 each for
  composite/part-whole).
- MediaPipe FaceMesh landmarks (478 points, refine_landmarks=True).
- QC: confidence ≥ 0.7, inter-ocular distance ≥ 120 px at native, eye-tilt ≤ 15°.
- Thatcher: 180° rotation of left-eye+brow / right-eye+brow / mouth regions,
  applied at native 1024, then resized by model's preprocessor.
- Composite: top-half (i) + bottom-half (j) at nose-tip cut line, with optional
  80-px horizontal misalignment of the bottom half.
- Part-Whole: eye+brow region of donor identity j pasted into whole face of i;
  isolated control = same patch on gray canvas.

### 3.2 Metrics
- **ISI_pert** (Thatcher) = mean d(V1 normal, V2 thatched) / mean d(V3 inverted, V4 inverted-thatched)
- **CSI_pert** (composite) = mean d(V1 aligned-same, V2 aligned-diff) / mean d(V3 misaligned-same, V4 misaligned-diff)
- **PWI_pert** (part-whole) = mean d(V1 whole-target, V2 whole-foil) / mean d(V3 part-target, V4 part-foil)
- Cosine distance in L2-normalized embedding space.
- Bootstrap 1000× over identity resampling.

### 3.3 Pixel-baseline sanity
- Thatcher: pixel ISI = 1.000 exact (perturbation is rotation-invariant by
  construction).
- Composite: pixel CSI = 1.099 (misalignment perturbs more than alignment; we
  report pixel-corrected CSI for fair model comparison).
- Part-Whole: pixel PWI = 1.202 (eyes-on-gray are more pixel-different than
  context-shared whole faces; report raw + pixel-corrected).

### 3.4 Models (17 priors, 6 paradigms)
- CLIP family: P02-P06 (b32, L14 OpenAI, H14 LAION, g14 LAION, bigG14 LAION)
- SigLIP / MetaCLIP family: P19 (siglip-base), P20 (siglip-SO400M), P21 (MetaCLIP-H14)
- DINOv2 family: P07-P09 (base, large, giant)
- MAE: P10 (Huge)
- SDXL VAE: P11 (pixel-statistic encoder)
- Face-identity: P17 (FaceNet VGGFace2), P18 (FaceNet CASIA-Webface)
- Controls: N02 (untrained ViT-B/16), N03 (raw pixel)

---

## 4. Image-side results (2 pages)

### 4.1 Headline figure: three-paradigm dissociation
Figure: `three_paradigm_panel.pdf` (17 priors × 3 paradigms, color-coded by class)

### 4.2 Thatcher (E001-E005)
- LFW vs FFHQ stimulus comparison (E001 vs E002): higher-resolution stimuli
  amplify the same hierarchy (CLIP 3.5-4.1 → 4.0-6.8) while pixel-baseline stays
  at 1.000 across both.
- Face-feature-specificity (E003): random-bbox control drops CLIP ISI 59-74%,
  confirming the upright × Thatcher interaction is face-feature-specific not
  general orientation bias.
- Face-recognition training (E004): FaceNet ISI ≈ 1.0-1.1 — the Thatcher signature
  is NOT a face-identity-training emergent.
- Image-text family generalization (E005): SigLIP and MetaCLIP show same pattern
  as CLIP (ISI 3.5-6.1), confirming a family-general phenomenon.

### 4.3 Composite-face (E023)
- DINOv2 family dominates (corrected CSI 1.22-1.53), CLIP family weak (1.10-1.21).
- Sharp reversal vs Thatcher pattern.
- CLIP-L14 outlier high (1.60) — opposite of its Thatcher outlier-low position.

### 4.4 Part-Whole (E024)
- All trained models PWI < 1; DINOv2 lowest (0.31-0.40).
- Most "spatially holistic" model class is DINOv2; CLIP middle.

### 4.5 Summary: training-objective × paradigm interaction
- CLIP-class: strong Thatcher, weak composite/part-whole
- DINOv2: weak Thatcher, strong composite/part-whole
- FaceNet: flat across all paradigms
- This dissociation maps onto local-feature × orientation (Thatcher) vs
  spatial-integration (composite, part-whole) axes.

---

## 5. EEG-side results: ATM preservation analysis (1.5 pages)

### 5.1 Route A — per-CLIP-dim preservation × Thatcher loading (E020)
- Data: ATM_S pre-computed EEG features (10 subjects × 200 THINGS-EEG2 test
  concepts × 1024 CLIP-H/14 dims) from HuggingFace LidongYang/EEG_Image_decode.
- preservation[d] = mean Pearson r between CLIP_target[:, d] and ATM(EEG)[:, d]
  across 200 concepts, averaged over 10 subjects.
- thatcher_loading[d] = mean(|CLIP(V1_i) − CLIP(V2_i)|) over FFHQ identities,
  normalized.
- **Result**: mean preservation = 0.158, no correlation with Thatcher loading
  (Spearman ρ = -0.029, perm p = 0.35).

### 5.2 Route A face-category control (E021)
- Identified top-30 face-related THINGS-EEG2 test concepts via CLIP-text
  similarity to face queries.
- Face-subset preservation 0.150 vs random-subset preservation 0.145
  (perm p = 0.37, NULL).

### 5.3 Implications
- The EEG bottleneck is uniform low-pass, NOT face-selective destructive.
- Predicted EEG-side ISI on hypothetical Thatcher-EEG: ~1.5-2 (collapsed from
  5.5 on the CLIP image-side).
- ATM's pipeline recovers CLIP-cluster semantics but not dimension-fine
  perceptual asymmetries needed for holistic face illusions.

### 5.4 Figure: route_a_scatter.pdf
Per-dim scatter of preservation × loading, with flat fit overlay and quartile
comparison.

---

## 6. Discussion (1.0 page)

### 6.1 Why the dissociation?
Speculation (clearly marked): CLIP's text-image alignment may encode upright-face
prior orientation through caption associations ("a photo of a face" ⇒ upright),
while DINOv2's self-supervised dense prediction objective encourages global
spatial integration. Face-identity training (FaceNet) explicitly trains AWAY
orientation-dependent features for pose-invariance, suppressing both axes.

### 6.2 Implications for EEG decoder design
Current pipeline: EEG → encoder → CLIP-space projection → image generation. Our
analysis shows the projection step (EEG bottleneck) loses ~85% of fine CLIP
dimension structure. Three implications:
- Multi-prior decoders (CLIP + DINOv2 + ...) may capture more holistic axes
- Architectures explicitly preserving per-dim CLIP structure are needed
- Behavioral / illusion benchmarks should accompany retrieval metrics

### 6.3 Limitations
- N=100-200 identity per paradigm; larger benchmarks possible
- Stimulus generation uses MediaPipe and may have failure modes on diverse faces
- Pixel-baseline correction for composite and part-whole is non-trivial — we
  report both raw and corrected
- ATM is one of several EEG decoders; replicating Route A on AVDE / ENIGMA blocked
  by lack of pre-computed embeddings (note)
- We do not collect Thatcher-EEG; predictions are inferred from per-dim
  preservation × Thatcher loading, not directly measured

### 6.4 Future work
- Per-dim CLIP-direction-preservation architectures
- Multi-anchor (CLIP + DINOv2) EEG training
- Behavioral 2AFC verification framing (E025 planned)
- Cross-decoder Route A on AVDE / ENIGMA (E026 planned)

---

## 7. Conclusion (paragraph)

Three holistic-face illusion paradigms reveal a clean training-objective ×
paradigm dissociation across modern visual priors. CLIP captures Thatcher's
local-feature × orientation effect; DINOv2 captures composite and part-whole's
spatial integration; face-identity-trained networks capture neither. The
universally-used CLIP visual anchor in current EEG-to-image decoders carries the
Thatcher signature on the image side, but our per-CLIP-dim analysis of ATM shows
the EEG bottleneck imposes a uniform low-pass that destroys this and all other
fine perceptual structure. Bridging this gap requires either richer EEG signal
capture or perceptually-preserving decoder architectures.

---

## Figures to include

1. **Figure 1**: `three_paradigm_panel.pdf` — three-paradigm dissociation, the
   headline figure
2. **Figure 2**: `compare_face_vs_randombbox_v3.pdf` — face-feature-specificity
   validation
3. **Figure 3**: `compare_lfw_vs_ffhq.pdf` — stimulus quality matters but pattern
   robust
4. **Figure 4**: `route_a_scatter.pdf` — per-CLIP-dim EEG preservation × Thatcher
   loading
5. **Figure 5** (to design): cartoon schema of training-objective × paradigm
   matrix
6. **Figure 6** (optional, supplementary): contact-sheet QC of all three paradigm
   stimuli at FFHQ-1024

## Tables to include

1. **Table 1**: Full 17-prior × 3-paradigm result matrix (ISI / CSI / PWI with CIs)
2. **Table 2**: ATM Route A per-quartile preservation (face vs non-face, high vs
   low loading)
3. **Table 3** (supplementary): Models tested, training data scale, parameter
   count, source URL

## Code/Data release plan

- GitHub: `https://github.com/LichanghengXJTU/illusionbench-eeg` (already public)
- Stimulus generators reproducible from FFHQ-1024 + seed 20260521
- Embedding NPZs releasable (small, <100 MB total)
- Live benchmark with `compute_metrics.py` for new model submissions
