# Paper outline v2 — IllusionBench-EEG (post-tick-85 consolidation)

**Status**: outline v2, drafted 2026-05-23 tick 86. Supersedes PAPER_OUTLINE.md
(v1 from tick 8, image-side + ATM Route A only). v2 integrates: HOLO-Net
architectural negatives (E044, E045) + Stage 3 EEG decoder (E046) + IllusionBench
transfer (E047) + MLP sensitivity (E048) = three-layer evidence stack.

**Target venues (re-evaluated post-headline)**:
- Primary: NeurIPS 2026 D&B Track (the benchmark is the centerpiece) or
  ICLR 2027 main (full results + sub-findings).
- Strong secondary: TMLR (deep reviewer engagement; no rebuttal pressure).
- Companion short for the PWI non-linearity sub-finding (E048): NeurIPS workshop
  (e.g. SVRHM 2026, UniReps 2026) or a 4-pager letter.

---

## Working title (decide later; v2 candidates)

1. **IllusionBench-EEG**: A Three-Paradigm Benchmark for Holistic-Face Decoder Alignment, Validated Across 25 Visual Priors, 3 Bio-Inspired Architectures, and a SOTA EEG Decoder
2. **The Benchmark That Refuses to Be Passed**: Why no current visual prior, no bio-inspired model, and no current EEG decoder produces human-aligned holistic-face processing
3. **A Sharp Test for EEG Visual Decoders**: Three Holistic-Face Illusions Reveal Decoder-Family-Invariant Bottlenecks in ATM-style Pipelines

Title 1 = descriptive (best for D&B reviewers). Title 2 = punchy. Title 3 =
ML-pragmatic (best for NeurIPS main). Final TBD.

---

## Abstract (v2 draft, ~270 words)

Current EEG-to-image decoders (ATM, AVDE, ENIGMA, HVF, ViEEG) all anchor visual
prediction to large-scale image-text contrastive encoders (CLIP-H/14 LAION).
Whether the resulting decoder representations capture human-like holistic-face
processing is an open question that has been hampered by the absence of a
discriminating benchmark. We introduce **IllusionBench-EEG**, a three-paradigm
benchmark (Thatcher illusion, composite-face, part-whole) constructed at FFHQ
1024² with strict pixel-baseline sanity and four pre-registered §6 falsification
criteria (ISI ≥ 3.0, CSI ≥ 1.5, PWI ≤ 0.5, random-bbox-control ≤ 1.5). We use
the benchmark to systematically interrogate **25 visual priors** (image-text
contrastive, SSL, MAE, VAE, face-identity, untrained controls), **3 bio-inspired
architectures** (HOLO-Net v1 CORnet + AdaFace identity; v3 frozen DINOv2 +
global FTPC face template; v2 from-scratch DINO SSL × 3 collapses), and an
end-to-end **Stage-3 EEG decoder** (per-subject ridge + MLP, ATM-EEG to frozen
DINOv2 ViT-S/14, applied transfer to the IllusionBench stimuli via per-dim r_d
filter). Four contributions: (1) the open benchmark + metrics + pixel sanity
+ §6 criteria; (2) a clean training-objective × paradigm dissociation across
25 priors; (3) three bio-inspired architectures whose §6 PASS profile saturates
at 2/4 — bio-fidelity is not enough; (4) **the benchmark's pre-registered
sharp prediction is vindicated at subject-level data**: even the best EEG
decoder + best target combination (ridge / MLP × DINOv2) achieves 0/10 subjects
4/4 PASS on the FFA layer. Sub-finding: ISI and CSI are bottlenecked decoder-
family-invariantly (linear and non-linear decoders fail identically), while PWI
has non-linear EEG structure that MLPs partially recover.

---

## Contributions (re-formulated)

1. **IllusionBench-EEG benchmark + metrics**: open three-paradigm stimulus
   battery (Thatcher / Composite / Part-Whole) + 4 pre-registered §6 criteria
   + pixel-baseline sanity correction, ready for any vision or EEG decoder.
2. **Image-side dissociation across 25 priors**: clean training-objective ×
   paradigm interaction. CLIP family dominates Thatcher (ISI 4-7) but is weak
   on composite / PW; DINOv2 family dominates composite + PW (CSI 1.4-1.7,
   PWI 0.31-0.40) but weak on Thatcher; face-identity training (FaceNet,
   ArcFace, AdaFace IR-50) is flat across all 3 paradigms.
3. **Bio-inspired architecture is not enough**: three independently-instantiated
   HOLO-Net variants — v1 (CORnet + AdaFace identity), v2 (CORnet + DINO SSL
   ×3 collapse), v3 (frozen DINOv2 + global FTPC) — all fail §6 4/4. v3
   achieves the best result (2/4 PASS at the raw feature layer; PWI flipped
   from v1's 2.20 FAIL to 0.446 PASS) but Thatcher stays anti-direction. The
   FTPC template residual adds no discriminative power over raw DINOv2.
4. **Stage-3 EEG decoder built and IllusionBench-transferred**: per-subject
   ridge ATM-EEG → frozen DINOv2 gives top-1 18.9% ± 3.8% on the 200-way
   THINGS-EEG2 test set (38× chance, in the pre-registered range);
   DINOv2 target outperforms CLIP-H/14 target by 63% relative under same ridge
   (a new finding refining E020). IllusionBench transfer: **0/10 subjects pass
   §6 4/4 (the headline)**; PWI flips from raw DINOv2 PASS (0.397) to ridge
   FAIL (0.537); ISIrbox control preserved for all subjects. Clean dose-response:
   the 3 subjects who DO preserve PWI have the highest per-dim r_d means.
5. **PWI has non-linear EEG structure (sub-finding)**: MLP decoder partially
   recovers PWI (7/10 subjects PASS vs ridge's 3/10) while ISI/CSI stay
   decoder-family-invariantly bottlenecked. Cognitive-science implication:
   part-whole binding has a non-linear EEG signature that the Thatcher and
   composite signals do not.

---

## Section structure (estimated lengths)

### 1. Introduction (1.5 pages)
- 1.1 Motivation: holistic face processing as a load-bearing constraint
- 1.2 Gap: current EEG decoders not benchmarked on this; current bio-inspired
  models not benchmarked either; need a sharp test
- 1.3 Contributions (the 5 above)
- 1.4 Roadmap

### 2. Related work (1.0 page)
- 2.1 Holistic face processing (Thompson 1980, Tanaka 1997, Carbon 2005,
  Murphy & Cook 2017, Rossion 2013, Psalta 2014)
- 2.2 Face-DNN alignment (Jacob 2021, Phillips & White 2026)
- 2.3 EEG visual decoding (ATM Li 2024, AVDE, ENIGMA, HVF, ViEEG)
- 2.4 Bio-inspired vision models (CORnet Kubilius 2019, NSD-aligned Conwell 2024,
  PredNet, harmonized networks)
- 2.5 Human-alignment metrics (Brain-Score, error consistency, neural harmonizer,
  shape-vs-texture). Our §6 criteria are illusion-specific + EEG-bottleneck-aware.

### 3. IllusionBench-EEG: stimuli, metrics, §6 criteria (1.5 pages)
- 3.1 Stimulus generation (FFHQ 1024, MediaPipe FaceMesh, QC + canonical sanity)
- 3.2 Metrics (ISI / CSI / PWI / ISIrbox; pixel-corrected with verified baselines)
- 3.3 §6 falsification criteria (ISI ≥ 3.0, CSI ≥ 1.5, PWI ≤ 0.5, random-bbox ≤ 1.5;
  pre-registered before any model evaluation)
- 3.4 Battery: 25 priors × 6 paradigms (image-text contrastive, SSL, MAE, VAE,
  face-identity, controls)
- 3.5 Models, training data, parameter counts (Supplementary Table 3)

### 4. Image-side dissociation across 25 priors (2.0 pages)
- 4.1 Headline figure: 3-paradigm panel (color-coded by training paradigm)
- 4.2 Thatcher: CLIP family dominates (E001-E005)
- 4.3 Composite: DINOv2 family dominates (E023)
- 4.4 Part-Whole: DINOv2 family dominates (E024)
- 4.5 Face-identity training is flat (E004 + E030/E032 AdaFace/ArcFace add-on)
- 4.6 §6 verdict: **no current prior passes 4/4** (the benchmark's sharpness)

### 5. Bio-inspired architecture-side: HOLO-Net (1.5 pages)
- 5.1 Motivation: bio-inspired ≠ trained-on-faces; could fill the dissociation
  gap by architectural inductive bias
- 5.2 HOLO-Net v1 (E040–E044): CORnet-S anatomy + AdaFace identity loss +
  Glint360K data → 25M params trained 30K steps + Stage-2 fine-tune.
  §6 verdict: **1/4 PASS** (ISIrbox only). Diagnosis: identity training
  drives the architecture to FaceNet-like ISI ≈ 1.0 (matches E004).
- 5.3 HOLO-Net v2 (the 3 collapses): from-scratch DINO SSL on CORnet-S
  collapses to uniform regardless of out_dim or PC-loss design.
  Lesson: SSL collapse risk is fundamental on small bio-architectures.
- 5.4 HOLO-Net v3 (E045): frozen pre-trained DINOv2 ViT-S/14 + FTPC face
  template head. §6 verdict at FFA: **2/4 PASS** — PWI 0.446 PASS (v1 was
  2.20 FAIL), ISIrbox 0.654 PASS, ISI 0.901 FAIL (anti-direction), CSI
  1.300 FAIL (close to threshold). Mechanistic finding: δ_pooled
  (template residual) ≈ raw DINOv2 CLS in CI overlap — the 2/4 pass is
  DINOv2-backbone-driven, not FTPC-driven. Diagnosis: global template
  cannot encode local feature-orientation (the Psalta 2014 Thatcher locus).
- 5.5 Architectural fix proposal (deferred / future work): K=4 part-aware
  sub-templates at fixed sub-regions of the patch grid; per-region δ
  aggregation. v2.2 design locked in (image-side complement).
- 5.6 §6 verdict across 3 architectures: **bio-inspired is not enough**.

### 6. EEG-side: Stage-3 decoder + IllusionBench transfer (2.0 pages)
- 6.1 Stage-3 pipeline (E046): frozen DINOv2 target, ATM EEG features (1024-d,
  CLIP-H/14-aligned by ATM training) as source; per-subject ridge
  (closed-form, 5-fold λ-CV) and per-subject MLP (1024→512→384 GELU+dropout,
  cosine loss, early-stop). Why DINOv2 target: E045 showed DINOv2 raw FFA
  gets 2/4 §6 PASS — strong substrate.
- 6.2 Retrieval result: per-subject top-1 ridge 18.9% ± 3.8%, MLP 18.1% ± 3.9%
  on the 200-way test (chance 0.5%, in pre-registered 18-32%). **Surprising**:
  same-pipeline CLIP-H/14 target top-1 = 11.6% ± 3.7% → DINOv2 outperforms
  CLIP by 63% relative under linear decoding. Refines E020 "uniform low-pass"
  thesis as target-specific.
- 6.3 IllusionBench transfer (E047): per-subject r_d filter applied to
  IllusionBench DINOv2 FFA features; recompute_metrics. Aggregate §6 verdict:
  - Ridge: ISI 0.811 FAIL, CSI 1.275 FAIL, PWI 0.537 FAIL, ISIrbox 0.603 PASS
    = 1/4 PASS
  - MLP: ISI 0.814 FAIL, CSI 1.273 FAIL, **PWI 0.479 PASS**, ISIrbox 0.603 PASS
    = 2/4 PASS
  - **0/10 subjects achieve 4/4 PASS** with either decoder family
  - 7/10 subjects pass PWI with MLP (vs 3/10 ridge)
- 6.4 Dose-response: per-subject r_d mean ↔ PWI PASS. Top-3 r_d subjects
  pass PWI under ridge; their MLP performance is even cleaner.
- 6.5 §6 verdict: **vindicated at subject-level data**. The benchmark's
  pre-registered sharp prediction holds across all decoder families we've
  tested. ISIrbox (control) preserved → not just a uniform noise artifact;
  the failure is specifically of holistic-face signal preservation.

### 7. Sub-finding: PWI non-linearity (0.5 page or short companion paper)
- 7.1 The MLP-vs-ridge contrast on PWI: 7/10 vs 3/10 subjects PASS
- 7.2 ISI and CSI invariant to decoder family
- 7.3 Cognitive interpretation: part-whole binding has non-linear EEG signature

### 8. Discussion (1.0 page)
- 8.1 The training-objective × paradigm dissociation explained
- 8.2 Why current EEG decoders are short of holistic-face alignment
- 8.3 Three orthogonal improvement axes (sensor, encoder, target)
- 8.4 The IllusionBench-EEG sharp-test paradigm as a model for future
  human-alignment benchmarks
- 8.5 Limitations:
  - We do not collect Thatcher-EEG (transfer simulation via per-dim r_d
    is best available; raw EEG of illusion stimuli is the next experimental
    step)
  - HOLO-Net is one bio-inspired architectural family; doesn't refute all
    bio-inspired approaches
  - Stage-3 uses ATM's frozen EEG encoder; Stage-4 end-to-end training
    against DINOv2 is the natural follow-up
  - 200-way test retrieval is the standard but small; THINGS-EEG2 doesn't
    permit larger retrieval set
  - All 10 subjects are from same THINGS-EEG2 collection — single-population

### 9. Future work (0.5 page)
- Stage-4: re-train ATM-equivalent encoder against DINOv2 target
- HOLO-Net v2.2: part-aware FTPC (image-side complement)
- Collect actual face-illusion EEG (200-trial subset, 32-ch headset)
- Cross-population validation
- ViT-Large or H DINOv2 backbone for higher-capacity target

### 10. Conclusion (paragraph)
Three holistic-face illusion paradigms + pre-registered §6 criteria reveal
that NO current visual prior (25 tested), NO current bio-inspired architecture
(3 tested), and NO current EEG decoder + target combination (linear or non-
linear, CLIP or DINOv2 target) delivers the human-aligned configural face
processing profile. The benchmark is sharp at subject-level data: 0/10
THINGS-EEG2 subjects pass 4/4 §6 with our best Stage-3 pipeline. ISI and
CSI are decoder-family-invariantly bottlenecked; PWI has non-linear EEG
structure recoverable by MLP for most subjects. We release the benchmark,
all 25 prior embeddings, the HOLO-Net checkpoints, and the Stage-3
decoder pipeline for future research.

---

## Figures inventory (updated for v2)

1. **F1** — 3-paradigm panel, 25 priors color-coded by training paradigm
   (the image-side headline; reuse `three_paradigm_panel.pdf` and extend)
2. **F2** — pixel-baseline sanity bar chart (verifies QC discipline)
3. **F3** — HOLO-Net per-layer verdict (v1/v2-collapse/v3 stacked)
4. **F4** — Stage-3 decoder schematic (ATM-EEG → ridge/MLP → DINOv2)
5. **F5** — IllusionBench transfer per-subject verdict heatmap
   (subjects × paradigms, color-coded by PASS/FAIL distance to threshold)
6. **F6** — PWI dose-response (per-subject r_d_mean vs PWI value, ridge vs
   MLP comparison)
7. **F7** (supp) — ridge vs MLP retrieval per-subject distribution
8. **F8** (supp) — pixel-baseline correction table sketch

## Tables inventory

1. **T1** — Full 25-prior × 4-criterion §6 result matrix (with CIs)
2. **T2** — HOLO-Net v1 / v2 / v3 §6 verdict comparison
3. **T3** — Stage-3 retrieval per-subject (top-1/5/10 + per-dim r_d, ridge and MLP)
4. **T4** — IllusionBench transfer per-subject + aggregate, ridge and MLP
5. **T5 (supp)** — Model training data scale, parameter count, source URL
6. **T6 (supp)** — Ridge λ-CV grid + chosen value per subject

---

## Risks / decisions for the user

**D1**: Single-paper or two papers (benchmark D&B + decoder methods)?
  - One paper: full story in one venue. Pros: complete narrative, cumulative
    contributions, easier publicity. Cons: long; the 5 contributions stretch
    the single-paper page budget.
  - Two papers: companion benchmark D&B + methods paper. Pros: each piece
    cleanly scoped. Cons: two writing cycles + duplicate intro/related work.
  - **My default recommendation**: single paper at NeurIPS D&B with full
    methods in main + extensive Supplementary. The negative-result methodology
    + 25-prior + 3-architecture + decoder transfer is a complete picture
    that's stronger together than apart.

**D2**: Companion paper on PWI non-linearity (E048 sub-finding)?
  - Option A: include §7 as sub-finding in main paper (default).
  - Option B: split out as 4-page short companion (SVRHM 2026 / UniReps).
  - Decision after main paper is drafted.

**D3**: HOLO-Net v2.2 part-aware FTPC — run before submission?
  - Run: adds positive evidence on architecture-side. Cost: ~1-2 ticks.
  - Skip: §5 already complete with 2/4 best-case + clean diagnostic.
  - **My default recommendation**: run it, opportunity-cost is small.
    Image-side complement to the EEG-side headline.

**D4**: Stage 4 (re-train ATM against DINOv2) before submission?
  - Run: removes the "ATM is CLIP-aligned by training" confound.
  - Skip: paper has the per-subject MLP-vs-ridge sensitivity check (E048)
    showing the failure is decoder-family-invariant.
  - **My default recommendation**: skip for submission; cite as next-step
    future work (would be Stage-4 in a follow-up paper).

---

## Drafting checklist (next steps post-this-outline)

1. [user decision] D1, D2, D3, D4 above
2. [next-tick] Re-pull all E### result tables into a consolidated
   `results_for_paper.md` (single source for tables T1-T6)
3. [next-tick] Draft §3 (benchmark construction) — mostly carry-over from
   PAPER_DRAFT_methods.md, add §6 criteria
4. [next-tick] Draft §5 (HOLO-Net) — fresh write-up integrating E044, E045,
   v2-collapse runs
5. [next-tick] Draft §6 (EEG decoder + transfer) — fresh, this is the
   headline section
6. [next-tick] Draft §1 (intro) — update PAPER_DRAFT_intro.md with all 5
   contributions
7. [parallel] Generate F3, F5, F6, F7 figures (we have all the data on
   server; matplotlib scripts ~30 min each)
