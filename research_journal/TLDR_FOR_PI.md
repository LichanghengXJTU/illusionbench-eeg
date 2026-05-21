# TL;DR for PI / lab — IllusionBench-EEG

**Read time**: ~5 minutes. Built by an autonomous research loop (20 ticks,
2026-05-21 → 22) under HKUST EEG-decoding lab.

---

## What we ask

Current EEG-to-image decoders — ATM (NeurIPS 2024), AVDE (ICLR 2026),
ENIGMA (NeurIPS 2025), HVF (ICLR 2026), ViEEG (2025) — all anchor visual
prediction to large-scale image-text contrastive vision encoders
(predominantly CLIP-ViT-H/14 LAION-2B). **Do these EEG decoders inherit
human-like holistic face processing from their CLIP anchor, and does the EEG
signal preserve that signature through the EEG-to-CLIP projection?**

---

## What we built

- **IllusionBench-EEG**: a 3-paradigm stimulus battery at FFHQ-1024 native
  resolution — Thatcher illusion, composite-face, part-whole — with pixel-
  baseline-verified metrics (Thatcher pixel ISI = 1.000 exact).
- **17-prior model zoo** across 6 training paradigms (CLIP variants, SigLIP,
  MetaCLIP, DINOv2, MAE, VAE, FaceNet identity-trained, controls).
- **Route A EEG analysis**: per-CLIP-dimension preservation on ATM's
  pre-computed test EEG embeddings for all 10 THINGS-EEG2 subjects.
- All experiments reproducible from seed 20260521.

---

## What we found (3 headline findings)

1. **Training-objective × paradigm dissociation** (the headline figure).
   - Thatcher: CLIP family wins (ISI 4-7, in or above the Carbon 2005 human
     range of 4-5). DINOv2 partial. FaceNet flat.
   - Composite: DINOv2 family wins (CSI 1.4-1.7 pixel-corrected). CLIP
     weaker. FaceNet flat.
   - Part-whole: DINOv2 lowest PWI (0.31-0.40 = most spatially-holistic).
     FaceNet flat.

2. **Face-identification training shows NO holistic effect on any paradigm**
   despite explicit identity supervision on millions of pairs. The
   pose-invariance objective actively suppresses the orientation-dependent
   configural features that emerge under other training paradigms.

3. **EEG bottleneck is uniform low-pass** (ATM Route A). Per-CLIP-H/14
   dimension preservation r ≈ 0.158 averaged across 10 subjects, with NO
   dimension-specific structure (Spearman ρ vs Thatcher loading = −0.029,
   p = 0.35) and NO face-category-specific structure (face-subset 0.150 vs
   random-subset 0.145, p = 0.37). The CLIP-side Thatcher signal collapses
   non-specifically through ATM's EEG pipeline; predicted EEG-side ISI ≈ 1.7
   (vs CLIP image-side 5.5).

---

## What to look at, in priority order

1. **`outputs/figures/three_paradigm_polished.png`** — the headline figure.
   17 priors × 3 paradigms, color-coded by training paradigm. The
   dissociation is visually unmistakable.
2. **`research_journal/ABSTRACT.md`** — v2 (190 words) for the quick read.
3. **`research_journal/TABLES.md`** — Table 1 full numerical matrix with CIs.
4. **`research_journal/PAPER_DRAFT.md`** — single ~5,400-word paper draft if
   you want to read the whole thing.
5. **`research_journal/CLAIMS_SKELETON.md`** — 5 main claims with
   evidence audit; useful for sanity-checking interpretation.
6. **`research_journal/DECISIONS.md`** — tick-by-tick log of what happened
   when. Useful as a project diary.

---

## What's the implication for the lab

The paper writes itself as a NeurIPS 2026 Evaluations & Datasets track or
ICLR 2027 workshop / main contribution: **the first systematic mapping of
holistic-face illusion sensitivity across modern visual priors AND the
first per-CLIP-dim EEG-bottleneck analysis on a publicly-released decoder**.
Idea-001 internal score is 8.7/10.

**Architectural prediction for future EEG decoders**: dimension-fine
perceptual preservation (not single CLIP-cluster anchoring) is required to
recover human-aligned holistic processing. Multi-prior anchoring (CLIP +
DINOv2) and explicit dimension-preserving alignment objectives are the
suggested directions.

---

## What we did NOT do (and what would help)

- **Could not extend Route A to AVDE / ENIGMA**: their pre-computed
  embeddings are not publicly released; cross-decoder Route A is a
  future-work item.
- **Did not collect EEG of Thatcher / composite / part-whole stimuli**:
  predicted EEG-side ISI is inferred from per-dim preservation × CLIP-side
  ISI multiplication. Direct measurement requires a face-illusion EEG
  collection.
- **Harmonized (Serre lab) perception-aligned baseline blocked** by
  TF/Keras 3 compatibility (see `NOTES_FOR_USER.md` NEED-001). Suggested
  fallback: DINOv2 + THINGS-similarity fine-tune.

---

## How to engage with this artifact

- **5 min**: read this TL;DR + look at the headline figure
- **15 min**: + read Abstract v2 + scan Table 1
- **30 min**: + read full Discussion section (5.1-5.3)
- **1 hour**: + full PAPER_DRAFT.md
- **2 hours**: + walk through journal experiments E001-E025 to see the
  thought process

---

## GitHub

https://github.com/LichanghengXJTU/illusionbench-eeg

Every tick of work is a separate commit; the commit message is a one-line
summary. Look at the commit log for a chronological history.
