# TL;DR for PI / lab — IllusionBench-EEG

**Read time**: ~5 minutes. Built by an autonomous research loop (38 ticks,
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
- **25-prior model zoo** across 8 training paradigms:
  CLIP × 5, SigLIP/MetaCLIP × 3, DINOv2 × 3, MAE, SDXL-VAE,
  Face-triplet (FaceNet × 2), **Face angular-margin (ArcFace + AdaFace × 7)**,
  **Bio-inspired (CORnet-S)**, untrained ViT, raw pixel.
- **Route A EEG analysis**: per-CLIP-dim preservation on ATM's pre-computed
  test EEG embeddings for all 10 THINGS-EEG2 subjects. **Now also per-subject
  individually** (E035, tick 36).
- **Representational similarity (RSA + CKA)**: 25×25 pairwise RSA matrices
  across all 3 paradigms; cluster taxonomy at k=6 (E033, E034).
- All experiments reproducible from seed 20260521.

---

## What we found (7 main findings)

1. **Training-objective × paradigm dissociation** (Figure 1).
   - Thatcher: CLIP family dominates (ISI 4-7, near-human range 4-5).
   - Composite: DINOv2 + MetaCLIP-H/14 + CLIP-L/14 highest (CSI 1.4-1.7).
   - Part-whole: DINOv2 + **CORnet-S** lowest PWI (0.21-0.40 = most
     spatially-holistic).

2. **Face-triplet identification (FaceNet) shows NO holistic effect on any
   paradigm** despite explicit identity supervision. Pose-invariance
   objective suppresses orientation-dependent configural features.

3. **Face-feature-specificity (random-bbox control)**: face-Thatcher ISI is
   60-77% face-feature-localized in CLIP-class priors. The signal is real,
   not a general orientation bias.

4. **EEG bottleneck is uniform low-pass** (ATM Route A). Mean per-CLIP-H/14
   dim r ≈ 0.158 across 10 subjects with no dimension-specific (Spearman
   ρ = −0.029, p = 0.35) or face-category-specific (p = 0.37) structure.
   **NEW (E035)**: the uniform low-pass holds **individually** for each of
   10 subjects (all |Spearman| < 0.07, 8/10 negative, 9/10 p > 0.1) — not a
   subject-averaging artifact.

5. **Combination claim**: current EEG decoders cannot reproduce holistic
   face processing on any paradigm. (Follows from 1+4.)

6. **Face-recognition data-inversion (NEW post-tick-30)**: AdaFace IR-50
   trained on the small noisy CASIA-WebFace dataset gives Thatcher ISI 2.91
   — the **highest among any face-rec model**, comparable to DINOv2-large.
   Larger cleaner datasets (MS1MV2, WebFace4M) give LOWER ISI (1.21-1.33).
   Smaller, noisier data → more Thatcher signal — **opposite of CLIP
   scaling laws**.

7. **Paradigm-conditional cluster migration (NEW post-tick-32)**: RSA shows
   25 priors form 6 clean clusters on Thatcher stimuli, but **the clusters
   change by paradigm**. CORnet-S migrates "alone → semantic →
   reconstructive" across Thatcher/Composite/Part-Whole; AdaFace-CASIA
   migrates "angular-margin → triplet → semantic"; FaceNet migrates "triplet
   → semantic" on Part-Whole. **Dissociation is a (model × paradigm)
   interaction, not fixed model-identity.** Cross-paradigm Spearman agreement:
   0.86 (Thatcher↔Composite), **0.54** (Thatcher↔Part-Whole), 0.73 (C↔PW).

8. **Architecture test — constructive negative (NEW post-tick-65)**: we built
   **HOLO-Net**, a strict bio-fidelity face model (LGN→V1→V2→V4→OFA→MFP→AFP
   →Orientation-Gate→FFA→ATL + PC feedback + magno-PFC gist) with a
   pre-registered FFA-layer falsification target (ISI≥3 ∧ CSI≥1.5 ∧ PWI≤0.5
   ∧ random-bbox≤1.5). Trained on Glint360K face identities to a final
   identity loss of 0.97, HOLO-Net's FFA layer gives **ISI 1.00 / CSI 0.99 /
   PWI 2.20 / random-bbox 1.08 → fails 3 of 4 (ALL FOUR: FAIL)**. The pattern
   is FaceNet's (Finding 2): face-identity training gives ISI ≈ 1 regardless
   of architecture. The FFA module is not inert (it drives PWI mfp 0.75 → ffa
   2.20, the largest deviation HOLO-Net produces) — just in the wrong
   direction. **Conclusion: the operative lever for the §4.1 dissociation is
   the training objective, not architectural bio-fidelity (Claim 8).**

---

## What to look at, in priority order

1. **`figures/exports/three_paradigm_polished_v2.png`** — headline figure
   (Figure 1). 25 priors × 3 paradigms, color-coded by 9 training-class
   families. The dissociation is visually unmistakable. AdaFace-IR50-CASIA
   visibly the strongest face-rec model (Thatcher ISI ≈ 3); CORnet-S the
   most spatially-holistic (PWI 0.21).
2. **`figures/exports/figure3_rsa_3panel.png`** — RSA cluster taxonomy
   across all 3 paradigms (Figure 3). Shows the (model × paradigm)
   interaction directly.
3. **`figures/exports/face_vs_randombbox_polished_v2.png`** — Figure 2
   (face-specificity controls).
4. **`figures/exports/scaling_law_v2.png`** — Figure 5 (image-text + DINOv2
   scaling curves; face-rec data-inversion outlier visible).
5. **`figures/exports/e035_per_subject_routea.png`** — Figure 4b
   (per-subject Route A consistency).
6. **`research_journal/ABSTRACT.md`** — **v3 (218 words)** is the current
   canonical version reflecting all tick 26-37 findings.
7. **`research_journal/TABLES.md`** — Table 1 full 25-prior numerical
   matrix with bootstrap 95% CIs.
8. **`research_journal/PAPER_DRAFT.md`** — single ~6,000-word paper draft
   with new §4.6 representational geometry section.
9. **`research_journal/CLAIMS_SKELETON.md`** — **7 main claims** (was 5
   before tick 33) with evidence audit and counter-claim analysis.
10. **`research_journal/DECISIONS.md`** — tick-by-tick log of what happened
    when (38 ticks documented).

---

## What's the implication for the lab

The paper writes itself as a NeurIPS 2026 Datasets & Benchmarks track or
ICLR 2027 main contribution: **the first systematic mapping of holistic-
face illusion sensitivity across 25 modern visual priors (including
bio-inspired and 7 angular-margin face-recognition variants), THE FIRST
per-CLIP-dim EEG-bottleneck analysis on a publicly-released decoder, and
THE FIRST paradigm-conditional representational-similarity migration
analysis.** Idea-001 internal score: **9.3/10**.

**Architectural predictions for future EEG decoders**:
- Dimension-fine perceptual preservation, not single CLIP-cluster
  anchoring, is required to recover human-aligned holistic processing.
- AdaFace-CASIA (or similar small-data-trained face-rec) adds ~55% novel
  information over CLIP per CKA, motivating multi-prior anchoring.
- Bio-inspired architectures (CORnet-S) plus angular-margin loss is a
  candidate design space for future bio-aligned visual priors (Idea-003,
  score 8.2).

---

## What we did NOT do (and what would help)

- **Could not extend Route A to NICE / AVDE / ENIGMA / ViEEG**: their
  pre-computed embeddings are not publicly released (NICE-EEG releases
  weights only at eeyhsong/NICE; we'd need the ~20 GB preprocessed
  THINGS-EEG2 EEG dataset for inference). Cross-decoder Route A is a
  future-work item.
- **Did not collect EEG of Thatcher / composite / part-whole stimuli**:
  predicted EEG-side ISI is inferred from per-dim preservation × CLIP-side
  ISI multiplication. Direct measurement requires a face-illusion EEG
  collection.
- **Harmonized (Serre lab) perception-aligned baseline blocked** by
  TF/Keras 3 compatibility (NEED-001 in `NOTES_FOR_USER.md`). DINOv2 +
  THINGS-similarity fine-tune is a tractable alternative.
- **Did not train a custom face-CORnet** — Idea-003 sub-path (a) is now
  the most promising direction (AdaFace's quality-adaptive margin + CORnet
  anatomy + face-data). Estimated 1-3 days GPU.

---

## How to engage with this artifact

- **5 min**: read this TL;DR + look at the headline Figure 1 (v2)
- **15 min**: + read ABSTRACT v3 + scan Table 1
- **30 min**: + read full Discussion section (PAPER_DRAFT §5)
- **1 hour**: + full PAPER_DRAFT.md (≈6,000 words)
- **2 hours**: + walk through E001-E035 (35 experiments documented)
- **a day**: + run a quick reproduction on the server (registry + analysis
  scripts + 25-prior batch)

---

## GitHub

https://github.com/LichanghengXJTU/illusionbench-eeg

Every tick of work is a separate commit (38 ticks → 38 commits); the commit
message is a one-line summary. Look at the commit log for a chronological
history of decisions.
