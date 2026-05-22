# E035 — Per-subject Route A consistency analysis (Claim 4 strengthening)

**Date**: 2026-05-22 (tick 36)
**Linked**: E020 (Route A on ATM EEG), E021 (face subset). Strengthens Claim 4.

## Goal

E020 reported a single mean across-subject per-dim r = 0.158 and Spearman
ρ(r, Thatcher-loading) = -0.029, both NULL on EEG-side Thatcher-preservation.
The CLAIMS_SKELETON identified Claim 4 as the WEAKEST claim because it was
based on a single across-subject summary.

To strengthen: show that the uniform low-pass result holds INDIVIDUALLY for
each of the 10 THINGS-EEG2 subjects, not just in aggregate. This rules out
the possibility that the null is a subject-averaging artifact.

## Method

For each subject 1-10:
1. Load ATM EEG embeddings (200, 1024) and the shared CLIP-H/14 target
   (200, 1024).
2. Compute per-dim Pearson r between EEG and target across 200 test concepts.
3. Compute Spearman(per-dim r, per-dim Thatcher-loading), where
   Thatcher-loading is from CLIP-H/14 embeddings on FFHQ Thatcher stimuli
   (200 identity V1 vs V2 deltas, normalized by per-dim natural-image std).
4. Compute high-quartile-loading mean r vs low-quartile-loading mean r.

Also pool all 10240 (dim, subject) r values into one distribution.

## Result

```
Subject  mean_r   Spearman(r, Thatcher_loading)   p       high-Q r   low-Q r
S01      0.1705   -0.018                           0.571   0.166      0.177
S02      0.1597   -0.028                           0.372   0.155      0.170
S03      0.1720   -0.023                           0.454   0.168      0.182
S04      0.1765   -0.047                           0.134   0.173      0.190
S05      0.1349   -0.051                           0.103   0.127      0.139
S06      0.1335   +0.031                           0.328   0.137      0.134
S07      0.1421   +0.036                           0.243   0.146      0.140
S08      0.2117   -0.028                           0.365   0.210      0.224
S09      0.1259   -0.048                           0.123   0.115      0.133
S10      0.1503   -0.063                           0.044   0.140      0.160

Pooled across 10240 (dim, subject) pairs:
  mean r          = 0.158
  per-subject mean range = [0.126, 0.212], std = 0.026
  Spearman(r, loading) range = [-0.063, +0.037], std = 0.033
  high-Q minus low-Q range = [-0.020, +0.006]
```

## Interpretation

[CONFIRMED] **The uniform low-pass holds for EACH individual subject**:
- All 10 subjects have mean per-dim r between 0.126 and 0.212 — comparable
  magnitudes, no extreme outlier subject
- 8/10 subjects show NEGATIVE Spearman(r, loading) — Thatcher-loaded dims
  are slightly LESS preserved, opposite of what would support
  "EEG-preserves-Thatcher" hypothesis
- 9/10 subjects have Spearman p > 0.1; the lone exception (S10, p=0.044)
  is in the NEGATIVE direction (further refuting Thatcher preservation)
- High-Q minus low-Q quartile difference range is [-0.020, +0.006],
  centered near zero with all 10 subjects close to it

[CONFIRMED] **Claim 4 is robust to inter-subject variability**. The null
result is not a subject-averaging artifact. Each individual subject's EEG
embedding shows the same uniform-low-pass pattern.

[CONFIRMED] **Distribution of per-(dim, subject) r is roughly Gaussian
centered at 0.158** with no obvious bimodality, ruling out "some dims
strongly preserved, others destroyed" explanations.

## Implication for Claim 4

Original Claim 4: "EEG bottleneck imposes uniform per-CLIP-dim attenuation,
not face-specific destruction".

Strengthened: "...AND this uniform low-pass holds INDIVIDUALLY for each of
10 THINGS-EEG2 subjects, with no subject showing positive Thatcher-loading
preservation (Spearman range [-0.063, +0.037], 8/10 negative)."

Claim 4 evidence quality raised from MEDIUM-STRONG to STRONG.

## Figure

`figures/exports/e035_per_subject_routea.png`:
- Left panel: histogram of all 10240 per-(dim, subject) r values, centered at 0.158
- Right panel: scatter of 10 subjects in (mean_r, Spearman_r_loading) plane

## Replicability

- Script: `/tmp/e035_per_subject_routea.py`
- Output: `outputs/tables/e035_per_subject_routea.csv`
- Figure: `figures/exports/e035_per_subject_routea.png`
- Inputs: `data/atm_emb_eeg/ViT-H-14_features_test.pt` +
  `data/atm_emb_eeg/emb_eeg/ATM_S_eeg_features_sub-{01..10}_test.pt` +
  `outputs/embeddings/thatcher_ffhq/P04_clip_h14.npz`

---
