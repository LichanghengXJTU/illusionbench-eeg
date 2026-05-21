# Tables — for paper draft (tick 19)

Three primary tables; can be paste directly into the paper draft.

---

# Table 1 — Three-paradigm matrix (full 17 priors × ISI/CSI/PWI with 95% bootstrap CI)

| # | Model | Class | Thatcher ISI [95% CI] | Composite CSI [95% CI] | Part-Whole PWI [95% CI] |
|---|-------|-------|----------------------|------------------------|-------------------------|
| 1 | CLIP-B/32 | CLIP | 5.176 [4.684, 5.693] | 1.234 [1.190, 1.278] | 0.858 [0.762, 0.959] |
| 2 | CLIP-L/14 (OpenAI) | CLIP | 4.046 [3.681, 4.499] | 1.754 [1.673, 1.845] | 0.890 [0.824, 0.952] |
| 3 | CLIP-H/14 (LAION-2B) | CLIP | 5.502 [5.112, 5.918] | 1.259 [1.221, 1.296] | 0.684 [0.627, 0.746] |
| 4 | CLIP-g/14 (LAION-2B) | CLIP | 5.535 [5.120, 6.017] | 1.254 [1.219, 1.292] | 0.770 [0.700, 0.842] |
| 5 | CLIP-bigG/14 (LAION-2B) | CLIP | 6.763 [6.276, 7.279] | 1.325 [1.278, 1.372] | 0.705 [0.648, 0.764] |
| 6 | SigLIP-base-384 | SigLIP/MetaCLIP | 3.513 [3.216, 3.844] | 1.409 [1.359, 1.463] | 0.674 [0.613, 0.740] |
| 7 | SigLIP-SO400M-384 | SigLIP/MetaCLIP | 5.588 [4.990, 6.247] | 1.568 [1.506, 1.637] | 0.946 [0.876, 1.017] |
| 8 | MetaCLIP-H/14 | SigLIP/MetaCLIP | 6.084 [5.291, 7.062] | 1.719 [1.635, 1.800] | 0.722 [0.665, 0.783] |
| 9 | DINOv2-base | DINOv2 | 1.369 [1.247, 1.505] | 1.346 [1.303, 1.397] | 0.307 [0.265, 0.354] |
| 10 | DINOv2-large | DINOv2 | 2.369 [2.114, 2.645] | 1.547 [1.483, 1.613] | 0.395 [0.343, 0.445] |
| 11 | DINOv2-giant | DINOv2 | 3.338 [2.947, 3.784] | 1.676 [1.603, 1.758] | 0.352 [0.299, 0.405] |
| 12 | MAE-Huge | MAE | 1.305 [1.224, 1.383] | 1.326 [1.236, 1.418] | 0.696 [0.576, 0.832] |
| 13 | SDXL-VAE | VAE | 0.997 [0.990, 1.004] | 1.162 [1.154, 1.171] | 0.436 [0.419, 0.455] |
| 14 | FaceNet (VGGFace2) | Face-trained | 1.117 [1.042, 1.198] | 1.111 [1.059, 1.166] | 0.711 [0.662, 0.760] |
| 15 | FaceNet (CASIA-Webface) | Face-trained | 1.026 [0.974, 1.080] | 1.154 [1.110, 1.201] | 0.953 [0.891, 1.020] |
| 16 | ArcFace (AuraFace, R100, BGR†) | Face-trained (angular-margin) | 1.697 [1.638, 1.766] | 1.161 [1.123, 1.197] | 1.428 [1.335, 1.520] |
| 16b | ArcFace (AuraFace, R100, RGB) | Face-trained (angular-margin) | 1.450 [1.392, 1.511] | 1.170 [1.136, 1.209] | 0.985 [0.940, 1.035] |
| 17 | AdaFace IR-101 MS1MV2 | Face-trained (quality-adaptive AM) | 1.237 [1.177, 1.297] | 1.602 [1.513, 1.703] | 0.590 [0.562, 0.618] |
| 18 | ArcFace IR-101 WebFace4M | Face-trained (angular-margin) | 1.172 [1.113, 1.227] | 1.320 [1.262, 1.389] | 0.501 [0.477, 0.526] |
| 19 | **AdaFace IR-50 CASIA** | Face-trained (quality-adaptive AM, small data) | **2.908 [2.627, 3.219]** | 1.284 [1.232, 1.341] | 0.751 [0.709, 0.798] |
| 20 | AdaFace IR-50 WebFace4M | Face-trained (quality-adaptive AM) | 1.329 [1.261, 1.406] | 1.340 [1.288, 1.402] | 0.516 [0.490, 0.544] |
| 21 | AdaFace IR-50 MS1MV2 | Face-trained (quality-adaptive AM) | 1.207 [1.144, 1.268] | 1.461 [1.398, 1.539] | 0.638 [0.610, 0.665] |
| 22 | CORnet-S (V1-V2-V4-IT) | Bio-inspired | 0.991 [0.903, 1.078] | 1.151 [1.109, 1.192] | 0.214 [0.188, 0.243] |
| 23 | ViT-B/16 (untrained) | Control | 1.006 [0.979, 1.043] | 1.131 [1.089, 1.175] | 0.776 [0.634, 0.923] |
| 24 | Raw pixel | Control | 1.000 [1.000, 1.000] | 1.099 [1.083, 1.116] | 1.202 [1.081, 1.350] |

† BGR preprocessing reflects insightface convention; RGB is the model-correct
ordering and SHOULD be used. P23 with BGR shows a Part-Whole inversion that
disappears with RGB (P23rgb / row 16b) — confirming the inversion was a
preprocessing artifact, not a real model property.

*Pixel-baseline (sanity)*: Thatcher = 1.000 (exact), Composite = 1.099, Part-Whole = 1.202.
*Human Thatcher ISI reference*: Carbon et al. (2005), 2AFC d′ ratio ≈ 4–5.
*N*: Thatcher = 200, Composite = 100 paired, Part-Whole = 100 paired (FFHQ-1024 identities).
*Bootstrap*: 1000 resamples over identities.

---

# Table 2 — Face-feature vs random-bbox controls (Claim 3 robustness)

| Model | Face Thatcher ISI | Random-bbox 0.5× | Random-bbox 1.0× | Random-bbox 1.5× | Δ (face − random 1.0×) |
|---|---|---|---|---|---|
| CLIP-H/14 (LAION-2B) | 5.502 | 1.244 | 1.441 | 1.398 | +4.061 |
| CLIP-bigG/14 (LAION-2B) | 6.763 | 1.634 | 1.972 | 1.467 | +4.791 |
| DINOv2-giant | 3.338 | 1.332 | 0.851 | 0.950 | +2.487 |
| FaceNet (VGGFace2) | 1.117 | 0.163 | 0.382 | 0.411 | +0.735 |
| ArcFace (AuraFace) | 1.697 | — | 0.387 | — | +1.310 |
| CORnet-S | 0.991 | — | 0.938 | — | +0.053 |
| ViT-B/16 (untrained) | 1.006 | 0.995 | 0.994 | 0.995 | +0.012 |
| Raw pixel | 1.000 | 1.001 | 1.000 | 1.000 | +0.000 |

*Bbox-scale robustness*: 0.5×, 1.0×, 1.5× scaling of face-feature bbox sizes
applied to random non-feature locations. CLIP residual general orientation
bias stays bounded between 1.2 and 2.0 regardless of bbox size, well below
face-Thatcher ISI. The ~3-4× face-Thatcher gap is robust to perturbation
magnitude (E025).

---

# Table 3 — EEG-side preservation analysis on ATM (Route A, all 10 THINGS-EEG2 subjects)

| Metric | Value |
|---|---|
| Mean per-dim Pearson r (CLIP target vs ATM EEG, averaged over 10 subjects) | 0.158 |
| Median per-dim Pearson r | 0.149 |
| Spearman ρ (preservation × Thatcher loading per 1024 dims) | -0.029 |
| Permutation p (two-sided, 10000 perms) | 0.351 |
| Pearson r (preservation × Thatcher loading per dim) | -0.030, p = 0.333 |
| High-loading-quartile preservation mean | 0.154 |
| Low-loading-quartile preservation mean | 0.165 |
| Δ (high − low loading quartile) | -0.011 |
| Face-subset preservation mean | 0.150 |
| Random-subset preservation mean | 0.145 |
| Face vs random subset, permutation p | 0.374 |
| N concepts (test set, THINGS-EEG2) | 200 |
| N subjects | 10 |
| N dims (CLIP-H/14 LAION) | 1024 |

*Both per-dimension and per-category contrasts are NULL ⇒ EEG bottleneck is
uniform low-pass*. The Thatcher-loaded dimensions and face-related concepts
receive no privileged preservation through ATM's EEG-to-CLIP projection.
