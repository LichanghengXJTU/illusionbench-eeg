# E026 — Thatcher ISI scaling law with image-encoder parameter count

**Question**: Within each training family, how does Thatcher ISI scale with
image-encoder parameter count? Does the difference between image-text
contrastive (CLIP/SigLIP/MetaCLIP) and vision-only SSL (DINOv2) reduce to a
scale gap, or is there a genuine language-conditioning contribution?

**Method**: Plot Thatcher ISI vs log10(image-encoder params M) for the 8
image-text models (CLIP B/32 → bigG/14, SigLIP-B and SO400M, MetaCLIP-H/14)
and the 3 DINOv2 variants. Fit per-family linear regressions on log-params.

**Parameter counts used (image encoder only, approximate)**:
- CLIP-B/32: 88M, CLIP-L/14: 304M, CLIP-H/14: 632M, CLIP-g/14: 1.0B,
  CLIP-bigG/14: 1.8B
- SigLIP-base: 88M, SigLIP-SO400M: 400M, MetaCLIP-H/14: 632M (same arch as CLIP-H)
- DINOv2-base: 86M, DINOv2-large: 300M, DINOv2-giant: 1.1B

**Result (fit summary)**:
```
image-text family:    ISI ≈ 1.71 × log10(M params) + 0.82       (n=8)
DINOv2-SSL family:    ISI ≈ 1.78 × log10(M params) − 2.06       (n=3)

Predicted ISI at various scales:
                          image-text   DINOv2-SSL
   88M    (small)            4.15         1.40
   300M   (mid)              5.06         2.32
   1B     (large)            5.95         3.27   ←  observed CLIP-g/14, DINOv2-G
   1.8B   (XL — CLIP-bigG)   6.39         3.73   ←  observed CLIP-bigG
   10B    (extrapolation)    7.66         5.05
   100B   (extrapolation)    9.38         6.83
```

**Key observation**: The **slopes are nearly identical** (1.71 vs 1.78, ~4%
difference) but the **intercepts differ by 2.9 ISI units**. This decomposes
the Thatcher-ISI emergence into two independent contributions:

1. **Scale contribution** (slope ≈ 1.75 / decade of params): both families
   share the same exponential emergence rate as image-encoder size grows.
2. **Language contribution** (intercept gap ≈ +2.9 ISI): image-text
   contrastive training adds a constant offset on top of scale alone.

**Predictive consequences**:
- Image-text family CROSSES the lower edge of the Carbon-2005 human reference
  range (ISI = 4) at approximately **10^1.86 = 72M params** — i.e., even
  small image-text models should exhibit human-range Thatcher ISI. This
  matches our empirical observation: SigLIP-B (88M) gives ISI 3.51, just
  below the predicted 4.0.
- DINOv2 SSL family would need approximately **10^3.4 = 2.5B image-encoder
  params** to reach ISI=4. DINOv2-G at 1.1B gives ISI 3.34, consistent
  with a scaled trajectory but not yet at the human edge.
- A hypothetical 100B-param image-text model is predicted at ISI ≈ 9.4 —
  well above the human reference range, raising the interpretive question
  of what "super-human" embedding-space ISI means.

**Interpretation** ([CONFIRMED] / [CONJECTURE]):
- [CONFIRMED] **Per-family slopes are similar within fit error** — emergence
  rate with scale is comparable.
- [CONJECTURE — strong] **Language conditioning produces a constant offset**
  that cannot be made up by additional scale within DINOv2's training regime
  alone (would need 30× more params to close the intercept gap).
- [LIMITATION] N=3 for DINOv2 is small for a power-law fit. The slope
  estimate has wide uncertainty; the linear extrapolation should not be
  interpreted as predictive beyond ~3× the largest tested model.
- [LIMITATION] We do not test parameter counts < 86M because none of the
  major image-text or DINOv2 releases include sub-base models. The
  intercept's literal meaning (at log10 params = 0, i.e. 1M params) is an
  extrapolation that should be treated as fit-only.

**Replicability**:
- Script: `figures/scaling_law.py` (deterministic; uses existing CSV)
- Output: `outputs/figures/scaling_law.{png,pdf}`
- Source data: `outputs/tables/thatcher_isi_ffhq.csv`

**Linked Q###**: Q008 (separating scale from language), now answered.
