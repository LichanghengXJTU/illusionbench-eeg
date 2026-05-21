# E020 — Route A: ATM EEG preservation × Thatcher loading per CLIP-H/14 dimension

**Hypothesis (Q005 Route A)**: If the EEG bottleneck SPECIFICALLY destroys
face-configural information (the Thatcher-loaded dimensions of CLIP), then
Thatcher-loaded dimensions should show LOWER preservation than other dimensions.
Conversely, if Thatcher dimensions are PRIVILEGED, they should be MORE
preserved. NULL: Thatcher-loaded dimensions are no different from others.

**Method**:
- ATM-S (Li et al. 2024) trained pipeline outputs were downloaded from HF
  `LidongYang/EEG_Image_decode/emb_eeg/ATM_S_eeg_features_sub-XX_test.pt`
  for all 10 THINGS-EEG2 subjects. These are the (200 concept, 1024-dim)
  tensors of ATM-decoded EEG embeddings in CLIP-ViT-H/14-LAION space.
- CLIP target: `ViT-H-14_features_test.pt` `img_features` (200, 1024) — exactly
  the CLIP-H/14 LAION encoding of the THINGS-EEG2 test images.
- Per CLIP dimension d ∈ [0, 1024):
  - `preservation[d]` = mean across 10 subjects of Pearson r between
    `clip_target[:, d]` and `atm_eeg_pred[s][:, d]` (across 200 test concepts).
  - `thatcher_loading[d]` = mean(|CLIP-H14(V1_i) − CLIP-H14(V2_i)|) over 200
    FFHQ identities, normalized by σ of CLIP-H14(V1_i) on dim d (taken from
    E002 P04 NPZ).
- Spearman correlation between preservation and loading vectors, with 10,000-perm
  null. Plus high-vs-low loading quartile means.

**Pre-registered prediction** (set 2026-05-22 01:00 BEFORE running):
- Three scenarios:
  1. Thatcher dims preserved (ρ > 0.15 positive): EEG carries the signal
  2. Thatcher dims destroyed (ρ < -0.15 negative): EEG-bottleneck specifically washes face-feature info
  3. Null (|ρ| < 0.10): uniform attenuation

**Result**:
```
preservation:    mean 0.158   median 0.149   min -0.131   max 0.504
thatcher_load:   mean 0.614   median 0.558   min  0.265   max 1.961
Spearman ρ:      -0.029   permutation p_two-sided = 0.351  (10,000 perms)
Pearson r:       -0.030   p = 0.333
High-loading quartile preservation mean: 0.154
Low-loading  quartile preservation mean: 0.165
```

**Interpretation** ([CONFIRMED] / [CONJECTURE]):

- [CONFIRMED — scenario 3]: NULL relationship between Thatcher loading and EEG
  preservation. The EEG bottleneck does NOT specifically destroy face-configural
  CLIP dimensions; instead it attenuates ALL dimensions roughly equally to a
  mean per-dim r ≈ 0.158.

- [CONFIRMED]: EEG preservation in CLIP-H/14 LAION space is LOW (mean r=0.158,
  median 0.149). This is consistent with prior knowledge that EEG visual
  decoding achieves coarse cluster-level retrieval (ATM reports top-1 200-way
  retrieval ~27-30%) but does NOT recover dimension-level CLIP target with
  high fidelity. Per-dim correlation is the right scale of severity.

- [CONJECTURE — extrapolation]: With uniform attenuation factor ≈ 0.158 across
  all 1024 dims, the ISI signature (which lives as a structured signal across
  many dims, magnitude scaled in CLIP space) should be attenuated by roughly
  the same factor. If we could compute ISI on the actual ATM-decoded
  embeddings of FFHQ Thatcher (which we can't — no EEG of Thatcher), we PREDICT
  ISI_atm_eeg ≈ 1.0 + 0.158 × (5.50 − 1.0) ≈ 1.7 (very rough first-order
  estimate; treats preservation as a linear scalar). This means the Thatcher
  signature would be largely lost through the EEG bottleneck.

- [CONJECTURE — IMPLICATION for Idea-001]: Current EEG-to-image decoders use
  CLIP-class anchors that have strong face-Thatcher ISI on the image side, but
  the EEG signal recovery is too noisy to carry that fine perceptual structure.
  The negative result reshapes Idea-001 from "EEG decoders inherit/destroy
  Thatcher" into "**EEG decoders' image-side anchor has the signal, but the
  EEG-bottleneck is a non-specific low-pass that washes out all fine
  perceptual structure including illusions**". This is an honest, falsifiable,
  and immediately publishable finding.

- [KEY UNCERTAINTY]: This analysis uses CLIP-target preservation across the 200
  natural object test concepts of THINGS, NOT across face Thatcher stimuli.
  The natural-image preservation profile may not be identical to the
  hypothetical-face-Thatcher preservation profile (e.g., maybe the directions
  used by ATM are biased toward objects-not-faces). To fully validate Route A,
  we'd need to repeat the preservation analysis on face-containing THINGS
  categories specifically, OR collect EEG of face Thatcher stimuli.

- [LIMITATION]: ATM was trained on THINGS-EEG2 which has ~7% face-related
  categories. The 1024-dim per-dim correlation measures only the dimensions
  most heavily exercised by the training. A more thorough Route A would
  bootstrap dim-correlations restricted to face categories and compare.

**Replicability**:
- ATM ckpts source: HF `LidongYang/EEG_Image_decode` files `ViT-H-14_features_test.pt` + 10× `emb_eeg/ATM_S_eeg_features_sub-XX_test.pt`
- FFHQ CLIP P04: `outputs/embeddings/thatcher_ffhq/P04_clip_h14.npz`
- Code: `analysis/route_a_preservation.py` (seeded 20260521)
- Output: `outputs/tables/route_a/{route_a_arrays.npz, route_a_summary.json}`
- Figure: `outputs/figures/route_a_scatter.{png,pdf}`

**Linked Q###**: Q005 (Route A produces first numerical result; predicts EEG-side ISI collapse).
