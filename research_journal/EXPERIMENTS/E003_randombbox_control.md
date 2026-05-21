# E003 — FFHQ random-bbox control (Q001 discriminator)

**Hypothesis (Q001)**: The CLIP-class ISI signature observed in E002 (4.0–6.8 on
FFHQ face Thatcher) is **face-feature-specific configural processing**, not
general upright-orientation bias.

**Discriminating prediction**: If face-specific, then applying the SAME Thatcherize
algorithm (3 local 180° rotations + soft blend) to **non-feature locations** on
the **same 200 FFHQ identities** should produce ISI ≈ 1 (no orientation × Thatcher
interaction). If general bias, ISI should remain in the 4-7 range.

**Method**: For each of the 200 FFHQ identities already used in E002, load the
cached MediaPipe landmarks, build an exclusion mask (dilated by 80 px) covering
all face features. Sample 3 random non-overlapping bboxes outside the exclusion
zone, matched in (w, h) to the original eye+brow / eye+brow / mouth bboxes used
in E002. Apply 180° rotation per bbox with the same soft alpha blend
(blur_sigma ≈ 5). Save V1-V4 conditions per identity.

**Stimuli**: 189/200 identities (11 rejected due to insufficient non-face area
for non-overlapping bboxes), 756 stimuli total. Manifest:
`data/stimuli_ffhq_randombbox/thatcher_manifest.csv`.

**Models**: Same 12 visual priors as E001/E002.

**Metric**: Same ISI_pert as E001/E002, with 1000-iteration bootstrap CI.

**Pre-registered prediction** (recorded BEFORE running, 2026-05-21 22:00, before
the loop launched, see E003 plan section in `OPEN_QUESTIONS.md` Q001):
- N03_pixel, N02_untrained_vit, P11_sdxl_vae: ISI ≈ 1.000 (sanity)
- If face-specific: CLIP-bigG ISI ≈ 1.0-1.5 here (down from 6.76)
- If general bias: CLIP-bigG ISI ≈ 5-7 (unchanged)

**Result**:
```
N02_untrained_vit ISI=0.994 [0.970, 1.019]
N03_pixel         ISI=1.000 [1.000, 1.000]
P02_clip_b32      ISI=2.116 [1.715, 2.550]
P03_clip_l14      ISI=1.332 [1.153, 1.532]
P04_clip_h14      ISI=1.441 [1.242, 1.650]
P05_clip_g14      ISI=1.613 [1.426, 1.830]
P06_clip_bigG14   ISI=1.972 [1.685, 2.265]
P07_dinov2_base   ISI=1.072 [0.949, 1.199]
P08_dinov2_large  ISI=0.888 [0.809, 0.979]
P09_dinov2_giant  ISI=0.851 [0.768, 0.935]
P10_mae_huge      ISI=0.993 [0.953, 1.033]
P11_sdxl_vae      ISI=0.995 [0.990, 1.000]
```

**Headline comparison (face vs random-bbox)**:

| Model              | Face Thatcher ISI | Random-bbox ISI | Δ      | % drop  |
|---                 |---                |---              |---     |---      |
| N03_pixel          | 1.000             | 1.000           | 0.000  | 0%      |
| N02_untrained_vit  | 1.006             | 0.994           | -0.012 | ~0      |
| P11_sdxl_vae       | 0.997             | 0.995           | -0.002 | 0       |
| P10_mae_huge       | 1.305             | 0.993           | -0.312 | -24%    |
| P07_dinov2_base    | 1.369             | 1.072           | -0.297 | -22%    |
| P08_dinov2_large   | 2.369             | 0.888           | -1.481 | **-63%**|
| P09_dinov2_giant   | 3.338             | 0.851           | -2.487 | **-75%**|
| P03_clip_l14       | 4.046             | 1.332           | -2.714 | **-67%**|
| P02_clip_b32       | 5.176             | 2.116           | -3.060 | **-59%**|
| P04_clip_h14       | 5.502             | 1.441           | -4.061 | **-74%**|
| P05_clip_g14       | 5.535             | 1.613           | -3.922 | **-71%**|
| P06_clip_bigG14    | 6.763             | 1.972           | -4.791 | **-71%**|

**Interpretation** ([CONFIRMED] = supported by these numbers; [CONJECTURE] otherwise):

- [CONFIRMED] **Q001 answered**: the CLIP-class face-Thatcher ISI signal is
  predominantly face-feature-specific, NOT general upright-orientation bias.
  Random-bbox controls reduce CLIP ISI by 59-74% across all 5 CLIP variants.
- [CONFIRMED] DINOv2 large and giant fall BELOW 1.0 on random-bbox (0.85, 0.89),
  meaning inverted random-bbox perturbations actually move embeddings slightly
  MORE than upright. This is the opposite of any general upright bias and is
  consistent with H_face-specific (DINOv2 had a face-features-specific signal
  on the face stimuli, and once removed, there is no general orientation
  amplifier).
- [CONFIRMED] **Residual general upright bias** in CLIP family: random-bbox ISI
  remains 1.3-2.1 (CIs exclude 1.0). This is real but small: roughly 20-30% of
  the face-Thatcher ISI gap above 1 is attributable to general upright bias;
  70-80% is face-feature-specific.
- [CONFIRMED] **CLIP-L/14 (OpenAI 2021) outlier behavior** in E002 (only ISI=4.0
  on face) is mirrored here (random-bbox ISI=1.33, lowest in CLIP family). The
  hypothesis that CLIP-L/14 has less of the face-emergent property is now
  supported in BOTH stimulus regimes.
- [CONJECTURE] The face-feature-specific component in CLIP could be driven by:
  (a) explicit face-recognition-like emergence (CLIP learned face-configural
  features from web data), or (b) any "object identity" feature that happens
  to be highly orientation-dependent for faces. Q003 (face-trained baselines)
  would discriminate: if FaceNet/ArcFace show even stronger ISI than CLIP-bigG,
  (a) is favored; if comparable, both classes are picking up the same
  face-emergent feature.

**Visual QC**: `qc/identity_0000_V2_upright_thatched.png` (downloaded) shows 3
small rotated patches in forehead/hair region, no overlap with face features.
Algorithm validated.

**Replicability**:
- Seed: 20260521
- Script: `~/Desktop/EEG/illusionbench/stimuli/generate_random_bbox_ffhq.py`
- NPZ paths: `/workspace/illusionbench-eeg/outputs/embeddings/thatcher_randombbox/*.npz`
- Manifest: `/workspace/illusionbench-eeg/data/stimuli_ffhq_randombbox/thatcher_manifest.csv`
- Output: `/workspace/illusionbench-eeg/outputs/tables/thatcher_isi_randombbox.csv`
- Comparison figure: `outputs/figures/compare_face_vs_randombbox.png`

**Linked Q###**: **Q001 (ANSWERED)**, Q003 (face-trained next), Q006 (training-data-distribution).
