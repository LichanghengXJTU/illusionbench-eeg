# E001 — LFW funneled Thatcher ISI

**Hypothesis**: Visual priors differ systematically in their orientation × Thatcher
perturbation sensitivity. Pixel/VAE/untrained baselines should give ISI ≈ 1.

**Method**: Generate 200 LFW funneled identities × 4 Thatcher conditions (V1 upright_normal,
V2 upright_thatched, V3 inverted_normal, V4 inverted_thatched). MediaPipe FaceMesh
defines left-eye-region, right-eye-region (eye + brow), and mouth bboxes. Each is rotated
180° in-place with Gaussian-soft boundary.

**Stimuli**: 200 LFW identities, 224×224 resolution. Manifest
`data/stimuli/thatcher_manifest.csv` on server.

**Models**: 12 visual priors — N02_untrained_vit, N03_pixel, P02-P06 CLIP family,
P07-P09 DINOv2 family, P10 MAE-Huge, P11 SDXL VAE. All forward-pass only, L2-normalized.

**Metric**: ISI_pert = mean(d(V1,V2)) / mean(d(V3,V4)) over identities. Cosine distance
in L2-normalized embedding space. Bootstrap 1000× over identity resampling.

**Pre-registered prediction** (set 2026-05-21 22:00 BEFORE running):
- N03_pixel, N02_untrained_vit, P11_sdxl_vae: ISI ≈ 1.000 (sanity)
- P10_mae_huge, P07_dinov2_base: ISI weakly > 1
- P02-P06 CLIP family: ISI > 1, magnitude unknown

**Result** (numbers; no interpretation):
```
N02_untrained_vit ISI=0.997 [0.981, 1.013]
N03_pixel         ISI=1.000 [0.999, 1.000]
P02_clip_b32      ISI=4.073 [3.603, 4.604]
P03_clip_l14      ISI=3.906 [3.542, 4.311]
P04_clip_h14      ISI=3.705 [3.263, 4.156]
P05_clip_g14      ISI=3.507 [3.106, 3.890]
P06_clip_bigG14   ISI=3.473 [3.008, 3.906]
P07_dinov2_base   ISI=1.296 [1.135, 1.473]
P08_dinov2_large  ISI=1.638 [1.359, 1.948]
P09_dinov2_giant  ISI=2.210 [1.869, 2.588]
P10_mae_huge      ISI=0.873 [0.812, 0.931]
P11_sdxl_vae      ISI=1.006 [0.996, 1.016]
```

**Interpretation** ([CONJECTURE] marked):
- [CONFIRMED] Sanity baselines (pixel, VAE, untrained ViT) all = 1.000 ± 0.02 → metric implementation has no trivial bug.
- [CONFIRMED] CLIP family is in the 3.5-4.1 range across all 5 variants — consistent within architecture family.
- [CONFIRMED] DINOv2 shows monotonic scale dependence (base < large < giant).
- [CONJECTURE] User raised concern (2026-05-21) that V1/V2 of LFW stimuli look very similar to human observers, suggesting the LFW resolution (224×224) may not produce canonical Thatcher illusion. Verified visually in contact sheets — many LFW stimuli show subtle Thatcher effect only.

**Replicability**: 
- Seed: 20260521 (numpy + random)
- Stimulus code: `~/Desktop/EEG/illusionbench/stimuli/generate_thatcher.py` (commit will be recorded once git initialized)
- NPZ paths: `/workspace/illusionbench-eeg/outputs/embeddings/thatcher/*.npz`
- Manifest: `/workspace/illusionbench-eeg/data/stimuli/thatcher_manifest.csv`
- Output: `/workspace/illusionbench-eeg/outputs/tables/thatcher_isi_v1.csv`

**Linked Q###**: Q001 (face-specificity), Q006 (training-data-orientation explanation)
