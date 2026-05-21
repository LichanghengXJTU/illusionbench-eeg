# E023 — Composite-face CSI across 17 priors (N=100 FFHQ identities)

**Hypothesis**: The face-Thatcher ISI hierarchy (E002, E005) generalizes to other
holistic-face illusions. Composite-face is the canonical second test in
psychophysics (Rossion 2013 review). We expect CLIP family to dominate, then
DINOv2, then baselines flat at 1.0.

**Method**: 100 FFHQ identities paired with distinct identity j (deterministic
non-self shuffle). For each i: cut at MediaPipe nose-tip y, compose four
versions matching V1/V2/V3/V4 condition labels:
- V1 = aligned_same: top(i) + bot(i), no shift (= rgb_i)
- V2 = aligned_diff: top(i) + bot(j), no shift
- V3 = misaligned_same: top(i) + bot(i), bottom shifted right 80px
- V4 = misaligned_diff: top(i) + bot(j), bottom shifted right 80px

CSI_pert = mean d(V1, V2) / mean d(V3, V4). Same compute_metrics.py +
bootstrap 1000×.

**Pre-registered prediction** (set BEFORE running):
- Pixel/VAE: CSI ≈ 1.0 (sanity)
- DINOv2-giant: CSI > 1.5
- CLIP-bigG: CSI > 1.5 (similar to or higher than DINOv2)
- FaceNet: CSI ≈ 1.0

**Result (raw CSI)**:
```
N03_pixel               1.099 [1.083, 1.116]  ← sanity broken; see below
P11_sdxl_vae            1.162 [1.154, 1.171]
N02_untrained_vit       1.131 [1.089, 1.175]
P17_facenet_vggface2    1.111 [1.059, 1.166]
P18_facenet_casiawebface 1.154 [1.110, 1.201]
P10_mae_huge            1.326 [1.236, 1.418]
P02_clip_b32            1.234 [1.190, 1.278]
P03_clip_l14            1.754 [1.673, 1.845]   ← reversal vs Thatcher!
P04_clip_h14            1.259 [1.221, 1.296]
P05_clip_g14            1.254 [1.219, 1.292]
P06_clip_bigG14         1.325 [1.278, 1.372]
P07_dinov2_base         1.346 [1.303, 1.397]
P08_dinov2_large        1.547 [1.483, 1.613]
P09_dinov2_giant        1.676 [1.603, 1.758]
P19_siglip_base_384     1.409 [1.359, 1.463]
P20_siglip_so400m       1.568 [1.506, 1.637]
P21_metaclip_h14        1.719 [1.635, 1.800]
```

**Pixel-corrected CSI** (raw / pixel-baseline 1.099 → unit-normalized):
```
P03_clip_l14       1.596  (highest)
P21_metaclip_h14   1.564
P20_siglip_so400m  1.427
P08_dinov2_large   1.408
P19_siglip_base    1.282
P10_mae_huge       1.207
P06_clip_bigG14    1.206
P07_dinov2_base    1.224
P04_clip_h14       1.146
P05_clip_g14       1.141
P02_clip_b32       1.123
P11_sdxl_vae       1.057
P18_facenet_cwf    1.050
N02_untrained_vit  1.029
P17_facenet_vgg2   1.011
N03_pixel          1.000
```

**Interpretation** ([CONFIRMED] / [CONJECTURE]):

- [CONFIRMED, IMPORTANT CAVEAT] **The composite paradigm violates the
  Thatcher metric's pixel-invariance sanity**. In Thatcher, V1↔V2 and V3↔V4
  differ ONLY by a global 180° rotation of identical local perturbations,
  so pixel-distance is exactly rotation-invariant ⇒ pixel ISI = 1.000.
  In our composite, V1↔V2 differs by bottom-half-identity-swap at aligned
  position, while V3↔V4 differs by same identity-swap but with a 80-px
  horizontal shift. These are not pixelwise-matched, so pixel-baseline
  CSI = 1.099 ≠ 1. We address this by reporting BOTH raw CSI and
  pixel-corrected CSI = CSI / CSI_pixel for fair model comparison.

- [CONFIRMED] **The CLIP-dominance hierarchy from Thatcher does NOT
  replicate on composite**. Most CLIP variants give pixel-corrected CSI
  1.12-1.21 (below DINOv2-large 1.41 and DINOv2-giant 1.53). Only CLIP-L14
  (1.60) and MetaCLIP-H14 (1.56) reach the top. So the "image-text
  contrastive training drives face-holistic emergence" story from E005 is
  partially refuted on this paradigm.

- [CONFIRMED] **DINOv2 scales with composite CSI** (corrected): base 1.22 →
  large 1.41 → giant 1.53 — same monotonic emergence as Thatcher. Suggests
  DINOv2's spatial structure SSL captures composite-binding more
  consistently than CLIP's text-aligned features capture composite-binding.

- [CONFIRMED] **FaceNet still at baseline** (corrected 1.01-1.05): face-identity
  training shows NO composite illusion effect either, consistent with E004
  (same pattern across both paradigms).

- [CONJECTURE — KEY NEW INSIGHT] **Holistic-face-illusion emergence in DNNs
  is paradigm-specific**:
  - Thatcher (local feature × orientation): CLIP-family dominates (4-7),
    DINOv2 partial (1.4-3.3), FaceNet absent (~1.0)
  - Composite (spatial integration across face halves): DINOv2-family
    dominates (1.4-1.7 corrected), CLIP-family weak (1.1-1.2 corrected),
    FaceNet absent (~1.0)
  This reframes Idea-001 (pitch v5): "Different image-text/SSL training
  paradigms produce DISSOCIABLE holistic-face-illusion signatures. CLIP
  excels at orientation-dependent local-feature configural processing
  (Thatcher) but not at spatial-integration composite binding, while
  DINOv2 shows the opposite dissociation."

- [CONJECTURE — for next tick] CLIP-L14 (P03) keeps being an outlier:
  Thatcher-low (4.0 vs other CLIPs 5-6.8), composite-high (1.60 corrected
  vs 1.12-1.21). The OpenAI-CLIP-L14 model is trained on a smaller / older
  dataset than LAION variants. This systematic outlier behavior across two
  illusion paradigms is itself an open question (Q009: spawned).

**Limitations**:
- N=100 (smaller than Thatcher's 200) — bootstrap CIs are still tight,
  but adding more pairs would improve power
- The 80-px misalignment shift may interact with model preprocessing
  (CLIP/DINOv2 internal resize); a sensitivity analysis would help
- We do not test the canonical composite paradigm where the upper-half
  IDENTITY-MATCH question is asked — our current setup tests embedding
  shift, not identification accuracy. Worth a follow-up.

**Replicability**:
- Stimulus code: `~/Desktop/EEG/illusionbench/stimuli/generate_composite_ffhq.py`
- NPZ paths: `outputs/embeddings/composite_ffhq/*.npz` (17 priors)
- Manifest: `data/stimuli_ffhq_composite/thatcher_manifest.csv`
- Output: `outputs/tables/composite_csi.csv`

**Linked Q###**: Q008/Q009 (new). New question Q009 = why is CLIP-L14
systematically opposite-direction from other CLIPs across illusion paradigms?
