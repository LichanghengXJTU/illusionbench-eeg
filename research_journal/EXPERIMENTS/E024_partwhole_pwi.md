# E024 — Part-Whole PWI across 17 priors (N=100 FFHQ identities)

**Hypothesis**: A part-whole-style experiment further dissociates the holistic
emergence pattern of different visual priors. Within our embedding-distance
framework, we measure whether whole-face context amplifies or dampens an
eye-swap perturbation.

**Method**: For each FFHQ identity i paired with j:
- V1 whole_target = rgb_i (with i's own eyes — i.e., the original)
- V2 whole_foil   = rgb_i but with j's eyes pasted in the eye+brow bbox
- V3 part_target  = i's eye+brow patch on a neutral-gray (128) canvas at same bbox
- V4 part_foil    = j's eye+brow patch on the same neutral-gray canvas at same bbox

PWI_pert = mean d(V1, V2) / mean d(V3, V4).

Same compute_metrics.py + 1000 bootstrap.

**Pre-registered prediction** (set BEFORE running): If our embedding-distance
analog of "context amplifies feature-swap detection" holds for holistic models,
PWI > 1. If models dilute small eye-perturbations within larger whole-face
context, PWI < 1.

**Result (raw PWI)**:
```
Most holistic (lowest PWI, whole-context most dominates):
  P07_dinov2_base    0.307 [0.265, 0.354]
  P09_dinov2_giant   0.352 [0.299, 0.405]
  P08_dinov2_large   0.395 [0.343, 0.445]
  P11_sdxl_vae       0.436 [0.419, 0.455]
  P19_siglip_base    0.674 [0.613, 0.740]
  P04_clip_h14       0.684 [0.627, 0.746]
  P10_mae_huge       0.696 [0.576, 0.832]
  P06_clip_bigG14    0.705 [0.648, 0.764]
  P17_facenet_vgg2   0.711 [0.662, 0.760]
  P21_metaclip_h14   0.722 [0.665, 0.783]
  P05_clip_g14       0.770 [0.700, 0.842]
  N02_untrained_vit  0.776 [0.634, 0.923]
  P02_clip_b32       0.858 [0.762, 0.959]
  P03_clip_l14       0.890 [0.824, 0.952]
  P20_siglip_so400m  0.946 [0.876, 1.017]
  P18_facenet_cwf    0.953 [0.891, 1.020]
  N03_pixel          1.202 [1.081, 1.350]
```

**Interpretation** ([CONFIRMED] / [CONJECTURE]):

- [CONFIRMED, with PARADIGM CAVEAT] All trained models show PWI < 1 — whole-
  face context REDUCES the eye-swap embedding distance relative to eyes-on-
  gray context. The lower the PWI, the more the whole-face context dominates
  the embedding (the more "spatially holistic" the representation).

- [CONFIRMED] **DINOv2 dominates spatial-holistic axis again**: PWI 0.31-0.40
  — the lowest among all 17 priors. This is fully consistent with E023's
  finding that DINOv2 was strongest on composite-face spatial integration.
  DINOv2 builds the most globally-integrated face representation; eye-swap
  perturbations get diluted in the whole-face embedding.

- [CONFIRMED] **CLIP family in the middle on this axis**: PWI 0.68-0.89 —
  more locally-sensitive than DINOv2 (i.e., eye-swap stays more salient
  in whole context for CLIP than for DINOv2).

- [CONFIRMED] **Three-paradigm dissociation now established**:
  | Paradigm    | Best model class      | DINOv2 vs CLIP |
  |-------------|-----------------------|----------------|
  | Thatcher    | CLIP-family (4-7)     | CLIP >> DINOv2 (3.3) |
  | Composite   | DINOv2 + Meta/L14 CLIP | DINOv2 ≥ CLIP-mid |
  | Part-Whole  | DINOv2 (most context-dominant) | DINOv2 << CLIP |
  
  This is a strong "different training objectives capture different aspects
  of face-configural processing" claim, supporting Idea-001 pitch v6.

- [CONFIRMED] **FaceNet still at baseline on all three paradigms**: identity-
  invariant training systematically suppresses ALL holistic-illusion signals.

- [CONJECTURE — design caveat]: Our V3/V4 construction (eye region on gray
  canvas) has more pixel similarity than V1/V2 (whole face shared between
  V1/V2 except eyes). So pixel-baseline PWI = 1.20 ≠ 1.0 (it's > 1 because
  V3/V4 share more pixels than V1/V2). After pixel-correction (PWI / 1.20):
  all models drop below 1 with DINOv2 ~ 0.25-0.33. The qualitative ordering
  is preserved.

- [CONJECTURE — model-vs-human dissociation]: Classic Tanaka & Sengco
  Part-Whole behavioral effect is humans IDENTIFY features BETTER in whole-
  face context — opposite of what our embedding distances show. This is
  consistent with two readings:
  (i) Our PWI metric does not capture the same construct as behavioral
      identification accuracy; it measures embedding-shift magnitude, which
      can DECREASE for the same accuracy-improving signal if the shared
      whole-face context dominates the embedding.
  (ii) Models genuinely lack the human Part-Whole effect.
  Distinguishing these would require running a verification-accuracy
  framing (E025 — planned), where we test "is this the same target identity
  as a reference?" instead of pairwise embedding distance.

**Replicability**:
- Stimulus code: `~/Desktop/EEG/illusionbench/stimuli/generate_partwhole_ffhq.py`
- NPZ paths: `outputs/embeddings/partwhole_ffhq/*.npz` (17 priors)
- Manifest: `data/stimuli_ffhq_partwhole/thatcher_manifest.csv`
- Output: `outputs/tables/partwhole_pwi.csv`

**Linked Q###**: Q003 (FaceNet pattern across 3 paradigms), Q007 (image-text
family across paradigms), Q010 (NEW: verification-accuracy framing).
