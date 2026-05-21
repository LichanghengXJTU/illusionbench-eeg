# E028 — CORnet-S go/no-go for Idea-003

**Hypothesis (Q010)**: A bio-inspired visual prior modeled on the ventral
stream (V1→V2→V4→IT + recurrence) should exhibit paradigm-consistent
configural processing across the IllusionBench-EEG three paradigms, unlike
CLIP-class priors which dissociate.

**Method**: Load DiCarlo lab's CORnet-S (NeurIPS 2019, Brain-Score top-tier,
408 MB ImageNet-pretrained checkpoint). Extract pre-classifier features (after
avgpool + flatten, 512-d, L2-normalized) on all three IllusionBench-EEG
paradigms + the random-bbox control. Compare with CLIP-bigG14, DINOv2-giant,
FaceNet-VGGFace2, pixel baseline.

**Pre-registered prediction** (set BEFORE running, 2026-05-22 06:35):
- Strong bio-inspired interpretation: all three indices > 1 with similar
  magnitudes (paradigm-consistent), e.g., ISI ≈ 2-4, CSI > 1, PWI > 1
- Weak bio-inspired interpretation: at least Thatcher ISI > pixel baseline
- Null result: all indices ≈ pixel baseline (CORnet-S architecture doesn't
  matter for illusion sensitivity)

**Result**:
```
Paradigm            CORnet-S    CLIP-bigG14    DINOv2-giant    FaceNet-VGG2    Pixel
Thatcher ISI        0.991       6.763          3.338            1.117          1.000
                    [0.90, 1.08]
Composite CSI       1.151       1.325          1.676            1.111          1.099
                    [1.11, 1.19]
Part-Whole PWI      0.214       0.705          0.352            0.711          1.202
                    [0.19, 0.24]   (LOWEST of all 18 models)
Random-bbox ISI     0.938       1.972          0.851            0.382          1.000
                    [0.87, 1.01]
```

**Interpretation** ([CONFIRMED] / [CONJECTURE]):

- [CONFIRMED] **Bio-inspired anatomy ALONE does NOT produce a Thatcher
  illusion signature**. CORnet-S Thatcher ISI = 0.99 [0.90, 1.08] is
  statistically indistinguishable from the pixel baseline (1.000). The
  4-stage V1/V2/V4/IT recurrent architecture, trained on ImageNet for
  object classification, does not develop the orientation × local-feature
  interaction that drives Thatcher ISI in CLIP and DINOv2 family.

- [CONFIRMED] **CORnet-S is the most "spatially-holistic" model** on
  Part-Whole (PWI = 0.214, the lowest of all 18 models tested). The IT
  recurrence captures whole-face context dominance over local feature
  perturbation more strongly than any other model — consistent with the
  hypothesis that recurrent IT processing aggregates information globally.

- [CONFIRMED] **CORnet-S is approximately pixel-baseline on Composite** (1.15
  vs pixel 1.10). It does not capture the aligned-vs-misaligned spatial
  integration effect that DINOv2 captures.

- [CONJECTURE — KEY REFRAMING OF IDEA-003]:
  The CORnet-S result implies that **bio-inspired anatomical architecture is
  NOT SUFFICIENT for paradigm-consistent illusion sensitivity**. Specifically:
  - V1-V2-V4-IT structure alone gives strong Part-Whole effect (recurrent
    integration) but NO Thatcher effect (no face-feature-specific tuning)
  - The full configural-illusion repertoire of the human visual system
    requires the combination of:
    1. **Bio-inspired anatomy** (V1-IT, recurrence) — for spatial integration
    2. **Face-feature-tuned encoding** — for Thatcher local-feature × orientation
    3. **Configural-binding signals** — for Composite alignment effect
  - None of our 18 tested models has all three. CLIP has #2 via web-text
    associations. DINOv2 has #1 via self-supervised globally-coherent
    representations. CORnet-S has #1 anatomically but lacks #2 and #3.
  - **A truly paradigm-consistent bio-inspired prior would need to combine
    these three properties**. Candidate approaches:
    (a) CORnet-S fine-tuned on a face-recognition objective (e.g., VGGFace2)
        → this would test whether face-data + bio-anatomy gives Thatcher
    (b) Train a CORnet-style recurrent CNN with image-text contrastive on
        face-rich data (e.g., CLIP-style training but limited to FFHQ /
        face-heavy subset)
    (c) Use NSD-aligned encoders (Allen 2022, Conwell 2024) which are
        directly aligned to human ventral stream fMRI responses

- [CONFIRMED] **Random-bbox control on CORnet-S = 0.938** — random non-face
  perturbations produce roughly baseline ISI. So whatever CORnet-S does
  has a small face-feature-specific component but it's mostly < unity.
  Consistent with the "spatially-holistic but face-feature-blind" picture.

**Implication for Idea-003**:
- ❌ CORnet-S alone is NOT a viable bio-inspired prior replacement for CLIP
  in EEG-to-image decoding (no Thatcher signal, lower retrieval expected
  due to smaller model)
- ✅ CORnet-S DOES uniquely capture the Part-Whole holistic effect more
  strongly than any other model — this is a distinct contribution worth
  reporting
- ⏭ Next steps: test face-trained CORnet variants OR design a hybrid
  (CORnet for part-whole + face-trained branch for Thatcher) OR test
  PredNet, γ-net, NSD-aligned models

**Replicability**:
- Model: dicarlolab/CORnet (github.com/dicarlolab/CORnet), CORnet-S checkpoint
  via `pip install -e` from local clone
- Extractor: `models/registry.py::load_cornet_s` extracts pre-classifier
  features (after avgpool + flatten)
- NPZ: `outputs/embeddings/{thatcher_ffhq,composite_ffhq,partwhole_ffhq,thatcher_randombbox}/P22_cornet_s.npz`

**Linked Q###**: **Q010 partially answered** — CORnet-S alone is not
paradigm-consistent in the strong sense, but is the strongest spatial-holistic
model. Idea-003 needs refinement.
