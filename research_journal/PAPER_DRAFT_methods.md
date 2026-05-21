# 3. Methods — draft v1 (tick 13)

**Length target**: ~1.5 pages = ~800 words. Current word count: see end.

---

## 3.1 Stimulus generation

All stimuli are deterministically generated from the Flickr-Faces-HQ (FFHQ)
1024×1024 dataset (Karras et al., 2019) using a fixed seed (20260521). We use
MediaPipe FaceMesh (refine_landmarks=True, 478 landmarks per face,
min_detection_confidence ≥ 0.7) to localize face features. Each identity must
pass three quality filters: (a) all 478 landmarks detected, (b) inter-ocular
distance ≥ 120 px at native 1024 resolution, (c) eye-line tilt from horizontal
≤ 15° (absolute). For the Thatcher paradigm we sample 200 identities; for
composite and part-whole we use 100 paired identities each (deterministic
non-self pairing via seeded shuffle).

**Thatcher.** Following Thompson (1980), we identify three face regions per
identity — left eye+eyebrow, right eye+eyebrow, mouth — using the
FACEMESH_LEFT_EYE/RIGHT_EYE/LEFT_EYEBROW/RIGHT_EYEBROW/LIPS connection sets,
with bounding boxes padded ~20-25 px at native resolution. Each region is
rotated 180° in-plane around its centroid and pasted back with a 5-px Gaussian
soft alpha boundary. Four conditions are saved per identity:
V1 upright_normal (original), V2 upright_thatched (the in-place 180° feature
inversion), V3 inverted_normal (180° global rotation of V1), V4
inverted_thatched (180° global rotation of V2). By construction, V1↔V2 and
V3↔V4 differ by identical local-feature transformations, so pixel-distance is
exactly rotation-invariant: ISI_pixel = 1.000.

**Composite-face.** Following Young et al. (1987) and Murphy & Cook (2017), we
cut each face horizontally at the nose-tip landmark and pair each base
identity i with a non-self distractor j. Four conditions: V1 aligned_same
(top(i) + bottom(i) = original), V2 aligned_diff (top(i) + bottom(j), no
shift), V3 misaligned_same (top(i) + bottom(i), bottom shifted right 80 px),
V4 misaligned_diff (top(i) + bottom(j), shifted 80 px). The seam is hidden by
a 5-px isotropic Gaussian blend along the cut row.

**Part-Whole.** Following Tanaka & Sengco (1997), we extract the combined
left+right eye+eyebrow region of donor identity j and substitute it into
identity i's whole face for the swap condition. Four conditions: V1
whole_target (rgb_i), V2 whole_foil (rgb_i with j's eyes), V3 part_target
(i's eye-region patch alone on a 128-gray canvas at the same bbox), V4
part_foil (j's eye-region patch on the same gray canvas at the same bbox).

## 3.2 Embedding-space sensitivity indices

For each prior M and stimulus pair (V_a, V_b), we compute the cosine distance
d_M(a, b) = 1 − cos⟨M(a), M(b)⟩, where M(x) is the L2-normalized image
embedding for x produced by M. The three per-paradigm indices are computed
identically:

- **Thatcher ISI_pert** = mean_i d_M(V1_i, V2_i) / mean_i d_M(V3_i, V4_i)
- **Composite CSI_pert** = mean_i d_M(V1_i, V2_i) / mean_i d_M(V3_i, V4_i)
- **Part-Whole PWI_pert** = mean_i d_M(V1_i, V2_i) / mean_i d_M(V3_i, V4_i)

Higher values indicate the model's embedding shifts more under the binding-
preserving condition than the binding-disrupting condition. We compute 95%
bootstrap confidence intervals over 1000 resamples of identities with
replacement. Effect sizes between model classes are reported as raw indices;
between-paradigm comparisons are reported as both raw and pixel-baseline-
corrected (dividing by the pixel-distance baseline of the same paradigm).

## 3.3 Pixel-baseline sanity

Thatcher's construction guarantees pixel ISI = 1.000 because V3, V4 are exact
180° rotations of V1, V2 respectively. For composite (pixel CSI = 1.099) and
part-whole (pixel PWI = 1.202), V1↔V2 and V3↔V4 are not pixel-matched (V3-V4
of composite differs from V1-V2 by the horizontal shift of the bottom-half;
V3-V4 of part-whole consists mostly of gray canvas). We therefore report
both raw indices and pixel-corrected indices (index / pixel-baseline) to
isolate the model-representational signal.

## 3.4 Visual prior model zoo

We evaluate 17 visual priors spanning 6 training paradigms:
**(i) image-text contrastive**: CLIP-RN50, CLIP-ViT-B/32 OpenAI, CLIP-ViT-L/14
OpenAI, CLIP-ViT-H/14 LAION-2B, CLIP-ViT-g/14 LAION-2B, CLIP-ViT-bigG/14
LAION-2B; **(ii) other image-text contrastive**: SigLIP-base-patch16-384,
SigLIP-SO400M-patch14-384 (Zhai et al., 2023), MetaCLIP-H/14
(Xu et al., 2024 ICLR); **(iii) DINOv2 self-supervised** (Oquab et al., 2024):
base, large, giant; **(iv) MAE self-supervised pixel-prediction** (He et al.,
2022): ViT-MAE-Huge; **(v) SDXL VAE** (Podell et al., 2023): pixel-statistics
encoder; **(vi) face-identity-trained**: InceptionResnetV1 (facenet-pytorch)
trained on VGGFace2 and CASIA-Webface (Schroff et al., 2015); **(controls)**:
untrained ViT-B/16, raw pixel cosine distance. All image embeddings are L2-
normalized.

## 3.5 EEG-side analysis (Route A)

For the EEG bottleneck analysis we use ATM (Li et al., 2024 NeurIPS) — the
canonical first-generation EEG-to-image decoder, whose pre-computed test-set
EEG embeddings for all 10 THINGS-EEG2 subjects are released publicly on
HuggingFace (`LidongYang/EEG_Image_decode`). For each subject s and each
of the 200 test concepts c, ATM's released `ATM_S_eeg_features_sub-XX_test.pt`
provides f_s(e_s,c) ∈ R^1024, the EEG-decoded image embedding in CLIP-ViT-H/14
LAION space. The corresponding CLIP target g(c) = CLIP(image_c) ∈ R^1024 is
provided as `ViT-H-14_features_test.pt`. We compute per-CLIP-dimension
preservation:

preservation[d] = mean_s [ Pearson(g[:, d], f_s[:, d]) over 200 concepts ]

We compute Thatcher-loading[d] from our FFHQ Thatcher CLIP-H/14 embeddings
(E002 P04 NPZ): for each identity i, Δ_i = CLIP(V1_i) − CLIP(V2_i) ∈ R^1024,
and thatcher_loading[d] = mean_i |Δ_i[d]| / std_i CLIP(V1_i)[d] (per-dimension
normalization by the natural-image scale on that dimension). We then compute
Spearman correlation across all 1024 dimensions between preservation and
thatcher_loading, with a 10,000-permutation null. We further partition the
1024 dimensions into high-loading (top 25%) and low-loading (bottom 25%) and
compare their mean preservation values. As a category-control we use CLIP-
text similarity to ["face", "person", "human", ...] to identify the top-30
face-related THINGS-EEG2 test concepts and recompute preservation on this
subset, comparing to a matched-size random control via permutation.

All statistics use a deterministic seed (20260521).

---

**Word count check**: approximately 880 words. Slightly over target 800 but
covers all five subsections precisely. Can be trimmed during revision.
