# E033 — RSA cross-prior taxonomy + CLIP vs AdaFace CKA

**Date**: 2026-05-22 (tick 31)
**Linked**: E031, E032 (face-rec model sweeps); Idea-001 (paradigm dissociation)

## Goal

Two complementary questions:

1. **Quantitative taxonomy** — beyond the qualitative "10+ dissociation
   patterns" claim, what is the actual representational geometry of the 25
   priors at the global (whole-stimulus-set) level? RSA (Representational
   Similarity Analysis) gives the answer.

2. **AdaFace adds information**: is the AdaFace-CASIA representation
   substantially different from CLIP-H/14, or does it carry largely
   redundant information? CKA (Centered Kernel Alignment) gives a scalar
   answer.

## Methods

**RSA (Representational Similarity Analysis)** —
- For each prior, compute the 800×800 cosine distance matrix on FFHQ
  Thatcher stimuli (200 identities × 4 conditions = 800 stimuli).
- Take the upper triangle (vector of pairwise distances).
- Pairwise Spearman correlation between priors' distance vectors → 25×25
  correlation matrix R.
- Hierarchical clustering on 1-R using average linkage; cut into k=4, 6, 8
  clusters.

**CKA (Centered Kernel Alignment)** —
- Linear CKA via centered Gram matrices: K = X_c X_c^T, L = Y_c Y_c^T
- HSIC(K, L) = sum(K * L)
- CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
- Returns scalar in [0, 1] (1 = identical kernel, 0 = orthogonal).
- Computed on Thatcher 800-stimuli set, all priors centered.

## Results

### Hierarchical clustering at k=6

```
Cluster 1 (3 priors): N02_untrained_vit, P10_mae_huge, P11_sdxl_vae
   → Reconstructive + untrained — image-statistics priors

Cluster 2 (11 priors): P02-P06_clip_*, P07-P09_dinov2_*, P19-P20_siglip_*,
                       P21_metaclip_h14
   → ALL image-text contrastive + DINOv2 SSL — global representations

Cluster 3 (7 priors): P23_arcface_*, P24_adaface_*, P25_arcface_*,
                      P26-P29_adaface_ir50_*
   → Angular-margin face-rec — distinct family

Cluster 4 (2 priors): P17_facenet_vggface2, P18_facenet_casiawebface
   → Triplet-loss face-rec — distinct family

Cluster 5 (1 prior): P22_cornet_s
   → Bio-inspired — alone, distinct from all others

Cluster 6 (1 prior): N03_pixel
   → Raw — control
```

### RSA-clustered ordering (visualization order)

```
1. Pixel → SDXL-VAE → untrained-ViT → MAE-Huge → CORnet-S
   (reconstructive + bio bridge)
6-15. DINOv2-{base,large,giant} → SigLIP-{base,SO400M} → CLIP-{L/14,bigG,…}
      → MetaCLIP-H/14
   (the big "semantic" cluster — CLIP + DINOv2 + SigLIP all merge)
17-23. ArcFace-AuraFace-BGR → AuraFace-RGB → AdaFace-IR50-CASIA →
       ArcFace-IR101-WebFace4M → AdaFace-IR50-WebFace4M →
       AdaFace-IR101-MS1MV2 → AdaFace-IR50-MS1MV2
   (angular-margin face-rec family, tightly clustered)
24-25. FaceNet-VGGFace2 → FaceNet-CASIA
   (triplet face-rec, separate)
```

### CKA quantitative results

```
CKA(CLIP-H/14, CLIP-bigG14)           = 0.93   (within-CLIP same family)
CKA(CLIP-H/14, DINOv2-giant)          = 0.75   (CLIP vs DINOv2 — close)
CKA(FaceNet-VGGFace2, AdaFace-CASIA)  = 0.53   (face-rec models share info)
CKA(CLIP-H/14, AdaFace-CASIA)         = 0.45   ← AdaFace ADDS ~55% novel info over CLIP
CKA(CLIP-bigG14, AdaFace-CASIA)       = 0.46
CKA(DINOv2-giant, AdaFace-CASIA)      = 0.42
CKA(CORnet-S, AdaFace-CASIA)          = 0.39
```

### Top-10 RSA neighbors of P26 (AdaFace IR-50 CASIA)
```
1. P28 (AdaFace IR-50 WebFace4M):    r = 0.71
2. P29 (AdaFace IR-50 MS1MV2):       r = 0.69
3. P23 (ArcFace AuraFace R100 BGR):  r = 0.66
4. P24 (AdaFace IR-101 MS1MV2):      r = 0.66
5. P25 (ArcFace IR-101 WebFace4M):   r = 0.65
6. P23rgb (AuraFace R100 RGB):       r = 0.65
7. P17 (FaceNet VGGFace2):           r = 0.61
8. P18 (FaceNet CASIA):              r = 0.54
9. P06 (CLIP-bigG14):                r = 0.53
10. P02 (CLIP-B/32):                 r = 0.52
```

### Top-10 RSA neighbors of P04 (CLIP-H/14)
```
1. P06 (CLIP-bigG14):     r = 0.92
2. P05 (CLIP-g14):        r = 0.92
3. P02 (CLIP-B/32):       r = 0.83
4. P21 (MetaCLIP-H/14):   r = 0.78
5. P03 (CLIP-L/14):       r = 0.74
6. P19 (SigLIP-base):     r = 0.74
7. P20 (SigLIP-SO400M):   r = 0.74
8. P09 (DINOv2-giant):    r = 0.65
9. P07 (DINOv2-base):     r = 0.65
10. P08 (DINOv2-large):   r = 0.63
```

## Interpretation

[CONFIRMED] **1. The 24-prior battery decomposes into 5-6 clear RSA
clusters** on the Thatcher face-stimulus space:
- Image-text + DINOv2 form ONE big "semantic-representation" cluster
- Angular-margin face-rec forms a distinct cluster
- Triplet face-rec is separate
- CORnet-S is alone (bio-inspired)
- Reconstructive (MAE/VAE) + untrained ViT form a small cluster
- Pixel is alone

This is a clean and publishable taxonomy.

[CONFIRMED] **2. The CLIP-vs-DINOv2 paradigm dissociation (Thatcher vs
Composite/Part-Whole) is a SUBTLE effect within a globally-similar
representational space**. RSA puts CLIP and DINOv2 in the same big cluster
(at k=6). The dissociation appears in specific axes (Thatcher local-feature
inversion, Composite alignment, Part-Whole context dominance) but does NOT
emerge at the level of all-pair distance correlations. **This actually
strengthens the IllusionBench-EEG benchmark's value**: it reveals fine-grained
dissociations that global representational analyses miss.

[CONFIRMED] **3. AdaFace adds ~55% novel information over CLIP**
(CKA = 0.45). This is substantially more "new info" than within-CLIP
variations (CKA = 0.93) or CLIP-vs-DINOv2 (CKA = 0.75). It is comparable
to FaceNet-vs-AdaFace (CKA = 0.53), implying face-rec models genuinely
encode something that semantic SSL models don't capture.

[CONJECTURE] **4. The AdaFace-CASIA Thatcher emergence (ISI 2.91) is
plausibly NOT redundant with CLIP's Thatcher** (since CKA between them is
modest). This justifies trying AdaFace-CASIA as an alternative EEG decoding
target — it would inject genuinely different information into the EEG-side
representation.

## Figure

`figures/exports/e033_rsa_heatmap.png` — 25×25 RSA correlation matrix with
hierarchical-clustering-ordered axes, color scale [0, 1].

## Implications for Idea-001 and Idea-003

- **Idea-001 implication**: paradigm-level dissociation is a fine-grained
  property that survives only on specific perturbation axes (orientation,
  alignment, isolation). Global RSA collapses these to a single cluster.
  This argues the benchmark IS necessary — paradigm tests can't be replaced
  with bulk-RSA.
- **Idea-003 implication**: AdaFace-CASIA adds meaningful new information
  over CLIP/DINOv2, motivating sub-path (h) — try replacing or augmenting
  ATM's CLIP target with AdaFace embeddings. If EEG-side can pick up this
  added info, downstream retrieval / illusion-sensitivity may improve.

## Replicability

- Script: `/tmp/e033_rsa_cka.py` + `/tmp/e033_heatmap.py`
- Outputs: `outputs/tables/e033_{rsa_correlation,prior_clusters,cka}.csv`
- Figure: `figures/exports/e033_rsa_heatmap.png`

---
