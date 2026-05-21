# E034 — RSA cross-prior taxonomy PER PARADIGM (Composite + Part-Whole)

**Date**: 2026-05-22 (tick 32)
**Linked**: E033 (Thatcher RSA + CKA); extends to Composite + Part-Whole.

## Goal

E033 showed 6 clean RSA clusters on Thatcher 800-stimulus set, with all
CLIP+DINOv2+SigLIP merging into one big "semantic" cluster. The natural
follow-up question: **does the cluster structure change by paradigm?**
If YES → dissociation is a (model × paradigm) interaction, not a fixed
model-identity property.

## Methods

Same RSA procedure as E033, but on:
- Composite stimuli (400 = 100 paired × 4 conditions: V1, V2 with aligned/misaligned)
- Part-Whole stimuli (400 = 100 identities × 4 conditions: isolated/whole × normal/perturbed)

Hierarchical clustering at k=6 (average linkage on 1-R). Also compute
cross-paradigm Spearman correlation between the upper-triangle vectors
of the three paradigm RSA matrices.

## Results

### Composite RSA — clusters at k=6

```
Cluster 1 (3 priors): N02_untrained_vit, P10_mae_huge, P11_sdxl_vae
   → Reconstructive (same as Thatcher)

Cluster 2 (12 priors): P02-P06_clip_*, P07-P09_dinov2_*, P19-P20_siglip_*,
                       P21_metaclip_h14, P22_cornet_s
   → SEMANTIC (CLIP + DINOv2 + SigLIP) + CORnet-S MIGRATES IN

Cluster 3 (4 priors): P24_adaface_ir101_ms1mv2, P25_arcface_ir101_webface4m,
                      P28_adaface_ir50_webface4m, P29_adaface_ir50_ms1mv2
   → AdaFace/ArcFace IR-101 + IR-50 with cleaner data
     (P26 CASIA is NOT here)

Cluster 4 (2 priors): P23_arcface_auraface, P23rgb_arcface_auraface_rgb
   → AuraFace BGR + RGB (its own little cluster)

Cluster 5 (3 priors): P17_facenet_vggface2, P18_facenet_casiawebface,
                       P26_adaface_ir50_casia
   → Triplet face-rec + AdaFace IR-50 CASIA MIGRATES IN

Cluster 6 (1 prior): N03_pixel
```

**Migrations vs Thatcher**:
- CORnet-S: from "alone" (Thatcher) → "semantic cluster" (Composite)
- AdaFace-CASIA: from "angular-margin face cluster" (Thatcher) → "triplet
  face cluster" (Composite)
- AuraFace BGR/RGB: split off into their own pair (Composite)

### Part-Whole RSA — clusters at k=6

```
Cluster 1 (3 priors): P24_adaface_ir101_ms1mv2, P28_adaface_ir50_webface4m,
                       P29_adaface_ir50_ms1mv2
   → AdaFace IR-50/IR-101 cleaner data only

Cluster 2 (1 prior): P25_arcface_ir101_webface4m
   → ArcFace IR-101 ALONE

Cluster 3 (3 priors): N02_untrained_vit, P10_mae_huge, P22_cornet_s
   → Reconstructive + CORnet-S MIGRATES IN

Cluster 4 (16 priors): P02-P06_clip_*, P07-P09_dinov2_*, P17_facenet_vggface2,
                       P18_facenet_casiawebface, P19_siglip_base_384,
                       P20_siglip_so400m, P21_metaclip_h14,
                       P23_arcface_auraface, P23rgb_arcface_auraface_rgb,
                       P26_adaface_ir50_casia
   → THE BIG MIXED CLUSTER: CLIP + DINOv2 + SigLIP + FaceNet
     + AuraFace + AdaFace-CASIA all merge

Cluster 5 (1 prior): P11_sdxl_vae

Cluster 6 (1 prior): N03_pixel
```

**Migrations vs Thatcher**:
- CORnet-S: from "alone" → "reconstructive cluster"
- FaceNet ×2: from "triplet cluster" → "semantic cluster" (merged with CLIP/DINOv2)
- AdaFace-CASIA: → "semantic cluster" (further migration from Composite)
- AuraFace BGR/RGB: → "semantic cluster" (further migration)
- ArcFace IR-101 WebFace4M: → ALONE (split out)

### Cross-paradigm RSA agreement

Computed as Spearman correlation between the upper-triangle vectors of
the three paradigm RSA matrices (across all priors).

| Paradigm pair | Spearman r |
|---|---|
| Thatcher ↔ Composite | **0.86** |
| Thatcher ↔ Part-Whole | **0.54** |
| Composite ↔ Part-Whole | **0.73** |

**Part-Whole is the most distinct paradigm** — its representational geometry
diverges most from the other two. This is consistent with its different
stimulus structure: Part-Whole pairs eye-region-isolated vs eye-in-whole-face
rather than perturbation-vs-normal.

## Interpretation

[CONFIRMED] **1. Cluster structure changes by paradigm**. Several priors
migrate between clusters across paradigms:
- CORnet-S: alone (Thatcher) → semantic (Composite) → reconstructive
  (Part-Whole)
- AdaFace-CASIA: face-rec angular-margin (Thatcher) → face-rec triplet
  (Composite) → semantic (Part-Whole)
- FaceNet ×2: face-rec triplet (Thatcher/Composite) → semantic (Part-Whole)
- AuraFace: face-rec (Thatcher) → its own pair (Composite) → semantic
  (Part-Whole)

This implies **dissociation is a (model × paradigm) interaction, not a
fixed model-identity property**. A model can look "face-rec-like" in one
paradigm and "semantic-like" in another, on the same stimuli.

[CONFIRMED] **2. Part-Whole is the least correlated paradigm**.
Spearman(RSA_Thatcher, RSA_PartWhole) = 0.54 — only modest agreement. This
reflects that Part-Whole stimulus pairs are structurally different (whole vs
isolated), engaging different representational properties than perturbation-
based paradigms.

[CONFIRMED] **3. CORnet-S is the most paradigm-sensitive prior**. It
migrates across all three paradigms (alone → semantic → reconstructive).
This is consistent with E028 showing CORnet has extremely paradigm-
dependent metrics (Thatcher 0.99, Composite 1.15, Part-Whole 0.21).

[CONJECTURE] **4. The "semantic" cluster is a paradigm-conditional grouping**.
On Part-Whole, FaceNet/AuraFace/AdaFace-CASIA all collapse into the big CLIP-
DINOv2-SigLIP cluster. This suggests that on the part-whole task, the
representational geometry of these models becomes more similar — possibly
because the dominant axis of variation (whole-face vs isolated) is captured
by all of them, swamping the finer face-feature distinctions.

[CONJECTURE] **5. Implication for face-vision modeling**: a model's
illusion-fingerprint is not a one-dimensional property. The same architecture
can have different effective representations depending on what the task
requires. This supports designing **paradigm-conditional decoders** rather
than expecting one CLIP-like embedding to capture all configural-face
processing.

## Figures

- `figures/exports/e034_rsa_composite.png` — RSA heatmap on Composite 400-stim
- `figures/exports/e034_rsa_partwhole.png` — RSA heatmap on Part-Whole 400-stim
- Compared with `e033_rsa_heatmap.png` for Thatcher

For paper: combine into a 3-panel "Figure 3" showing RSA across all three
paradigms, with the migration arrows annotated for CORnet, AdaFace-CASIA,
and FaceNet.

## Implications for paper

This is a substantial new result section. Proposed paper organization:
- §3.1 Image-side dissociation (existing — paradigm × prior matrix)
- §3.2 NEW: Representational similarity across priors (E033 + E034) — cluster
  taxonomy + paradigm-conditional migration
- §3.3 EEG-side preservation (E020 — Route A)
- §3.4 Idea-003 sub-paths (E028-E032)

## Implications for Idea-001 and Idea-003

- **Idea-001**: paradigm-conditional cluster migration adds evidence that
  the dissociation we measure is real and not a model-identity artifact.
  Confidence raised to 9.0+/10.
- **Idea-003**: AdaFace-CASIA's migration from "face-rec cluster" (Thatcher)
  to "semantic cluster" (Part-Whole) is striking — it means face-rec-trained
  models can behave "semantic-like" depending on what the task asks. This
  may motivate a hybrid bio-prior + face-rec representation for ATM,
  where the embedding adapts to the task.

## Replicability

- Script: `/tmp/e034_rsa_per_paradigm.py`
- Outputs: `outputs/tables/e034_rsa_{composite,partwhole}.csv`
- Figures: `figures/exports/e034_rsa_{composite,partwhole}.png`

---
