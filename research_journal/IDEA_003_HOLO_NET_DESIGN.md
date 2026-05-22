# HOLO-Net — Detailed Design Document (Idea-003 sub-path e)

**Status**: design doc v1.0 (tick 40 pivot)
**Date**: 2026-05-22
**Author**: research-loop-eeg-illusion autonomous loop, user-directed pivot

---

# 0. Why a model, not a recipe?

Sub-path (a) — "AdaFace loss + CORnet anatomy + CASIA data" — is a training
recipe. It combines three off-the-shelf components without proposing a new
architecture. Even if it works, it explains **nothing**.

HOLO-Net is a **new architectural design** where each component
correspond to a specific brain region or brain mechanism, and each
component predicts a specific illusion-paradigm signal. The design's
existence stands or falls on whether the **emergent** representational
properties on IllusionBench match human face-illusion sensitivity, with
training data deliberately scrubbed of any illusion-stimulus exposure.

## Design principles (per user 2026-05-22)

1. **Strict bio-fidelity**: every layer maps to a specific brain region or
   mechanism. The mapping must be auditable against fMRI region masks (FFA,
   OFA, MFP, AFP, ATL) and EEG ERP components (P1, N170, P2, N250).
2. **Multi-layer EEG readout**: the model's representations at multiple
   stages (V1, V2, V4, MFP, AFP, FFA, ATL) must each be available for an
   EEG decoder to read out — enabling "which brain region does EEG best
   resolve" empirical analysis.
3. **Train on >100GB face data**: scale comparable to industrial face-
   recognition; not the 4M-sample regimes of CVLFace releases.
4. **NO illusion-stimulus contamination**: training set explicitly excludes
   Thatcher / composite / part-whole perturbations. Hold-out test
   integrity is paramount. Whatever sensitivity emerges must be a property
   of the architecture + identity training, NOT a memorization of
   benchmark stimuli.

---

# 1. Brain-science foundations (each maps to one architectural component)

| Brain region / mechanism | Function | Key references | Model component |
|---|---|---|---|
| **LGN (Magno + Parvo)** | LGN-Magno: low-SF, fast (~30ms), color-blind; LGN-Parvo: high-SF, slow (~50ms), color-sensitive | Bar (2003) PNAS; Sugase et al. (1999) Nature | **Dual input streams**: Magno (Gaussian σ=2.5px low-SF) and Parvo (full-res high-SF) before V1 |
| **V1 — primary visual cortex** | Oriented edge detectors, simple/complex cells | Hubel & Wiesel 1962; Kubilius et al. (2019) NeurIPS — CORnet-S V1 spec | **V1-COR block**: 7×7 conv stride 2 + maxpool + 3×3 conv, feedforward only |
| **V2** | Contour integration, illusory contour, surface form | Lamme & Roelfsema (2000); CORnet-S V2 spec | **V2-COR block**: 2-cycle recurrent bottleneck conv |
| **V4** | Mid-level shape, curvature, face-fragment selectivity | Pasupathy & Connor (2002); CORnet-S V4 spec | **V4-COR block**: 4-cycle recurrent bottleneck conv |
| **OFA — occipital face area** | Face-feature detection (eyes, mouth, nose); part of core network | Pitcher et al. (2007); Rossion et al. (2003) | **OFA branch**: face/non-face classifier head off V4 → drives V4 specialization toward face features without disrupting general object processing |
| **MFP — middle face patches** | Strongly face-selective; view-specific identity cells; ~97% face-selective fraction (Tsao 2006) or 84% (Freiwald 2010) | Freiwald & Tsao (2010) Science; Tsao et al. (2006) Science | **IT-COR block (MFP)**: 2-cycle recurrent CORnet IT; output = "MFP representation" |
| **AFP — anterior face patches** | View-invariant identity coding; pose-tolerant face cells | Freiwald & Tsao (2010); Meyers et al. (2015) | **AFP block**: 1-block bottleneck after MFP, trained with view-invariance contrastive aux loss |
| **FFA — fusiform face area** | Holistic / configural face representation; impaired by inversion | Yovel & Kanwisher (2005); Liu et al. (2010); Kanwisher (1997) | **FFA module**: 3-step recurrent self-attention over MFP/AFP spatial locations, gated by Magno-gist global query |
| **ATL — anterior temporal lobe** | Face identity + semantic person knowledge; lesion → prosopagnosia | Collins & Olson (2014); Rajimehr et al. (2009) PNAS | **ATL head**: 512-d L2-normalized identity embedding with AdaFace AM loss |
| **OFC / PFC top-down magno-gist** | Rapid object category prediction via low-SF magno bypass (Bar 2003 model) | Bar (2003) Nature Neurosci; Bar et al. (2006) PNAS | **PFC-Gist MLP**: small 2-layer MLP on Magno-pathway pre-V1 output → outputs gist token used as global query in FFA attention |
| **Predictive coding feedback** | Top-down predictions explain away expected input; prediction errors drive learning | Rao & Ballard (1999) Nat Neurosci; Friston (2010); Mozafari et al. (2021) NeurIPS — Predify | **Predify-style feedback paths**: IT → V4 → V2 → V1, MSE reconstruction loss at each level, 3 iterations |
| **Inversion-sensitive computation** | N170 face-inversion effect ~150ms; inverted faces lack holistic binding | Yin (1969); Rossion & Jacques (2008) Neuroimage; Liu-Shuang et al. (2014) | **Orientation Gate**: 1-layer MLP upright/inverted classifier output as gating signal (×) on FFA input; selectively suppresses FFA holistic recurrence for inverted faces |

---

# 2. Full architecture (HOLO-Net v1.0)

```
                                                                  ┌─────────────┐
                                                                  │ Magno-Gist  │  (Bar 2003)
                                              ┌────────────┐      │   PFC-MLP   │
                              ┌──→ LGN-Magno ─→ V1-magno   │      │ face/non-   │
INPUT (224×224 RGB)           │   (Gaussian   │             │      │  face gist  │
                              │    σ=2.5px)   └────────────┘      └──────┬──────┘
        │                     │                       │                  │ gist_token
        ├─→ Pre-LGN ──────────┤                       │                  │
        │   (channel split)   │                       │                  │
                              │   ┌────────────┐      │                  │
                              └──→ LGN-Parvo  ─→ V1-parvo │              │
                                  (full-res)    └────────────┘           │
                                                       │                  │
                                                       ▼                  │
                                              ┌───────────────┐           │
                                              │ V1-CORnet     │           │
                                              │ 7×7 conv      │           │
                                              │ +maxpool      │           │
                                              │ +3×3 conv     │           │
                                              └───────┬───────┘           │
                                                      ▼                   │
                                              ┌───────────────┐           │
                                              │ V2-CORnet     │  ← PC fb  │
                                              │ 2-cycle       │           │
                                              │ recurrent     │           │
                                              └───────┬───────┘           │
                                                      ▼                   │
                                              ┌───────────────┐           │
                                              │ V4-CORnet     │  ← PC fb  │
                                              │ 4-cycle       ├──→ OFA    │
                                              │ recurrent     │   branch  │
                                              └───────┬───────┘   (face/  │
                                                      │           non-    │
                                                      │           face)   │
                                                      ▼                   │
                                              ┌───────────────┐           │
                                              │ IT-CORnet     │  ← PC fb  │
                                              │ (MFP)         │           │
                                              │ 2-cycle       │           │
                                              │ recurrent     │           │
                                              └───────┬───────┘           │
                                                      ▼                   │
                                              ┌───────────────┐           │
                                              │ AFP block     │           │
                                              │ pose-invariant│           │
                                              │ contrastive   │           │
                                              └───────┬───────┘           │
                                                      ▼                   │
                                              ┌───────────────┐           │
                                              │ Orientation   │           │
                                              │ Gate (gating  │           │
                                              │ signal)       │           │
                                              └───────┬───────┘           │
                                                      ▼                   │
                                            ┌──────────────────┐          │
                                            │  FFA module      │ ←────────┘
                                            │  3-step recurrent│  Magno-gist
                                            │  self-attention  │   as global
                                            │  Query=gist      │   query
                                            │  K,V=AFP spatial │
                                            └────────┬─────────┘
                                                     ▼
                                            ┌────────────────┐
                                            │  ATL head      │
                                            │  512-d emb     │
                                            │  AdaFace AM    │
                                            └────────────────┘
                                                     ▼
                                            ID classification

PC fb = Predify-style predictive-coding feedback paths
         IT ↔ V4 ↔ V2 ↔ V1 (3 iterations)
```

## Layer-by-layer specs

### LGN module (NEW vs CORnet)

```
LGN-Magno:
  Pre-LGN(3, 224, 224) → GaussianBlur(σ=2.5px) → grayscale 1ch upsample → 8ch projection
  Output: (B, 8, 112, 112) low-SF "gist"

LGN-Parvo:
  Pre-LGN(3, 224, 224) → identity (full res) → 3ch
  Output: (B, 3, 224, 224) high-SF "detail"
```

### V1-CORnet (modified for dual input)

```
V1-magno: Conv(8, 32, 7x7, stride=2) + MaxPool(3x3, stride=2) + Conv(32, 32, 3x3)
V1-parvo: Conv(3, 64, 7x7, stride=2) + MaxPool(3x3, stride=2) + Conv(64, 64, 3x3)
Output: concat([V1-magno upsampled to V1-parvo res, V1-parvo]) = (B, 96, 56, 56)
```

### V2, V4, IT (standard CORnet-S blocks)

Standard CORnet-S V2 (2-cycle), V4 (4-cycle), IT (2-cycle) recurrent
bottleneck convs. Channel dims: V1→V2=128 → V4=256 → IT(MFP)=512.

### AFP block

```
MFP(B, 512, 7, 7) → 1×1 conv (512→256) → 3×3 conv recurrent (1 cycle)
                  → 1×1 conv (256→512) → GlobalAvgPool → 512-d AFP embedding
View-invariance loss: same-identity-different-pose embeddings should be close
```

### Orientation Gate

```
AFP(B, 512) → MLP(512 → 64 → 2) → upright/inverted logits
Gate output: sigmoid(upright_logit - inverted_logit) ∈ [0, 1]
Applied: FFA_input = AFP_spatial × gate (broadcasting per-batch)
```

### FFA module (NEW vs CORnet)

```
MFP_spatial(B, 512, 7, 7) reshape → (B, 49, 512) tokens
Magno-gist token from PFC MLP: (B, 1, 512)
Query: gist_token (B, 1, 512)
Key/Value: AFP_spatial + gate (B, 49, 512)
3-step recurrent self-attention:
  for t in [1, 2, 3]:
    h_t = MultiHeadAttention(Q=gist, K=K_t, V=V_t, num_heads=8)
    K_{t+1} = LayerNorm(K_t + h_t)  # feedback
    V_{t+1} = LayerNorm(V_t + h_t)
FFA_output = h_3 (the last attention output, holistic-bound representation)
Output shape: (B, 1, 512) → squeeze → (B, 512)
```

### ATL head

```
FFA_output(B, 512) → AdaFace quality-adaptive angular margin head
→ Identity classification on training set classes
Output: 512-d L2-normalized embedding (= "ATL representation")
```

### Predify predictive-coding feedback

Using `miladmozafari/predify` package (PyTorch, GNU GPL v3):

```toml
# predify config (.toml)
[[modules]]
name = "V1"
predictor_idx = 1
[[modules]]
name = "V2"
predictor_idx = 2
[[modules]]
name = "V4"
predictor_idx = 3
[[modules]]
name = "IT"
predictor_idx = 4

[settings]
num_iterations = 3
prediction_loss_weight = 0.1
```

Each predictor is a deconvolution from layer_{n+1} to layer_n that
reconstructs the pre-activations. MSE reconstruction is added to the
total loss with weight 0.1.

### PFC-Gist MLP

```
LGN-Magno(B, 8, 112, 112) → AdaptiveAvgPool(1, 1) → (B, 8)
→ MLP(8 → 64 → 256 → 512) → ReLU → Norm → gist_token (B, 1, 512)
Auxiliary task: face/non-face binary cross-entropy with weight 0.05
(provides supervision signal for Magno-Gist pathway)
```

---

# 3. Training losses (multi-task)

```
L_total = α₁ · L_identity     (AdaFace AM, primary, α₁ = 1.0)
        + α₂ · L_predcode     (V1+V2+V4 reconstruction MSE, α₂ = 0.1)
        + α₃ · L_orientation  (upright/inverted CE, α₃ = 0.1)
        + α₄ · L_face_detect  (OFA face/non-face CE, α₄ = 0.05)
        + α₅ · L_view_invar   (same-id-diff-pose contrastive, α₅ = 0.2)
        + α₆ · L_gist         (PFC face/non-face from magno, α₆ = 0.05)
```

**Critical**: NONE of these losses use thatcherized or perturbed face
stimuli. All Thatcher / composite / part-whole are test-only.

---

# 4. Training data (>100GB, no illusion perturbations)

## Primary: **Glint360K** (recommended)

- 17M images, 360K identities, ~140 GB on disk
- WebDataset format, 16GB shards, streaming-friendly
- Already used to train AuraFace, ArcFace
- Cleaned via insightface pipeline
- Non-commercial research license — appropriate for our use

## Augmentation strategy (clean, no illusion stimuli)

- **Random crop / horizontal flip** (standard)
- **Pose augmentation** (random head rotation ±15°) — for view-invariance loss
- **Orientation augmentation**: random vertical flip (= face inversion) — for Orientation Gate training. **Crucially, only WHOLE-face inversion, never feature-region inversion (no Thatcher)**.
- **Color jitter / brightness** (standard)
- **Random erase** (NOT in face-feature regions — would risk overlap with Thatcher test perturbation; instead use random non-feature locations from MediaPipe landmarks)

## Non-face balancing

- Mix in 10% **ImageNet-1K** images as negative examples for the OFA face/non-face classifier and PFC gist task
- Ensures the model develops face-specific specialization rather than treating everything as faces

## Total dataset on disk

- Glint360K: ~140 GB
- ImageNet-1K (10% mix): ~15 GB
- **Total: ~155 GB** — exceeds the >100GB requirement

---

# 5. Training plan (3-5 days H100)

## Stage 1 — ImageNet pretrain (12-18 hours on H100)

- Backbone: CORnet-S full pipeline (V1+V2+V4+IT)
- Loss: ImageNet 1K classification only
- This stage = standard CORnet-S behavior; gives V1-V4 sensible features
- Skip if we use the CORnet-S released checkpoint as init

## Stage 2 — face fine-tune (36-72 hours on H100)

- Initialize from Stage 1 checkpoint
- Train all components (LGN, V1-V4, IT/MFP, AFP, FFA, ATL, OFA branch,
  Orientation Gate, PFC-Gist MLP, Predify feedback)
- Multi-task loss as in Section 3
- Batch size 256, AdamW lr 1e-4 with linear warmup + cosine decay
- 50-100 epochs (depending on convergence on validation)

**Validation set (NOT illusion stimuli)**:
- LFW verification accuracy (target ≥ 99%)
- IJB-B / IJB-C verification curves (held-out faces, no Thatcher)

## Stage 3 — multi-layer EEG decoder training (8-12 hours)

For each of layers {V1, V2, V4, MFP, AFP, FFA, ATL}:
- Train an ATM-style EEG-encoder → target-layer head
- THINGS-EEG2 raw EEG data → HOLO-Net layer activations on
  THINGS-EEG2 image stimuli
- Loss: per-layer contrastive (InfoNCE) like ATM
- 7 EEG-encoder heads, each ~50M params

Output: 7 EEG → layer mappings, each gives per-subject preservation r

---

# 6. Pre-registered falsification criteria

The design **must** simultaneously satisfy on IllusionBench-EEG (without
having seen these stimuli during training):

1. **Thatcher ISI ≥ 3.0** at the FFA layer
   - This exceeds AdaFace-IR50-CASIA (2.91), the previous best face-rec
   - It's also the lower bound for "approaching human ISI 4-5"
2. **Composite CSI ≥ 1.5** at the FFA layer
   - Comparable to DINOv2 (1.4-1.7)
3. **Part-Whole PWI ≤ 0.5** at the FFA layer
   - Comparable to DINOv2 (0.31-0.40) or even better
4. **Random-bbox ISI ≤ 1.5** at the FFA layer
   - i.e., face-feature-specific, not general orientation bias
5. **All four simultaneously** — no current prior in our 25-prior battery
   satisfies this, so success would be a clean existence proof

**Additional bonus criteria (not falsification but desired)**:
6. Per-layer dissociation: Thatcher ISI should INCREASE through stages
   (V4 < MFP < AFP < FFA), monotonically following the brain-region depth
7. Magno-Gist ablation: removing the gist token from FFA query should
   reduce Part-Whole effect (i.e., the magno feedback drives whole-context)
8. Orientation Gate ablation: removing the gate should reduce Thatcher
   ISI by > 30% (i.e., the inversion-aware gating drives upright-specific
   binding)

**EEG-side claim**: at least one layer (V4 / MFP / AFP / FFA) must give
per-dim preservation r > 0.20 on THINGS-EEG2 face-subset (current ATM
result on face subset with CLIP target is 0.150). If r > 0.25, that's a
clear win over the CLIP anchoring.

---

# 7. Per-paradigm mechanistic predictions

| Paradigm | Mechanism in HOLO-Net | Quantitative prediction |
|---|---|---|
| **Thatcher** | Orientation Gate suppresses FFA recurrence on inverted faces → only upright faces engage configural binding via 3-step self-attention → small d_inv, large d_up | FFA-layer ISI ≥ 3.0 |
| **Composite** | When face halves are aligned, magno-gist sees a complete face shape → query attends across all AFP spatial positions, binding both halves into a unified holistic representation. When misaligned, gist sees a "broken" shape → attention fails to bind → larger d_aligned | FFA-layer CSI ≥ 1.5 |
| **Part-Whole** | When eye region is in whole-face context, magno-gist provides face shape priors → eye representation is bound to whole. When isolated on gray, magno-gist provides no template → eye is processed as a generic visual fragment → very different from whole-context eye | FFA-layer PWI ≤ 0.5 |

---

# 8. Falsification scenarios

If after Stage 2, FFA-layer fails ANY of (Thatcher ≥ 3, Composite ≥ 1.5,
Part-Whole ≤ 0.5, Random-bbox ≤ 1.5):

### Diagnosis path
1. **Per-layer profiling**: extract embeddings at every layer (V1 → ... →
   FFA). Identify where the metric breaks. Three diagnoses:
   - Metric breaks early (V1, V2, V4): backbone CORnet-S is the bottleneck
   - Metric breaks at MFP/AFP: face-feature-tuning insufficient
   - Metric breaks at FFA: holistic binding fails despite per-feature tuning
2. **Component ablation**: train ablation versions
   - HOLO-Net minus Magno → Parvo only
   - HOLO-Net minus FFA module → AFP-only readout
   - HOLO-Net minus Orientation Gate
   - HOLO-Net minus Predify feedback
3. Report which components are necessary vs sufficient

### Architecture revisions if all components present but still falsified
- Replace 3-step self-attention with 6-step (longer settling)
- Add **Capsule network**-style explicit part-whole binding in FFA
  module (Hinton 2017, Sabour 2017) — capsule routing-by-agreement
  may suit part-whole binding better than self-attention
- Add **GLOM**-style nested hierarchical representations (Hinton 2021)

---

# 9. Open implementation questions

1. **Magno/Parvo channel mixing**: should V1-Magno and V1-Parvo concat
   immediately, or should they each go through V2-V4 separately and only
   merge at IT (more brain-like)? Plan: try both; expect simpler version
   (concat at V1) to be sufficient.

2. **Predify configuration**: how many iterations (3 vs 5 vs 10)?
   Predify paper used 3-5 for VGG16/EffNet. Plan: start with 3, ablate.

3. **Orientation Gate hard vs soft**: should gate be a hard binary mask
   (upright=1, inverted=0) or soft probabilistic (sigmoid)? Soft allows
   gradient flow but may not produce strong enough Thatcher asymmetry.
   Plan: start soft (sigmoid), with auxiliary CE loss to push toward
   confidence; ablate hard vs soft.

4. **AdaFace IR-50 init vs from-scratch**: should AFP/ATL head init from
   pretrained AdaFace IR-50 CASIA (P26, our best face-rec prior), or train
   from scratch? Init = warm-start (faster); from-scratch = clean.
   Plan: warm-start, but verify Thatcher emergence isn't just from the
   AdaFace init (compare with from-scratch ablation).

5. **EEG Stage 3 implementation**: ATM uses single InfoNCE; for multi-
   layer readout, do we train 7 independent ATM heads, or a single
   multi-head ATM that simultaneously predicts all layers? Plan: start
   with independent heads (cleaner ablation), then try multi-head if
   compute permits.

---

# 10. Implementation milestones (suggested tick mapping)

| Tick | Task | Duration |
|---|---|---|
| 40 | Design doc (this) + lit review | 1 tick (done) |
| 41 | PyTorch skeleton (architecture only, no training) — verify dims work | 1 tick |
| 42 | Predify integration + LGN dual-stream sanity check | 1 tick |
| 43 | Acquire Glint360K dataset (download + verify) | 1 tick |
| 44 | Single-batch overfit test (Stage 2 starting) | 1 tick |
| 45 | Launch 12-hour H100 training (Stage 1: ImageNet) | 1 tick |
| 46-48 | Stage 2 face fine-tune training (multi-day) | 2-3 ticks |
| 49 | IllusionBench evaluation on FFA layer | 1 tick |
| 50 | Per-layer dissociation analysis | 1 tick |
| 51 | Ablation studies | 1 tick |
| 52-54 | Stage 3 EEG decoder training | 2-3 ticks |
| 55-56 | EEG-side analysis + writeup | 2 ticks |

**Total estimated calendar time**: 5-7 days if user is available to
monitor and the H100 stays online. Probably 8-10 days realistic.

---

# 11. References

- Bar, M. (2003). A cortical mechanism for triggering top-down facilitation in visual object recognition. *Journal of Cognitive Neuroscience*. https://www.tandfonline.com/doi/full/10.1162/089892903321593117
- Collins, J. A., & Olson, I. R. (2014). Beyond the FFA: The role of the ventral anterior temporal lobes in face processing. *Neuropsychologia*. https://pmc.ncbi.nlm.nih.gov/articles/PMC4122611/
- Felleman, D. J., & Van Essen, D. C. (1991). Distributed hierarchical processing in the primate cerebral cortex. *Cerebral Cortex*.
- Freiwald, W. A., & Tsao, D. Y. (2010). Functional compartmentalization and viewpoint generalization within the macaque face-processing system. *Science*.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*.
- Hinton, G. (2021). How to represent part-whole hierarchies in a neural network. arXiv:2102.12627 (GLOM).
- Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex.
- Kanwisher, N., McDermott, J., & Chun, M. M. (1997). The fusiform face area: a module in human extrastriate cortex specialized for face perception. *Journal of Neuroscience*.
- Kim, M., Jain, A. K., & Liu, X. (2022). AdaFace: Quality Adaptive Margin for Face Recognition. *CVPR*. arXiv:2204.00964
- Kubilius, J., et al. (2019). Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs (CORnet-S). *NeurIPS*. https://arxiv.org/abs/1909.06161
- Lamme, V. A. F., & Roelfsema, P. R. (2000). The distinct modes of vision offered by feedforward and recurrent processing. *Trends in Neurosciences*.
- Mozafari, M., et al. (2021). Predify: Augmenting deep neural networks with brain-inspired predictive coding dynamics. *NeurIPS*. arXiv:2106.02749. GitHub: miladmozafari/predify
- Pitcher, D., et al. (2007). TMS evidence for the involvement of the right occipital face area in early face processing. *Current Biology*.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*.
- Rossion, B., & Jacques, C. (2008). Does physical interstimulus variance account for early electrophysiological face sensitive responses in the human brain? Ten lessons on the N170. *Neuroimage*.
- Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic Routing Between Capsules. *NeurIPS*.
- Sugase, Y., et al. (1999). Global and fine information coded by single neurons in the temporal visual cortex. *Nature*.
- Tanaka, J. W., & Sengco, J. A. (1997). Features and their configuration in face recognition. *Memory & Cognition*.
- Thompson, P. (1980). Margaret Thatcher: a new illusion. *Perception*.
- Tsao, D. Y., et al. (2006). A cortical region consisting entirely of face-selective cells. *Science*.
- Yin, R. K. (1969). Looking at upside-down faces. *Journal of Experimental Psychology*.
- Yovel, G., & Kanwisher, N. (2005). The neural basis of the behavioral face-inversion effect. *Current Biology*.
- Young, A. W., Hellawell, D., & Hay, D. C. (1987). Configurational information in face perception. *Perception*.

WebFace42M / Glint360K dataset references:
- Zhu, Z., et al. (2021). WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition. *CVPR*. https://arxiv.org/abs/2103.04098
- An, X., et al. (2021). Partial FC: Training 10 Million Identities on a Single Machine. arXiv:2010.05222 (Glint360K)

---

# 12. Outstanding decisions to be discussed with user

The design above is **proposed** but several choices are open. Before
implementation:

1. **Falsification thresholds OK?** Are the (ISI≥3, CSI≥1.5, PWI≤0.5,
   Random-bbox≤1.5) criteria acceptable? Or should some be tighter
   (e.g., ISI≥4 to match human range)?

2. **Predify is the right predictive-coding implementation, or should we
   roll our own?** Predify is convenient but locks us to specific
   patterns; a custom impl gives more flexibility.

3. **Capsule networks / GLOM as fallback?** If HOLO-Net v1 fails, do we
   want capsule-routing or GLOM-nested-representations as the next
   architectural avenue (vs another revision of the self-attention FFA
   module)?

4. **Calendar time budget**: 5-7 days minimum for full
   training+evaluation. Is this OK or do we need a faster prototype?
   - Faster: skip Stage 1 ImageNet pretrain (use CORnet-S released ckpt),
     skip Stage 3 EEG (do image-side only first), → 2-3 days.
   - Full plan: 5-7 days.
