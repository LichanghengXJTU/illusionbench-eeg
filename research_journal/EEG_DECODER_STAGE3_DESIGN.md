# EEG Decoder Stage 3 — frozen DINOv2 target, ATM-EEG features as source

**Status**: design locked tick 81; scaffolded (`eeg_decoder/stage3_ridge.py`);
data fetched (ATM EEG features train+test + ViT-H-14 image features).
**Design lock-in date**: 2026-05-23 (tick 81).
**Prior context**: tick 80 closed with HOLO-Net v3 (frozen DINOv2 + global FTPC)
giving 2/4 §6 pass — best HOLO-Net yet, but FAIL overall. The DINOv2 backbone
itself was responsible for the PWI flip (2.20→0.446); the FTPC residual added
nothing significant. This makes frozen DINOv2 a CREDIBLE EEG-decoder target
that already shows measurable human-alignment on 2 of 4 IllusionBench criteria.

---

## 1. The deliverable

A trained EEG-decoder mapping that, given (per-subject) ATM-style EEG features
recorded under THINGS-EEG2, produces a per-trial embedding in **frozen DINOv2
ViT-S/14 feature space** (384-d), with two evaluation prongs:

  **(A)** **Standard EEG-decoder retrieval** on THINGS-EEG2 test set (200
       images): top-K retrieval accuracy of the EEG-decoded embedding against
       the 200 DINOv2 image features. Comparable to ATM's reported CLIP-H/14
       retrieval; our prediction is comparable retrieval (DINOv2 is a strong
       SSL feature) with the added benefit of human-aligned illusion
       processing on the IllusionBench transfer.

  **(B)** **IllusionBench-EEG human-alignment transfer (the headline)**: the
       trained EEG→DINOv2 mapping is then applied to project hypothetical
       EEG-decoded representations of the Thatcher / Composite / Part-Whole /
       Random-bbox stimuli (we don't have EEG for these — we use the inverse
       map: DINOv2-stimulus → EEG-pseudoembedding via the ridge transposed,
       then measure if ISI/CSI/PWI are preserved). Equivalent to E020's
       Route A analysis but using DINOv2 instead of CLIP-H/14 as the
       feature substrate.

---

## 2. Why this is the right Stage-3 design (red-team checks)

**Why DINOv2 target, not CLIP-H/14**:
- CLIP-H/14 was the ATM target and E020 already characterised it (uniform
  per-dim attenuation, no face-specific structure). No new info to extract.
- DINOv2 already shows IllusionBench paradigm-consistent profile from E033
  (composite/PW) AND from E045 (PWI 0.446 PASS, ISIrbox 0.654 PASS at the
  raw-feature layer). Re-targeting to DINOv2 is the natural test of whether
  the EEG bottleneck preserves DINOv2's better-aligned signal.

**Why ATM-EEG features as source (not raw EEG)**:
- The ATM EEG encoder is a strong, well-tested spatiotemporal CNN. Re-training
  it is multi-day single-GPU work and adds an architectural confound to the
  human-alignment claim.
- ATM features are 1024-d, CLIP-H/14-aligned by training objective, but
  contain the EEG-recoverable information per-trial. Re-projecting them to
  DINOv2 via ridge isolates the "what part of EEG-recoverable information
  aligns with DINOv2" question cleanly.
- This DOES couple us to the CLIP-H/14 attentional prior of ATM's encoder —
  acknowledged as a limitation; if Stage 3 results warrant, Stage 4 re-trains
  ATM from raw EEG with DINOv2 target.

**Why ridge regression as the mapping**:
- 16,540 training pairs; 1024 → 384 dims; ridge is the principled minimum-
  capacity linear regressor and has closed-form solution.
- Per-subject + leave-one-subject-out for cross-subject generalization.
- More expressive mappings (MLP) considered for Stage 3b after ridge baseline
  is in place — the headline measurement is the per-subject mapping behaviour,
  not maximum top-K.

**Why frozen DINOv2, no fine-tuning**:
- E045 already evaluated frozen DINOv2 on IllusionBench. Fine-tuning would
  change the reference frame and invalidate the comparison.
- The point IS to measure whether the EEG signal preserves DINOv2's existing
  perceptual structure.

---

## 3. Data inventory (tick 81, downloaded to server)

| What | Path | Shape | Notes |
|------|------|-------|-------|
| ATM EEG train (×10 sub) | `/workspace/illusionbench-eeg/data/atm_emb_eeg/emb_eeg/ATM_S_eeg_features_sub-{01..10}_train.pt` | (66160, 1024) ea | 16540 stims × 4 reps per sub, CLIP-H/14-aligned |
| ATM EEG test (×10 sub) | same dir, `*_test.pt` | (200, 1024) ea | 200 unique test stims, averaged across the 80 reps per sub |
| ViT-H-14 image train | `/workspace/illusionbench-eeg/data/things_features/ViT-H-14_features_train.pt` | dict with img_features (16540, 1024), text_features (1654, 1024) | THINGS image-side CLIP features |
| ViT-H-14 image test | same dir, `_test.pt` | dict with img_features (200, 1024), text_features (200, 1024) | THINGS image-side CLIP features |
| **NOT YET ON SERVER** — THINGS RGB images (1854 concepts) | needed for DINOv2 extraction | — | osf.io THINGS official + per-concept image fetch script needed (tick 82) |
| **NOT YET COMPUTED** — DINOv2 features on THINGS images | will go to `/workspace/illusionbench-eeg/data/things_features/dinov2_vits14_*.pt` | (16540, 384) train, (200, 384) test | tick 82 task |

---

## 4. Module layout

```
eeg_decoder/
├── __init__.py
├── stage3_ridge.py       — main training + eval driver (tick 83)
├── data.py               — loaders for ATM EEG + image features (tick 81)
├── extract_dinov2_things.py — DINOv2 feature extraction on THINGS images (tick 82)
└── illusionbench_transfer.py — IllusionBench transfer eval (tick 84)
```

---

## 5. Step-by-step execution plan

- **Tick 81 (now)**: this design doc + data fetched + `data.py` loader stub.
- **Tick 82**: download THINGS images from official OSF + extract DINOv2
  features (`extract_dinov2_things.py`). Estimated 2 GB download + 1 min
  DINOv2 forward at 16540×384 on H100.
- **Tick 83**: `stage3_ridge.py` — train per-subject ridge ATM-EEG (16540 × 1024)
  → DINOv2 (16540 × 384); evaluate top-1 / top-5 retrieval on test set; record
  per-subject metric distribution. Compare to ATM's published CLIP-target
  retrieval as sanity check.
- **Tick 84**: `illusionbench_transfer.py` — apply trained mapping to project
  Thatcher / Composite / Part-Whole / Random-bbox DINOv2 features, measure
  ISI/CSI/PWI on the EEG-decoded embeddings. **The headline result**: does
  DINOv2-targeted EEG decoding preserve any of HOLO-Net v3's 2/4 PASS at the
  raw feature layer?

---

## 6. Pre-registered predictions (for the headline tick 84)

Filed BEFORE running tick 83+84 so we can't ex-post-rationalize:

- **Top-K retrieval** (tick 83): per-subject top-1 on 200-class retrieval
  expected in 18-32% range (ATM CLIP-target reported ~28-46%; DINOv2 target is
  expected slightly weaker on object retrieval). Below 18% (= 36× chance level
  but quite low for ATM-quality EEG) → ridge insufficient; consider Stage 3b
  MLP mapping.
- **IllusionBench transfer (tick 84)** at the EEG-decoded DINOv2 layer:
  - ISIrbox preserved (≤1.5) — high confidence (DINOv2 raw was 0.654)
  - PWI preserved (≤0.5) — moderate confidence (raw 0.446; ridge attenuation
    could push toward chance 0.7-1.0)
  - CSI preserved (≥1.5 ideal, ≥1.2 partial) — low confidence (raw 1.300
    was already FAIL; attenuation likely makes worse)
  - ISI improved (≥1.0 = no anti-Thatcher) — low-moderate (raw 0.901
    anti-Thatcher; if ridge collapses to mean, would move to 1.0)

The cleanest **positive result we could honestly claim**: "EEG-decoded
DINOv2-targeted embeddings preserve part-whole and random-bbox §6 PASS but
attenuate Thatcher and composite — establishing the EEG bottleneck as the
limiting factor distinct from the visual prior choice, AND demonstrating that
DINOv2 is a paradigm-consistent EEG-decoder target where CLIP-H/14 (E020) was
not."

The cleanest **negative result we could honestly claim**: "Per E020 result the
EEG bottleneck is a low-pass on per-dim information; the DINOv2 retarget does
not change this — all 4 paradigms attenuate to baseline. This validates the
benchmark's discriminating power: even SOTA SSL priors don't pass through the
current EEG bottleneck, motivating sensor-side or training-side innovation."

---

## 7. Strict-fairness audit (per user constraint 公平严格)

- **Training data**: THINGS-EEG2 → 1654 concepts of natural objects. **NO
  faces** in training (concepts include "person" but THINGS images are object-
  centred, not face-centred). NO FFHQ. NO illusion stimuli.
- **Identity disjointness**: THINGS-EEG2 train (1654 concepts) ∩ test (200
  concepts) = ∅ by THINGS-EEG2 construction.
- **Zero illusion stimuli in training**: confirmed — IllusionBench stimuli are
  only used at tick 84 transfer eval, never in training.
- **No FFHQ in training**: confirmed — FFHQ is used only for IllusionBench
  test stimulus generation.
- **Subject splits**: per-subject training, no cross-subject leakage in the
  ridge.

---

## 8. Connection to other ideas
- **Idea-001 (IllusionBench-EEG)**: Stage 3 is the first EEG-side measurement
  on DINOv2; complements the 25-prior image-side data, gives the benchmark a
  cross-modal data point.
- **Idea-003 (HOLO-Net positive)**: independent of HOLO-Net v2.2; if v2.2
  passes §6 we have an even better Stage-3 target (frozen DINOv2 + part-aware
  FTPC); if v2.2 fails we still have the DINOv2-only target.
- **Idea-002 (benchmark suite)**: Stage 3 result feeds the benchmark's
  "current-prior performance" table.
