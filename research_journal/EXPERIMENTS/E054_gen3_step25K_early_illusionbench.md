# E054 — Gen 3 v4.0 step-25K early IllusionBench §6 eval

**Hypothesis** (pre-registered tick 102): If the orient-aux head trained
during Gen 3 v4.0 has biased the ViT-S/14 backbone toward orientation-
asymmetric features, raw `afp_pooled` features at intermediate step 25K
will already show measurable Thatcher signal (ISI > 1.0, breaking the
anti-Thatcher direction of frozen DINOv2).

**Method**:
- Load Gen 3 ckpt at step 25000 (epoch 4, partial training)
- Forward IllusionBench stims through `Gen3Backbone(mode="features")` →
  extract afp_pooled (mean-pooled patch tokens, 384-d)
- L2-normalize, save NPZ per paradigm
- Run `analysis.compute_metrics`
- No FTPC head (raw backbone features only)
- Compares to v3 frozen-DINOv2 baseline (Gen 1)

**Result** (pixel-corrected at "afp" layer = raw afp_pooled):

| Paradigm | Gen 3 step-25K | v3 frozen DINOv2 (E045) | Δ | §6 verdict |
|----------|----------------:|------------------------:|---:|---|
| ISI (Thatcher) | **0.943** | 0.908 | +0.035 | FAIL (need ≥3.0) |
| CSI (Composite) | **1.176** | 1.282 | -0.106 | FAIL (need ≥1.5) |
| **PWI (Part-Whole)** | **0.145 ✓** | 0.397 ✓ | **-0.252 (= 2.7× lower!)** | **PASS** (need ≤0.5) |
| ISIrbox | 0.980 ✓ | 0.662 ✓ | +0.318 | PASS (need ≤1.5) |
| **ALL FOUR** | **2/4** | 2/4 | same count | FAIL |

CIs (Gen 3 step-25K):
- ISI: [0.891, 1.000]
- CSI: [1.241, 1.349] (raw 1.293, pixel-corrected 1.176)
- PWI: [0.152, 0.204] (raw 0.175, pixel-corrected 0.145)
- ISIrbox: [0.944, 1.020]

**Interpretation** (key findings):

1. [CONFIRMED] **PWI dramatic improvement**: 0.397 → 0.145 (-63%). The Gen 3
   backbone, even with only partial training (epoch 4 of 30), produces
   features with substantially stronger whole-context modulation of parts
   than frozen DINOv2. **New best-ever PWI across all Gen 1/2/3 variants
   we've tested**.

2. [CONFIRMED] **ISI essentially unchanged** (0.943 vs 0.908): still in
   anti-Thatcher direction, the orient-aux head did NOT translate to image-
   side Thatcher detection. **My pre-registered tick-102 hypothesis is
   refuted at step 25K**. May still emerge during remaining training, but
   unlikely to dramatically shift.

3. [CONJECTURE] **Mechanism diagnosis**: orient-aux head was trained on
   CLS-level 0°/180° classification. This biases the CLS feature to have an
   "orientation axis" but does NOT produce per-patch orientation-asymmetric
   features. Per Psalta 2014, Thatcher detection requires per-feature
   local orientation processing in the context of whole-face — exactly the
   structure CLS-level orient-aux cannot teach.

4. [CONFIRMED] **CSI slightly regressed** (1.282 → 1.176 pixel-corrected) and
   **ISIrbox margin shrank** (0.662 → 0.980). The new feature manifold
   shifted in ways that hurt composite-face and added some random-bbox
   sensitivity. Net effect on §6 pass count: same (2/4).

**Decision tree post-result**:

The Gen 3 backbone IS giving meaningful improvements (PWI) but not on the
critical Thatcher axis. Options:
- **A (default, continuing)**: Let training continue to step 50K (~10h)
  and re-eval. ISI MIGHT improve more with longer training.
- **B**: Stop now, build FTPC head on top of Gen 3 backbone, full eval.
  PWI 0.145 + ISIrbox 0.98 already pass — if FTPC adds Thatcher channel,
  could approach 3/4 pass.
- **C**: Stop now, redesign v4.1 with **per-patch orient task** (each
  patch predicts local orientation, not CLS-level) — most likely to break
  ISI but requires another full training (~50h).

**Status**: continuing default A. Will re-eval at step 50K (~10h from
tick 102). If ISI hasn't moved above 1.1 by then, escalate to option C.

**Replicability**:
- Code: `holo_net/eval_gen3_quick.py` (new)
- Checkpoint: `/workspace/runs/2026-05-24_gen3_v4.0_seed20260521/checkpoint.pt`
  (step 25000)
- NPZs: `outputs/embeddings/{paradigm}_HOLONET_v4.0_gen3_25K/`
- Tables: `outputs/tables/HOLONET_v4.0_gen3_25K_{paradigm}.csv`

**Linked Idea-###**:
- Idea-003: held at 5.5. The PWI improvement gives one image-side win;
  ISI/CSI still don't pass; backbone retrain alone doesn't solve Thatcher.
  If v4.1 (per-patch orient) works, score → 7.0+. Currently still in flight.
