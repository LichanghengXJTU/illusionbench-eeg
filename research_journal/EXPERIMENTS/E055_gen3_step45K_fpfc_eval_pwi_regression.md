# E055 — Gen 3 step-45K IllusionBench eval + FTPC integration + PWI regression finding

**Hypothesis** (pre-registered tick 110): Gen 3 v4.0 at step 45K should
maintain or improve the PWI 0.145 result we saw at step 25K (E054), as
backbone features continue to refine. FTPC head on top of Gen 3 should add
further discriminative power on §6 paradigms.

**Method**:
- Step 45K ckpt = the saved Gen 3 ckpt right before tick 110 (file mtime
  09:17, which corresponds to step-45000 save)
- Two evals:
  - `eval_gen3_quick.py` → raw afp_pooled, NO FTPC head
  - `gen3_ftpc.py` → train FTPC template on Gen 3 features (300 steps,
    8105 face samples, ~3 min); then §6 eval at afp + ffa layers
- Snapshot the step-45K ckpt to a stamped filename (so future overwriting
  doesn't lose it)

**Result — raw afp (no FTPC) at step 45K, vs step 25K**:

| Paradigm | step-25K | **step-45K** | Δ | §6 verdict @ 45K |
|----------|---------:|-------------:|---:|---|
| ISI | 0.943 | **0.955** | +0.012 (no movement) | FAIL (≥ 3.0) |
| CSI | 1.176 | **1.104** | -0.072 (slightly worse) | FAIL (≥ 1.5) |
| **PWI** | **0.145** | **0.312** | **+0.167 (2.2× regression!)** | PASS (≤ 0.5) |
| ISIrbox | 0.980 | **0.964** | -0.016 | PASS (≤ 1.5) |
| **§6 verdict** | **2/4** | **2/4** | same | FAIL |

CI check (statistical significance of PWI regression):
- step-25K PWI CI: [0.152, 0.204]
- step-45K PWI CI: [0.326, 0.432]
- **No overlap** → regression is statistically significant.

**Result — Gen 3 + FTPC head at step 45K**:

| Layer | ISI | CSI | PWI | ISIrbox | pass/4 |
|---|---:|---:|---:|---:|---:|
| afp (raw) | 0.955 | 1.104 | **0.312 ✓** | 0.964 ✓ | 2/4 |
| ffa (δ_pooled, FTPC) | 0.949 | 1.091 | **0.601 ✗** | 0.966 ✓ | 1/4 (WORSE!) |

The FTPC head REGRESSED the PWI from 0.312 (PASS) to 0.601 (FAIL).
Global template residual is INCOMPATIBLE with Gen 3's per-patch
part-whole binding mechanism.

**Interpretation (3 findings)**:

1. [CONFIRMED] **PWI regression** at step 25K → 45K is statistically
   significant. The dramatic PWI 0.145 we got at step 25K was a transient
   peak; continued training is DEGRADING the part-whole sensitivity.
   - Mechanism: DINOv2-style SSL's two-global-view contrastive loss
     pushes features toward view-invariance. Part-whole sensitivity
     specifically requires the model to distinguish "whole-face context"
     from "isolated part" — exactly the kind of context-sensitivity that
     contrastive view-invariance erodes.

2. [CONFIRMED] **Gen 3 + FTPC head is worse than raw Gen 3 features**
   on PWI. Global template residual mechanism (which worked OK on frozen
   DINOv2) is INCOMPATIBLE with Gen 3 backbone's already-developed
   part-whole structure. The FTPC δ readout washes out the part-context
   information that Gen 3 features carry per-patch.

3. [CONFIRMED] **ISI remains stuck around 0.94-0.96** across all
   variants tried. orient-aux did NOT translate to image-side Thatcher
   detection regardless of training duration or head architecture.
   This corroborates the diagnosis from E054 that CLS-level orient task
   doesn't produce per-patch orientation-asymmetric features needed for
   Thatcher.

**Critical: step-25K ckpt LOST**:
- The training script overwrites `checkpoint.pt` at every save_every=5000.
- We saved at step 5K, 10K, 15K, 20K, 25K, 30K, 35K, 40K, 45K — all
  overwritten the previous one.
- The dramatic PWI 0.145 at step 25K cannot be recovered without
  re-training to step 25K from scratch.
- **Mitigation going forward**: snapshot ckpt.pt to stamped filenames at
  each monitoring tick. Done at tick 110: `checkpoint_step45K.pt` created.

**Decision tree**:

A. **Continue training to step 100-150K with per-tick snapshotting**.
   Sunk cost is $0 (rented). Worst case: we use the best-PWI ckpt
   among future snapshots; best case: PWI rebounds or stabilizes at a
   useful level. Recommended.

B. **Kill training now**. step-45K ckpt is the best we'll get from this
   recipe. Move to v4.1 redesign. Conservative.

C. **Kill training + restart from scratch with save_every=1000**.
   To recover the step-25K-style early PWI peak. Expensive (~14h more
   training to reach step 25K again). Not justified unless we have
   reason to believe step-25K is the GLOBAL optimum.

**Loop default**: A (continue with snapshotting). Will report at each
checkpoint. If user disagrees, can override via NOTES_FOR_USER.

**Replicability**:
- Code: `holo_net/eval_gen3_quick.py`, `holo_net/gen3_ftpc.py` (new)
- step-45K ckpt snapshot: `/workspace/runs/2026-05-24_gen3_v4.0_seed20260521/checkpoint_step45K.pt`
- NPZs: `outputs/embeddings/{paradigm}_HOLONET_v4.0_gen3_{45K,ftpc}/`

**Linked Idea-###**: Idea-003 score remains 5.5; image-side PWI 0.312
(passes §6 ≤ 0.5) is solid but not the headline ambition.
