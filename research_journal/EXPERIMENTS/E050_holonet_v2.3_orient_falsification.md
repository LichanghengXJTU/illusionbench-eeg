# E050 — HOLO-Net v2.3 orientation-aware FTPC §6 falsification (Gen 1 variant 2)

**Hypothesis** (pre-registered in E049 §"v2.3 design"):
Global FTPC template T (from v3) + a second DINOv2 forward on vflip(x);
ffa = concat(δ_global, δ_orient) → 768-d. Predicted ISI 1.5-3.0 (pro-Thatcher),
CSI ≥ 1.3 (recover from v2.2 regression), PWI ≤ 0.5 (recover), ISIrbox PASS.

**Method**:
- Same v3 checkpoint (face_template T) — NO retraining
- 2× DINOv2 forward at inference (x and vflip(x))
- δ_global_pooled = AFP_pooled - T_pooled (the v3 readout)
- δ_orient_pooled = AFP_vflip_pooled - T_pooled (NEW orient channel)
- ffa = concat → 768-d

**Result (pixel-corrected, FFA = δ_concat 768-d)**:

| Layer | ISI | CSI | PWI | ISIrbox |
|---|---:|---:|---:|---:|
| v1/v2/v4/mfp/afp/atl (aliases) | 0.908 | 1.282 | 0.397 | 0.662 |
| **ffa (v2.3)** | **1.010** | **1.317** | **0.543** | **0.996** |

**§6 verdict at FFA**:
- ISI = 1.010 — need ≥ 3.0 → **FAIL** (but crossed 1.0 — no anti-direction)
- CSI = 1.317 — need ≥ 1.5 → **FAIL** (small improvement)
- PWI = 0.543 — need ≤ 0.5 → **FAIL** (lost v3's PASS by 0.04)
- ISIrbox = 0.996 — need ≤ 1.5 → **PASS** (margin shrank from v3's 0.654)
- ALL FOUR: **FAIL (1/4 PASS)**

**Diff vs v3 (E045) FFA**:

| Index | v3 ffa | v2.3 ffa | Δ | Direction |
|---|---:|---:|--:|---|
| ISI | 0.901 | 1.010 | +0.109 | Better (no more anti-Thatcher) |
| CSI | 1.300 | 1.317 | +0.017 | ≈ neutral |
| PWI | 0.446 (PASS) | 0.543 (FAIL) | +0.097 | **Worse** (lost PASS) |
| ISIrbox | 0.654 | 0.996 | +0.342 | **Worse** (margin shrank) |

**Interpretation** (4 observations):

1. [CONFIRMED] **Analytical prediction was right in DIRECTION but undershoot
   in MAGNITUDE**: ISI moved from anti-Thatcher (0.901) to neutral (1.010),
   but did not reach the predicted 1.5-3.0 range.

2. [CONFIRMED] **PWI and ISIrbox regressed**: the orient channel adds
   substantial NOISE that's unrelated to face-specific processing. vflip(x)
   moves the face from one half of the image to the other; DINOv2 patches
   at fixed positions then see very different image content, contributing
   variance to every paradigm — including random-bbox (where the bbox
   moves) and part-whole (where the part-context relationship changes
   in vflipped view).

3. [CONJECTURE] **Orient channel ≈ spatial-layout-change channel, not
   face-orientation channel**. The signal we wanted (does the face have
   orientation-asymmetric processing?) is contaminated by the broader
   signal (does the image look different when flipped?). For non-face
   regions, vflip(x) creates large feature differences purely from
   spatial-layout changes.

4. [CONFIRMED] **3 Gen 1 variants now sit at**:
   - v3 (global FTPC): ISI 0.901 (anti)
   - v2.2 (part-aware FTPC): ISI 0.981 (near zero asymmetry)
   - v2.3 (global + orient): ISI 1.010 (crossed to neutral, +0.11 from v3)
   
   Cumulative best ISI from Gen 1 head-only changes ≈ **1.01**.

**v2.4 design (locked in)**:

Two variants in parallel next tick (same eval pipeline):

- **v2.4a Weighted concat**: ffa = concat(δ_global, α × δ_orient), sweep
  α ∈ {0.25, 0.5, 0.75}. Reduces orient-channel noise dominance, may
  recover PWI/ISIrbox while keeping ISI signal.

- **v2.4b Face-restricted orient**: use v2.2's 4 part regions to compute
  orient channel ONLY over face-region patches:
  `δ_orient_face = concat over k of mean_pool(AFP_vflip[region_k] - T_part_k)`
  where T_part_k are the v2.2 trained part templates. The orient channel
  is then face-specific rather than image-wide.

**Meta-judgment on Gen 1**:

If v2.4a or v2.4b lift ISI past 1.5, Gen 1 has reach. If neither lifts ISI
past 1.2 (i.e. just past neutral), **Gen 1 is exhausted** — frozen DINOv2
backbone lacks the orientation-asymmetric processing capacity. Then:
- Jump to Gen 2 (training-time signal: orientation classification aux task
  during template training)
- Or jump to Gen 3 (from-scratch backbone retrain with no-vflip aug)

The latter triggers the compute escalation criterion (write NEED-002 for
8× H100 box).

**Replicability**:
- Random seed: 20260521
- Code: `holo_net/model_v2_dinov2_orient.py`, `eval_extract_dinov2_orient.py`,
  `eval_falsification_dinov2_orient.py`
- Checkpoint: REUSES v3's `/workspace/runs/2026-05-23_holonet-v2-dinov2_seed20260521/checkpoint.pt`
- NPZs: `/workspace/illusionbench-eeg/outputs/embeddings/{paradigm}_HOLONET_v2.3_orient/`

**Linked Idea-###**: Idea-003 still at 5.5. v2.3 confirms partial directional
progress on ISI but no breakthrough; v2.4 next.
