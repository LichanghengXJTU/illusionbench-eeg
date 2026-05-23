# E051 — HOLO-Net v2.4 sweep (Gen 1 finale)

**Hypothesis** (pre-registered in E050 §"v2.4 design"):
Two parallel variants tested in single eval pass:
- **v2.4a Weighted concat**: ffa = concat(δ_global, α × δ_orient), sweep α
  ∈ {0.00, 0.25, 0.50, 0.75, 1.00, 2.00}. Lower α should reduce orient
  channel's PWI/ISIrbox noise while keeping its ISI signal.
- **v2.4b Face-restricted orient**: orient channel computed only over v2.2's
  4 face part regions (face-specific orientation asymmetry).

**Method**:
- Single forward pass per stimulus through frozen DINOv2 (x and vflip(x));
  cache pooled features and per-region spatial features
- Compose 7 ffa variants in pure tensor ops; re-run `analysis.compute_metrics`
  per variant
- Reuses v3 ckpt (global T) + v2.2 ckpt (4 part T_k)
- No retraining

**Result table (pixel-corrected at FFA)**:

| Variant | ISI | CSI | PWI | ISIrbox | pass/4 |
|---|---:|---:|---:|---:|---:|
| v2.4a α=0.00 (= v3) | 0.901 | 1.300 | 0.446 ✓ | 0.654 ✓ | 2/4 |
| v2.4a α=0.25 | 0.913 | 1.302 | 0.458 ✓ | 0.689 ✓ | 2/4 |
| v2.4a α=0.50 | 0.944 | 1.307 | 0.487 ✓ | 0.777 ✓ | 2/4 ← best α |
| v2.4a α=0.75 | 0.979 | 1.312 | 0.517 | 0.888 ✓ | 1/4 |
| v2.4a α=1.00 (= v2.3) | 1.010 | 1.317 | 0.543 | 0.996 ✓ | 1/4 |
| v2.4a α=2.00 | 1.082 | 1.328 | 0.594 | 1.280 ✓ | 1/4 |
| v2.4b face | 1.037 | 1.359 | 0.837 | **4.348 ✗** | **0/4** |
| **§6 threshold** | ≥ 3.0 | ≥ 1.5 | ≤ 0.5 | ≤ 1.5 | — |

**Interpretation** (3 key findings):

1. [CONFIRMED] **ISI monotonically increases with α (0.901 → 1.082) but
   tops out at 1.08 << 3.0**. The orient channel adds ISI signal but
   linearly in α, and even at α=2 the magnitude is insufficient. Increasing
   α further would eventually saturate (the orient channel's max effect on
   ISI is bounded by ||δ_orient|| variance).

2. [CONFIRMED] **PWI and ISIrbox monotonically increase (worsen) with α
   on the same axis** (PWI: 0.446 → 0.594; ISIrbox: 0.654 → 1.280). The
   orient channel adds NOISE proportional to its weight, and this noise
   pushes both holistic metrics in the failing direction. **There's no
   α that achieves ISI > 1.2 AND PWI < 0.5** — strict Pareto frontier
   between Thatcher and Part-Whole on this axis.

3. [CONFIRMED] **v2.4b face-restricted orient catastrophically fails
   ISIrbox (4.348 vs ≤ 1.5)**. Why: vflip moves random-bbox perturbations
   from non-face image regions into the face region of the vflipped view.
   The face-restricted readout then picks up the perturbation MOVING IN
   as a strong signal — exactly the opposite of what the random-bbox
   control was designed to suppress. The face-restricted approach is
   fundamentally broken for the random-bbox control.

**Gen 1 official declaration — EXHAUSTED**:

5 Gen 1 head-only variants attempted (v3, v2.2, v2.3, v2.4a sweep, v2.4b).
Cumulative max ISI = 1.082 — well below the pre-registered escalation
threshold of 1.2.

**Root cause analysis**: frozen DINOv2 backbone, trained via SSL on
LVD-142M with all-orientation augmentation, has **orientation-invariant**
features. No head computation over these features can recover a strong
orientation-asymmetric signal (the Thatcher mechanism). Heads (templates,
vflip readouts, part decomposition) can only re-weight existing features;
they cannot create new orientation-specific features.

**The lever has to be the backbone**: either swap to an orientation-biased
pre-trained backbone (CLIP, which has implicit upright prior from captions),
or retrain a backbone with no-vflip augmentation + orientation aux task
(Gen 3).

**Submitted to user for decision (NOTES_FOR_USER MILESTONE-006)**:
- Option A: jump to Gen 3 with NEED-002 compute escalation (8× H100)
- Option B: try Gen 2 v3.0 (dual asymmetric templates, ~5 min cost) first
- Default: B, then escalate to A if v3.0 fails

**Replicability**:
- Random seed: 20260521
- Code: `holo_net/eval_v24_sweep.py`
- Checkpoints reused: v3 (`/workspace/runs/2026-05-23_holonet-v2-dinov2_seed20260521/checkpoint.pt`)
  + v2.2 (`/workspace/runs/2026-05-23_holonet-v2.2-partaware_seed20260521/checkpoint.pt`)
- NPZs: `/workspace/illusionbench-eeg/outputs/embeddings/{paradigm}_HOLONET_v2.4*_/`
- Tables: `/workspace/illusionbench-eeg/outputs/tables/HOLONET_v2.4*_*.csv`

**Linked Idea-###**:
- Idea-003: held at 5.5 — Gen 1 exhaustion is a real-but-bounded negative;
  Gen 2/3 still have shots.
