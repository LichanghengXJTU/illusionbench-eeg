# E052 — HOLO-Net v3.0 dual asymmetric templates (Gen 2 variant 1)

**Hypothesis** (pre-registered E051 + this experiment docstring):
Two global FTPC templates T_upright (EMA on AFP of detected face samples)
and T_inverted (EMA on AFP of vflip of same face samples). ffa = concat(
δ_up, δ_inv) → 768-d. Predicted ISI 1.05-1.15 (slight pro-Thatcher from
asymmetric template behavior); fallback if ISI ≤ 1.2 → escalate to Gen 3.

**Method**:
- 2 templates (D, 16, 16) each (T_up + T_inv)
- Template-only training: 500 batches × 256, 13,530 face samples
- Per-batch: forward x and vflip(x); T_up EMA on AFP(x)[face_mask],
  T_inv EMA on AFP(vflip(x))[face_mask]
- Inference: δ_up = AFP_pooled - T_up_pooled; δ_inv = AFP_pooled - T_inv_pooled;
  ffa = concat → 768-d
- 3.4 min wallclock on H100; final T_up norm 297.13, T_inv norm 323.77

**Result (per-layer, pixel-corrected)**:

| Layer | ISI | CSI | PWI | ISIrbox | pass/4 |
|---|---:|---:|---:|---:|---:|
| afp (raw DINOv2) | 0.908 | 1.282 | 0.397 ✓ | 0.662 ✓ | 2/4 |
| δ_up only | 0.904 | 1.295 | 0.456 ✓ | 0.658 ✓ | 2/4 |
| δ_inv only | **0.633** | 1.259 | 0.369 ✓ | 0.448 ✓ | 2/4 |
| **ffa = concat[δ_up, δ_inv]** | **0.753** | 1.275 | 0.408 ✓ | 0.541 ✓ | **2/4** |

**§6 verdict at FFA**:
- ISI = 0.753 — need ≥ 3.0 → **FAIL** (more anti-Thatcher than v3's 0.901!)
- CSI = 1.275 — need ≥ 1.5 → **FAIL**
- PWI = 0.408 — need ≤ 0.5 → **PASS** (slightly better than v3's 0.446)
- ISIrbox = 0.541 — need ≤ 1.5 → **PASS** (better margin than v3's 0.654)
- ALL FOUR: **FAIL (2/4 PASS)**

**Interpretation**:

[CONFIRMED — and pre-registered prediction WRONG] **v3.0 makes ISI WORSE,
not better**. The dual-template design produced ISI 0.753 (anti-Thatcher
direction stronger), contradicting my pre-registered prediction of 1.05-1.15.

**Mechanism diagnosis**:
- T_inverted template is trained on vflipped-face AFP samples (inverted
  face features as seen by frozen DINOv2)
- δ_inv = AFP - T_inv_pooled measures how unlike-inverted-face the input is
- For Thatcher pairs:
  - V1 upright_normal: AFP unlike T_inv → δ_inv large
  - V2 upright_thatched: AFP slightly more unlike T_inv → δ_inv slightly larger
  - V3 inverted_normal: AFP matches T_inv → δ_inv small
  - V4 inverted_thatched: AFP slightly off T_inv (parts disturbed) → δ_inv
    medium (bigger swing than V1→V2 because inverted-tuned template detects
    deviations from inverted-normal more sensitively)
- ⇒ d(V3, V4) in δ_inv > d(V1, V2) in δ_inv
- ⇒ ISI = d(V1,V2)/d(V3,V4) < 1 (anti-Thatcher direction reinforced)

T_inv has higher norm (323.77 vs T_up 297.13) → δ_inv channel dominates the
concat readout → drags the headline ISI further below 1.

[CONFIRMED] **Both templates inherit DINOv2's orientation-invariant feature
structure**. They don't capture orientation-asymmetric face processing
because the backbone doesn't have such processing.

**Gen 2 escalation trigger FIRED**:

Pre-registered trigger (NOTES_FOR_USER 22:20, MILESTONE-006):
> Gen 1-2 saturated with ISI ≤ 1.2 → request 8× H100 for Gen 3

Status:
- Gen 1: best ISI 1.082 (v2.4a α=2.0, but PWI fails); other variants ≤ 1.01
- Gen 2 v3.0: best ISI 0.904 (δ_up only); ffa 0.753 (regression)
- Cumulative: NO variant achieves ISI > 1.2

**ESCALATION TRIGGERED**. Writing NEED-002 for 8× H100 compute box. Loop
proceeds with: (a) write Gen 3 training code on current 1× H100 in parallel
(reusable when box arrives); (b) document v3.0 result; (c) commit + push.

**Replicability**:
- Random seed: 20260521
- Code: `holo_net/model_v2_dinov2_dualtemplate.py`, `train_v2_dinov2_dualtemplate.py`,
  `eval_falsification_dinov2_dual.py`
- Checkpoint: `/workspace/runs/2026-05-23_holonet-v3.0-dualtemplate_seed20260521/checkpoint.pt`
- NPZs: `/workspace/illusionbench-eeg/outputs/embeddings/{paradigm}_HOLONET_v3.0_dual/`

**Linked Idea-###**: Idea-003 unchanged at 5.5 (negative result expected;
the path forward is Gen 3, not more Gen 2 variants).
