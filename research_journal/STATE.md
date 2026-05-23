# State

**Tick #**: 80 (completed)
**Last updated**: 2026-05-23 ~19:15 (Asia/Hong_Kong)
**Current focus** (one sentence): HOLO-Net v3 (frozen DINOv2 + global FTPC)
**§6 verdict in: 2/4 PASS** (PWI 0.446, ISIrbox 0.654 PASS; ISI 0.901, CSI
1.300 FAIL) — improvement over v1's 1/4 but still FAIL overall; δ-residual
adds no significance over raw DINOv2 → mechanism must be part-aware (v2.2);
in parallel, pivot to EEG-decoder Stage 3 (frozen DINOv2 → THINGS-EEG2).
**Last action**: Tick 80 — template training completed (3.4 min wallclock,
13,586 face samples, T_ema 309.21 → 287.88 monotonic). Wrote
`eval_extract_dinov2.py` + `eval_falsification_dinov2.py` (thin variants
loading `HOLONetV2Dinov2`, ImageNet normalization). Local sanity ✓. SCP to
server, ran full 4-paradigm eval. Verdict logged to E045. Updated journal.
**Last action outcome**: **partial — best HOLO-Net result so far (2/4 vs
v1's 1/4), but does NOT meet pre-registered §6** (FAIL on Thatcher + Composite).
The δ residual against a global template is statistically indistinguishable
from raw DINOv2 features (CIs overlap on all 4 paradigms). FTPC needs
part-awareness to encode the local feature-orientation mechanism (Psalta 2014).
**Running tasks** (on server, H100 80GB): none — server idle.
**Stuck streak**: 0 (the result is informative and points to v2.2)
**Planned next action** (tick 81, ~20 min): **TWO-TRACK** —
  - **Track A (EEG)**: Start building EEG-decoder Stage 3 — frozen DINOv2 →
    THINGS-EEG2 (the headline EEG deliverable the user explicitly asked for).
    Locate THINGS-EEG2 prep on server; if not present, download; write a
    minimal ATM-style ridge decoder skeleton targeting `afp_pooled`.
  - **Track B (HOLO-Net v2.2)**: Sketch the part-aware FTPC design — K=4
    sub-templates {T_eyes, T_nose, T_mouth, T_chin} at fixed spatial sub-
    regions of the 16×16 patch grid, per-region δ aggregated by upright-vs-
    inverted concordance. Implement & re-eval next tick.

## Confidence in current best ideas
- **Idea-001** (IllusionBench-EEG): **9.5/10** — strongly reinforced. 3 distinct
  HOLO-Net instantiations now all fail §6 simultaneously, adding architecture-
  side corroboration to the training-objective thesis from 3 angles (identity
  loss, SSL-from-scratch-collapse, frozen-SSL+global-template). The benchmark
  itself is paper-ready; the v1+v2-collapse+v3 negatives are clean mechanistic
  evidence of why current vision SOTA can't satisfy it.
- **Idea-003** (HOLO-Net positive): **5.5/10** — slight up (was 5.0). v3
  (DINOv2-FTPC) gives **2/4 pass** vs v1's 1/4 (PWI flipped from 2.20 FAIL to
  0.446 PASS — large reversal in the right direction). Composite CSI 1.300 is
  within striking distance of the 1.5 threshold. The headline ambition is still
  alive if v2.2 (part-aware FTPC) lifts the Thatcher signal.

## The HOLO-Net result, two lines
v1 (CORnet + AdaFace identity): NO illusions (ISI/CSI/PWI all wrong direction).
v3 (frozen DINOv2 + global FTPC): PWI + random-bbox PASS; Thatcher ANTI-direction,
Composite close but FAIL. Diagnosis: global template can't isolate the
local-feature-orientation signal Psalta 2014 identifies as the Thatcher locus.

## Strategic note
The user's explicit #1 priority is the EEG decoder (主线 = EEG 关联). HOLO-Net
v3 makes the DINOv2 backbone production-ready for the EEG side; even if v2.2
fails, we now have a defensible Stage 3 EEG decoder built on a SOTA SSL prior
that already moves PWI from FAIL to PASS — that's a publishable IEEE-style
EEG-AI paper independent of the HOLO-Net headline.
**User directive (tick 75)**: 全权 / 持续 loop / 主线 = EEG 关联 + 模型设计 +
公平严格 / 主动调用科研 SKILLS.
**Confidence in current best idea**:
  - **Idea-001** (IllusionBench-EEG): **9.3/10** — unaffected, and strengthened:
    the HOLO-Net negative result is architecture-side corroboration of its
    training-objective thesis.
  - **Idea-003** (HOLO-Net): **5.0/10** — the pre-registered "first
    paradigm-consistent model" ambition is refuted; architecture trains fine;
    a redirect (objective re-aim, or consolidate as a mechanistic negative)
    is open.

## The HOLO-Net result in one line
A maximally bio-faithful face architecture, trained on face identity, shows NO
Thatcher / composite illusion (ISI/CSI ≈ 1.0) — exactly like FaceNet (E004).
Bio-fidelity of architecture does not substitute for the training objective.

## Strategic note (user directive 2026-05-22)
Idea-003 (HOLO-Net) was designated the headline. The v5 refutation means that,
as a positive headline, HOLO-Net v1.0 does not deliver — see FLAG-002 for the
three honest options. Loop continues; full autonomy on direction; will not stop.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (plain dir, not git)
- HOLO-Net: `holo_net_stage2/` (minimal/E040-E041), `holo_net_full_v5/`
  (full, trained, E042/E044); v1-v4 dead
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
