# State

**Tick #**: 72
**Last updated**: 2026-05-23 ~10:00 (Asia/Hong_Kong)
**Current focus** (one sentence): User confirmed v2 build + directed a bigger
redesign (no gates, illusion-transmission mechanism, normal tasks still
strong) — wrote `HOLONET_V2_DESIGN.md` proposing Face-Template Predictive
Coding (FTPC).
**Last action**: Tick 72 — design doc for **HOLO-Net v2.0 (FTPC)**: removes
all gates and the recurrent-FFA-binding; replaces with a learnable
canonical-upright-face template + an anomaly-stream (fSTS-analogue) that
carries the template-mismatch field δ as the explicit illusion signal;
anomaly back-modulates AFP additively (α init 0 → identity untouched). Five
open design choices flagged with proposed defaults.
**Last action outcome**: a coherent v2 framework grounded in the diagnostic
(addresses all 3 v1 implementation gaps in one mechanism) and the user's
constraints (no gates / transmission mechanism / normal-task safe / not
patches).
**Running tasks** (on server, H100 80GB): none.
**Stuck streak**: 0
**Planned next action** (tick 73): unless the user flags a concern on any
of the 5 open design choices, **start coding** `holo_net/model_v2.py`
(skeleton + sanity tests) using the documented defaults. Then tick 74 =
anomaly augmentation in `data.py`; tick 75 = launch training; tick 77 = eval.
**Cadence**: ~25-30 min so the user has a window to weigh in before code lands.
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
