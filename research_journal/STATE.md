# State

**Tick #**: 58
**Last updated**: 2026-05-23 ~00:55 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net — corrected a misjudgment
(the identity-loss "bounce" is normal, not failure); launched the full model v4
to run to completion without premature judgment.
**Last action**: Tick 58 — fetched the minimal run's full trajectory; it bounces
identity 10-13.6 until ~step 8000 then descends to ~1.5. This means v1/v2/v3 were
killed mid-bounce, and the tick-55/56 "failure" diagnoses were premature. Killed
the mis-scoped diagnostics, launched v4 (full HOLO-Net, 30000 steps).
**Last action outcome**: v4 training (step ~250, identity 10.5, 0.41 s/step,
full GPU). Verdict deferred to step ~15000+.
**Running tasks** (on server, H100 80GB):
  - **v4 — full HOLO-Net (E042 / Q017)** — `/workspace/holo_net_full_v4/`,
    OrientationGate-4×4 + FFA/ATL-additive fixes, all components on, seed
    20260521, 30000 steps (~3.3 h). step ~250.
**Stuck streak**: 0
**Planned next action** (tick 59): v4 will still be mid-bounce — do the deferred
non-blocking work instead: re-eval the minimal FINAL checkpoint on Thatcher
(E041 update), and locate the composite-CSI / part-whole-PWI metric code. Only
judge v4 at step ~15000+.
**Confidence in current best idea**:
  - **Idea-003 (HOLO-Net)**: ~7/10 — the "3 failed runs" narrative was a
    misjudgment; whether the full model trains is genuinely still open (Q017),
    being answered properly by v4 now.
  - **Idea-001**: 9.3/10 — groundwork, stable.

## Process lesson (tick 58)
Never judge a training run without the matched-config reference trajectory.
Ticks 55-57 misread the normal early identity bounce as failure because the
minimal-run reference (which also bounces 10-13.6 for ~8000 steps) had not been
pulled. Cost: 3 premature relaunches. The two model.py fixes from that period
are still sound (OrientationGate 4×4 = a real confirmed bug fix; FFA/ATL
additive wiring = a sound design) and are retained in v4.

## Deferred (non-blocking)
- Re-eval minimal FINAL checkpoint on Thatcher; locate composite-CSI /
  part-whole-PWI metric code (not in `analysis/`).
- OFA / PFC-Gist face-vs-nonface heads see only faces (no ImageNet non-face mix
  in `data.py`).

## Strategic note (user directive 2026-05-22)
Idea-001 is theoretical groundwork; **Idea-003 (HOLO-Net) is the headline and
the loop's primary focus**. Full autonomy on Idea-003 direction. If compute
becomes the blocker, write a NEED to `NOTES_FOR_USER.md`. Keep looping;
commit+push and report status every tick.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (plain dir, not git)
- HOLO-Net: `holo_net_stage2/` (minimal/E040, done, identity→1.5),
  `holo_net_full_v4/` (full/E042/Q017); v1/v2/v3 + diagA/B/D dead
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
