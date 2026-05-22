# State

**Tick #**: 62
**Last updated**: 2026-05-23 ~03:18 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net — v5 (full model + all
fixes) training on track; the falsification-eval wrapper is now written and
ready for when v5 finishes.
**Last action**: Tick 62 — wrote `holo_net/eval_falsification.py` (orchestrates
eval_extract + compute_metrics over all 4 stimulus sets per layer, pixel-
corrects, prints the FFA-layer §6 verdict). Monitored v5 at step ~10650:
identity ~8.6 (tracking minimal), orientation 0.12 (gate working).
**Last action outcome**: eval wrapper ready; v5 on track.
**Running tasks** (on server, H100 80GB):
  - **v5 — full HOLO-Net + all fixes (E042)** — `/workspace/holo_net_full_v5/`,
    30000 steps, seed 20260521. step ~10650, ~1.9 h to completion.
**Stuck streak**: 0
**Planned next action** (tick 63): monitor v5 (~step 15000 — the identity
verdict point; minimal there was 6.3). When v5 finishes → scp
`eval_falsification.py` to server, run it on the v5 final checkpoint = the
HOLO-Net falsification test.
**Confidence in current best idea**:
  - **Idea-003 (HOLO-Net)**: ~7.5/10 — the architecture is confirmed trainable
    (Q017); the gate fix is well-founded (v2 proved the gate learns when the
    identity gradient is removed). Decisive falsification result ~3 h out.
  - **Idea-001**: 9.3/10 — groundwork, stable.

## Resolved / done
- Q017: the full HOLO-Net trains identity at the minimal run's rate.
- 3-paradigm eval pipeline + minimal ablation floor (E041) — in place.
- v1/v2/v3/v4 narrative: v4 vindicated the tick-58 correction (the "bounce" is
  normal); the only real issues found were the AdaFace loss bug, the
  OrientationGate GAP flip-blindness, and the gate-detach wiring — all fixed.

## Strategic note (user directive 2026-05-22)
Idea-001 is theoretical groundwork; **Idea-003 (HOLO-Net) is the headline and
the loop's primary focus**. Full autonomy on Idea-003 direction. If compute
becomes the blocker, write a NEED to `NOTES_FOR_USER.md`. Keep looping;
commit+push and report status every tick.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (plain dir, not git)
- HOLO-Net: `holo_net_stage2/` (minimal/E040-E041, done), `holo_net_full_v5/`
  (full + all fixes/E042, running); v1-v4 dead
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
