# State

**Tick #**: 60
**Last updated**: 2026-05-23 ~02:10 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net — v4 confirmed the full
model trains (identity tracks the minimal run); fixed the inert OrientationGate
(`gate.detach()`); v5 is the candidate run for the falsification test.
**Last action**: Tick 60 — read v4 at step 11400: identity 8.47, descending,
≈ minimal at the same step → Q017 ANSWERED YES (the full HOLO-Net trains). But
v4's orientation loss stayed flat (gate inert — identity gradient overwhelmed
the orientation CE). Applied `gate.detach()`, killed v4, launched v5.
**Last action outcome**: Q017 = yes (architecture trains). v5 running (step
~250, identity 10.7, 0.37 s/step).
**Running tasks** (on server, H100 80GB):
  - **v5 — full HOLO-Net + gate.detach() (E042)** — `/workspace/holo_net_full_v5/`,
    30000 steps, seed 20260521. step ~250. ~3 h to completion.
**Stuck streak**: 0
**Planned next action** (tick 61): monitor v5 — confirm identity descends
(expected, like v4) AND orientation now descends below ln 2 (the gate.detach()
fix working). Judge at step ~15000+. When v5 completes → 3-paradigm per-layer
eval = the HOLO-Net falsification test (FFA layer: ISI≥3, CSI≥1.5, PWI≤0.5,
random-bbox≤1.5).
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
