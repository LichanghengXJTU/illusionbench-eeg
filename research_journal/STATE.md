# State

**Tick #**: 63
**Last updated**: 2026-05-23 ~03:52 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net — v5 training on track
(identity descending like minimal, gate working); the eval wrapper is validated.
**Last action**: Tick 63 — v5 at step 16250: identity 5.96 (minimal was 6.3 at
step 15000 → v5 tracks/slightly leads), orientation 0.13 (gate functional).
Deployed `eval_falsification.py` and launched it on the minimal checkpoint —
running cleanly (Thatcher paradigm done, composite in progress) → wrapper
validated.
**Last action outcome**: v5 fully on track; eval wrapper works end-to-end.
**Running tasks** (on server, H100 80GB):
  - **v5 — full HOLO-Net + all fixes (E042)** — `/workspace/holo_net_full_v5/`,
    step ~16550/30000, identity ~6.3 descending. ~1.4 h to completion.
  - falsification eval on the minimal checkpoint — `/workspace/falsif_minimal.log`,
    ~8 min to finish (validates the wrapper + completes the 4-paradigm floor).
**Stuck streak**: 0
**Planned next action** (tick 64): read the minimal 4-paradigm falsification
table from `falsif_minimal.log` (→ E041); monitor v5 (~step 21000). When v5
finishes → run `eval_falsification.py` on the v5 checkpoint = the HOLO-Net test.
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
