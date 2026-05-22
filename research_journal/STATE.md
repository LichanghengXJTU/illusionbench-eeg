# State

**Tick #**: 64
**Last updated**: 2026-05-23 ~04:28 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net — v5 ~70% trained and on
track; the minimal ablation floor is confirmed; next is v5's falsification test.
**Last action**: Tick 64 — read the minimal 4-paradigm falsification table
(`eval_falsification.py` validated end-to-end). v5 at step 21650, identity 3.94.
**Last action outcome**: minimal ablation floor confirmed — FFA layer ISI 0.975
/ CSI 1.192 / PWI 0.779 / random-bbox 1.083 → fails 3 of 4, "ALL FOUR: FAIL"
(only the easy random-bbox control passes). v5 on track.
**Running tasks** (on server, H100 80GB):
  - **v5 — full HOLO-Net + all fixes (E042)** — `/workspace/holo_net_full_v5/`,
    step ~21650/30000, identity 3.94 descending (tracks minimal). ~50 min left.
**Stuck streak**: 0
**Planned next action** (tick 65, ~v5 completion): run `eval_falsification.py`
on the v5 final checkpoint = **the HOLO-Net falsification test** (FFA layer:
ISI≥3, CSI≥1.5, PWI≤0.5, random-bbox≤1.5). Compare against the minimal floor.
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
