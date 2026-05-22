# State

**Tick #**: 55
**Last updated**: 2026-05-22 ~15:05 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net — the OrientationGate fix
worked (orientation loss 0.69→0.26) but exposed an identity-collapse; fixed the
FFA/ATL wiring and relaunched the full run as v3.
**Last action**: Tick 55 — diagnosed v2's stalled identity loss (bounced 11-14,
no descent): the now-functional gate zeroed the FFA input for inverted faces,
and `atl_input = ffa_out` was the sole identity path → inverted faces
unidentifiable. Fixed `model.py` (`atl_input = afp_pooled + gate·ffa_out`),
killed v2, relaunched v3.
**Last action outcome**: v3 training cleanly (step ~250, identity 10.65,
0.34 s/step). Identity/orientation descent to be confirmed next tick.
**Running tasks** (on server, H100 80GB):
  - **FULL HOLO-Net run v3 (E042)** — `/workspace/holo_net_full_v3/`, fixed
    OrientationGate (4×4 pool) + fixed FFA/ATL wiring (additive gated holistic
    term), `--seed 20260521`. step ~250/30000. GPU to itself (minimal done).
**Stuck streak**: 0
**Planned next action** (tick 56): confirm v3 — identity loss descends (the fix
works) AND orientation loss descends below ln 2. Then re-eval the minimal FINAL
(step-30000) checkpoint on Thatcher (E041 update); locate the composite-CSI /
part-whole-PWI metric code (not in `analysis/`).
**Confidence in current best idea**:
  - **Idea-003 (HOLO-Net)**: ~7.5/10 — two real bugs found and fixed
    (OrientationGate flip-blindness; gate-on-sole-identity-path collapse);
    architecture now training cleanly. Awaiting the falsification result.
  - **Idea-001**: 9.3/10 — groundwork, stable.

## Known minor issues (non-blocking, deferred)
- OFA / PFC-Gist face-vs-nonface heads see only faces (`is_face` always 1 —
  the design's 10% ImageNet non-face mix is not implemented in `data.py`).
  Their losses are near-degenerate; weight is small (0.05). Revisit if needed.

## Strategic note (user directive 2026-05-22)
Idea-001 is theoretical groundwork; **Idea-003 (HOLO-Net) is the headline and
the loop's primary focus**. Full autonomy on Idea-003 direction. Build / train /
validate HOLO-Net continuously. If compute becomes the blocker, write a NEED to
`NOTES_FOR_USER.md` — the user will rent more GPU. Keep looping; commit+push and
report status every tick.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (plain dir, not git)
- HOLO-Net runs: `/workspace/holo_net_stage2/` (minimal/E040, done),
  `/workspace/holo_net_full_v3/` (full, both fixes/E042); v1, v2 dead
- Stimulus manifests (server): `data/stimuli_ffhq{,_composite,_partwhole,_randombbox}/thatcher_manifest.csv`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
