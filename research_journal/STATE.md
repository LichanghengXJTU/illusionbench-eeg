# State

**Tick #**: 54
**Last updated**: 2026-05-22 ~14:25 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net — found and fixed a real
OrientationGate bug (global-pooling made it orientation-blind), relaunched the
full training as run v2.
**Last action**: Tick 54 — diagnosed Q015 (orientation loss flat at chance for
6450 steps). Confirmed the cause: `OrientationGate` GAP'd `afp_spatial` before
classifying, and a mean is flip-invariant. Fixed `model.py` (4×4 pool instead of
1×1), added a training `--seed` to `train.py`, killed E042 run v1, relaunched v2.
**Last action outcome**: CONFIRMED bug + fix (GAP flip-diff 0.000000 vs 4×4
flip-diff 0.43). v2 training cleanly: step ~300, identity ~10.6, 0.38 s/step.
**Running tasks** (on server, H100 80GB):
  - PID (v2) — **FULL HOLO-Net run v2 (E042)**, `/workspace/holo_net_full_v2/`,
    OrientationGate fixed, `--seed 20260521`. step ~300/30000.
  - minimal run (E040) — was at step 29K last poll; expected finished (GPU freed
    → v2 runs at 0.38 s/step). Confirm + locate final checkpoint next tick.
**Stuck streak**: 0
**Planned next action** (tick 55): (a) confirm the `orientation` loss now
descends below ln 2 in v2 — the real proof the gate fix works; (b) re-eval the
minimal FINAL (step-30000) checkpoint on Thatcher; (c) locate the composite-CSI
/ part-whole-PWI metric code (not in `analysis/`).
**Confidence in current best idea**:
  - **Idea-003 (HOLO-Net)**: ~7.5/10 — architecture trains; a load-bearing bug
    (OrientationGate) caught and fixed before it wasted the full run.
  - **Idea-001**: 9.3/10 — groundwork, stable.

## Strategic note (user directive 2026-05-22)
Idea-001 is theoretical groundwork; **Idea-003 (HOLO-Net) is the headline and
the loop's primary focus**. Full autonomy on Idea-003 direction. Build / train /
validate HOLO-Net continuously. If compute becomes the blocker, write a NEED to
`NOTES_FOR_USER.md` — the user will rent more GPU. Keep looping; commit+push and
report status every tick.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (plain dir, not git)
- HOLO-Net runs: `/workspace/holo_net_stage2/` (minimal/E040),
  `/workspace/holo_net_full_v2/` (full, fixed/E042); `holo_net_full_v1/` dead
- Stimulus manifests (server): `data/stimuli_ffhq{,_composite,_partwhole,_randombbox}/thatcher_manifest.csv`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
