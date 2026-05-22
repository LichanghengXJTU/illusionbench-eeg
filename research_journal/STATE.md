# State

**Tick #**: 52
**Last updated**: 2026-05-22 ~13:05 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net is the headline; the full
architecture is training (E042) and this tick built the evaluation harness so a
result can be computed the moment a checkpoint is ready.
**Last action**: Tick 52 — wrote `holo_net/eval_extract.py` (loads a HOLO-Net
checkpoint, emits one repo-standard NPZ per brain-region layer so the existing
`analysis/compute_metrics.py` computes Thatcher ISI unchanged); located the four
IllusionBench stimulus manifests on the server.
**Last action outcome**: artifact delivered (eval harness). Training healthy.
**Running tasks** (on server, H100 80GB):
  - PID 44488 — minimal-mode run (E040 / "minus all bio components" ablation),
    step ~22.5K/30K, identity ~3.3. checkpoint.pt auto-saved every 5K steps.
  - PID 47222 — **FULL HOLO-Net run (E042)**, step ~2K/30K, identity 13.2→12.2,
    predcode 4.2→1.2, descending past warmup. Output `/workspace/holo_net_full_v1/`.
**Stuck streak**: 0
**Planned next action** (tick 53): run `eval_extract.py` on the best available
minimal checkpoint → `compute_metrics.py` → first per-layer Thatcher ISI
(E041 — preliminary, partial-training). Also locate the composite-CSI /
part-whole-PWI metric code (not in `analysis/`).
**Confidence in current best idea**:
  - **Idea-003 (HOLO-Net)**: ~7.5/10 — architecture trains; awaiting the
    falsification-criteria evaluation.
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
  `/workspace/holo_net_full_v1/` (full/E042)
- Stimulus manifests (server): `data/stimuli_ffhq{,_composite,_partwhole,_randombbox}/thatcher_manifest.csv`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
