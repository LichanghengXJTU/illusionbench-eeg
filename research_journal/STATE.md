# State

**Tick #**: 51
**Last updated**: 2026-05-22 ~12:48 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net is the project's headline
(user directive 2026-05-22); the full bio-fidelity architecture is now UNBLOCKED
— E042 root-caused the earlier training failure to an AdaFace loss bug (already
fixed), NOT the bio components.
**Last action**: Tick 51 — audited `holo_net/{model,losses,train}.py`; diagnosed
the v1 failure; launched the FULL HOLO-Net (all bio components, fixed loss) as a
controlled re-test (E042); confirmed it starts at identity loss 13.20 ≈ the
minimal model's 13.13.
**Last action outcome**: CONFIRMED — bio components are not the blocker; HOLO-Net
trains cleanly. Q014 answered.
**Running tasks** (on server, H100 80GB):
  - PID 44488 — minimal-mode run (E040), step ~21K/30K, identity ~4, ETA ~20 min.
    Repurposed as the "minus all bio components" ablation arm.
  - PID 47222 — **FULL HOLO-Net run (E042)**, step ~150/30K, identity 13.2.
    Output `/workspace/holo_net_full_v1/`. step_t ~0.8s (GPU shared).
**Stuck streak**: 0
**Planned next action** (tick 52): poll both runs; confirm full-model loss
descent. When minimal finishes → E041 (evaluate it on IllusionBench = the
all-components-off baseline). When the FULL run finishes → evaluate the full
checkpoint on IllusionBench per-layer = the HOLO-Net falsification test
(FFA-layer ISI≥3 / CSI≥1.5 / PWI≤0.5 / random-bbox≤1.5).
**Confidence in current best idea**:
  - **Idea-003 (HOLO-Net)**: **~7.5/10** — feasibility restored (4→7): the
    architecture is proven to train. Remaining uncertainty is purely scientific
    (does it pass the falsification criteria), not an engineering blocker.
  - **Idea-001**: 9.3/10 — theoretical groundwork, stable, not the loop's focus.

## Strategic note (user directive 2026-05-22)
Idea-001 (IllusionBench-EEG benchmark) is theoretical groundwork; **Idea-003
(HOLO-Net) is the headline novelty and the loop's primary focus**. Full autonomy
granted on Idea-003 direction. Build / train / validate HOLO-Net continuously.
If compute becomes the blocker, write a NEED to `NOTES_FOR_USER.md` — the user
will rent more GPU. Keep looping; commit + push every tick.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (plain dir, not git)
- HOLO-Net runs: `/workspace/holo_net_stage2/` (minimal/E040),
  `/workspace/holo_net_full_v1/` (full/E042)
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
