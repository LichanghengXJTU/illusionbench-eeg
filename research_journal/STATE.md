# State

**Tick #**: 77
**Last updated**: 2026-05-23 ~13:30 (Asia/Hong_Kong)
**Current focus** (one sentence): EDA done + ImageNet-1K download started in
background; `data_v2.py` next tick.
**Last action**: Tick 77 — invoked **`experiment-run` SKILL**. Caught + fixed
a real bug in 3-step smoke at batch 256: template T shape (D, 7, 7) only
matches AFP_spatial on global views (224→7²); local views (96→3²) caused
shape mismatch in `δ = afp_spatial - T`. Fix (`model_v2.py`): compute δ only
when AFP shape matches T's; local views skip δ (the PC loss + template EMA
were already global-only by design). Smoke re-ran: L_dino 10.65 / L_pc 1.30 /
26-of-256 faces (10%) / T_ema norm 50.71. **Launched full 100-ep SSL run**:
`/workspace/runs/2026-05-23_holonet-v2-ftpc_seed20260521/` (snapshot files
staged: hypothesis.md, cmd.txt, env.txt, git.txt). 500,400 total steps.
**Last action outcome**: training RUNNING; GPU 27.4 G / 80 G at 100% util;
19 procs.
**Running tasks** (on server, H100 80GB):
  - **HOLO-Net v2.1 SSL on ImageNet-1K** —
    `/workspace/runs/2026-05-23_holonet-v2-ftpc_seed20260521/`. 100 ep × 5004
    steps/ep = 500,400 steps; wall-clock ETA TBD after first 100 steps.
**Stuck streak**: 0
**Planned next action** (tick 78, ~30 min): first monitoring tick — read
`train_log.jsonl` for steady-state step_time → real ETA; verify L_dino is
descending and L_pc is non-NaN; check T_ema norm growing as expected. If
steady-state too slow (>3 s/step), consider reducing local crops / batch.
Subsequent ticks: hourly monitor, then 1× intermediate eval at epoch ~25
(`eval_falsification.py` on early checkpoint, partial IllusionBench signal
check).
**User directive (tick 75)**: 全权 / 持续 loop / 主线 = EEG 关联 + 模型设计 +
公平严格 / 主动调用科研 SKILLS.
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
