# State

**Tick #**: 78
**Last updated**: 2026-05-23 ~14:05 (Asia/Hong_Kong)
**Current focus** (one sentence): EDA done + ImageNet-1K download started in
background; `data_v2.py` next tick.
**Last action**: Tick 78 — first monitor on run 1 caught COLLAPSE at step
3200 (~33 min): L_dino stuck at 11.09 = ln(out_dim) (uniform student
collapse); L_pc collapsed to ~0.001 (degenerate AFP↔T trivial match); T_ema
shrank 50→7 (AFP magnitude shrinking). DIAGNOSIS: PC loss `‖AFP − T‖²` had
gradient through BOTH AFP and T → trivial co-collapse. FIX (`model_v2.py`):
`face_afp = afp_spatial[face_mask].detach()` in `pc_loss` — predictive-
coding-correct (T learns to predict AFP; AFP shaped only by SSL). Killed
run 1, scp'd fix, launched **run 2** (tag `holonet-v2-pc-detach`) in fresh
dir. step 150 / 1.8 min: L_dino 11.27 (slightly above 11.09 baseline = NOT
yet collapsed), L_pc 1.18 (HOLDING, not collapsing to 0), T_ema 48 stable,
template_norm 3.16 (T learnable training healthily). Fix working at init.
**Last action outcome**: run 1 killed (expected, caught the bug); run 2
running healthily; pre-collapse signals NOT present.
**Running tasks** (on server, H100 80GB):
  - **HOLO-Net v2.1 SSL "pc-detach"** —
    `/workspace/runs/2026-05-23_holonet-v2-pc-detach_seed20260521/`. step ~150,
    L_dino 11.27 / L_pc 1.18, step_t ~0.62s. ETA 3.6 d at current rate.
**Stuck streak**: 0 (tick produced a real bug-fix + re-launch)
**Planned next action** (tick 79, ~30 min): monitor run 2 (~step 3000) —
verify L_dino starts descending below 11.09 (no collapse) AND L_pc stays
in healthy range (0.5-1.5, T learning T_face statistic, not zero), AND
T_ema stays in ~40-60 range (AFP not shrinking). If all 3 healthy → fix
works, continue running. If L_dino still flat → DINO setup itself needs
work (center momentum / teacher temp / different SSL impl).
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
