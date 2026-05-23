# State

**Tick #**: 79
**Last updated**: 2026-05-23 ~14:35 (Asia/Hong_Kong)
**Current focus** (one sentence): EDA done + ImageNet-1K download started in
background; `data_v2.py` next tick.
**Last action**: Tick 79 — monitored run 2 at step 3200: L_dino STILL collapsed
to 11.09 = ln(65536), L_pc fluctuating 0.15-3.0 (T learning but oscillates),
T_ema shrank 50→19 (slower than run 1's 50→7, the PC detach helped the worst
case but not the overall collapse). DIAGNOSIS: out_dim 65536 (DINOv2 default
for ViT-L) is too many prototypes for our CORnet-S-style ~25M backbone →
student can't sustain a discriminative distribution → collapses to uniform.
FIX: reduce `ssl_out_dim` 65536 → **4096** (the lower-capacity DINOv1 ViT-S
config). Uniform collapse baseline drops to ln(4096) = 8.32. Killed run 2,
scp'd model_v2.py fix, launched **run 3** (tag `holonet-v2-out4k`). step 150
/ 1.8 min: L_dino 8.46 (vs old 11.27; just above 8.32 baseline), L_pc 1.11
(held), T_ema 45 (stable). Initial signs much better.
**Last action outcome**: run 2 killed (2nd collapse, expected, validated the
hypothesis); run 3 running. If THIS also collapses → option-D pivot
(pre-trained DINOv2 backbone) presented to user.
**Running tasks** (on server, H100 80GB):
  - **HOLO-Net v2.1 SSL "out4k"** —
    `/workspace/runs/2026-05-23_holonet-v2-out4k_seed20260521/`. step ~150,
    L_dino 8.46 / L_pc 1.11, GPU 24.2 G (smaller head = less mem).
**Stuck streak**: 0 (each tick produces actionable signal + a real fix)
**Planned next action** (tick 80, ~30 min): monitor run 3 at step ~3000.
DECISIVE: does L_dino descend below 8.32 (discriminative) or settle at 8.32
(collapsed). If healthy → run 3 is the production run, continue. If
collapsed AGAIN → present option D (pre-trained DINOv2 backbone + FTPC head
+ template) to user as the pragmatic pivot; the FTPC novelty is preserved
without the from-scratch-SSL collapse risk.
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
