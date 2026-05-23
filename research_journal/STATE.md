# State

**Tick #**: 75
**Last updated**: 2026-05-23 ~12:35 (Asia/Hong_Kong)
**Current focus** (one sentence): EDA done + ImageNet-1K download started in
background; `data_v2.py` next tick.
**Last action**: Tick 75 — fixed disk issue (HF cache default = root partition,
30 G, filled at first download attempt) by clearing `/root/.cache/huggingface`
(freed 28 G) and relaunching the download with `HF_HOME=/workspace/.hf_cache`
(1.7 TB free). Wrote `holo_net/data_v2.py` (DINOv2 multi-crop: 2 global ×
224² + 8 local × 96² + ImageNet-norm + RandomHFlip + ColorJitter +
RandomGrayscale; per-worker MediaPipe lazy-init; collate produces per-view
batched tensors + face_mask + labels). Deployed to server.
**Last action outcome**: data_v2.py ready; download at 73 % (38/52 files,
~3 min ETA for train + a moment for val). Sanity check pending download completion.
**Running tasks** (on server, H100 80GB):
  - ImageNet download (PID re-launched). ~3-5 min to complete.
**Stuck streak**: 0
**Planned next action** (tick 76): verify download done, run `data_v2.py`
sanity check; if OK → start writing `train_v2.py` (DINO teacher-student EMA +
DINO KL loss + PC loss + template EMA update + cosine LR + warmup).
Tick 77 = launch SSL training (invoke **`experiment-run` SKILL**, ~24-30 h
H100 for 100 epochs on ImageNet-1K).
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
