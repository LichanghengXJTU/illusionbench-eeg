# State

**Tick #**: 73
**Last updated**: 2026-05-23 ~11:10 (Asia/Hong_Kong)
**Current focus** (one sentence): v2.1 implementation started — `model_v2.py`
written + sanity-tested on server (forward / backward / EMA update / per-layer
readout all OK).
**Last action**: Tick 73 — wrote `holo_net/model_v2.py` (HOLO-Net v2.1 / FTPC):
ventral backbone (V1-V2-V4-IT-AFP, reusing v1 bricks) + learnable face
template T + δ = AFP_spatial − T pure read-out (no anomaly conv, no α
back-modulation) + DINOv2-style projection head for SSL. Per-layer aliases
match `eval_extract.py` (ffa → δ-pooled = fSTS-analogue; atl → afp_pooled).
Sanity check ran on server: all shapes correct, backward flows, EMA template
update verified.
**Last action outcome**: 76.05M params (53.44M backbone + 22.6M SSL head);
model is ready for data-pipeline + SSL training next tick.
**Running tasks** (on server, H100 80GB): none.
**Stuck streak**: 0
**Planned next action** (tick 74): write `holo_net/data_v2.py` —
ImageNet-1K dataloader with multi-crop DINO augmentation + MediaPipe face
detector providing the per-sample face_mask. Will invoke the
**`dataset-inspect` SKILL** to do an EDA pass on ImageNet face-detection
hit-rate first (sanity that we'll get enough face-containing samples for
template EMA to learn). Tick 75 = write `train_v2.py` (DINO teacher-student
+ PC loss + EMA template update); tick 76 = launch training (invoke
**`experiment-run` SKILL**).
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
