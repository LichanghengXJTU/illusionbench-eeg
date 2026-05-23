# State

**Tick #**: 76
**Last updated**: 2026-05-23 ~13:00 (Asia/Hong_Kong)
**Current focus** (one sentence): EDA done + ImageNet-1K download started in
background; `data_v2.py` next tick.
**Last action**: Tick 76 — ImageNet-1K download DONE (1,281,167 train + 50,000
val, 24 G in /workspace/.hf_cache). `data_v2.py` sanity on real data passed
(face_mask hit-rate ~11% averaged over 2 batches of B=64 — matches Yang 2022
~17% within sample variance; labels diverse → shuffle working). Wrote
`holo_net/train_v2.py` (DINO teacher-student, DINOLoss with centering +
sharpening, AdamW + cosine warmup, AMP fp16, teacher EMA momentum 0.996,
PC-loss-on-face-mask + template EMA, checkpoint every 5 ep). Deployed +
import-tested on server.
**Last action outcome**: all three v2 components in place (model_v2 + data_v2
+ train_v2). Ready to launch.
**Running tasks** (on server, H100 80GB): none.
**Stuck streak**: 0
**Planned next action** (tick 77): invoke **`experiment-run` SKILL** to
launch the SSL training: 1-step sanity (verify forward+backward+EMA flow)
then nohup the full 100-epoch run. Estimate ~24-40 h H100 single-GPU
(batch 256, AMP fp16, CORnet-S backbone). Then daily-cadence monitoring
ticks until training completes; intermediate checkpoint eval at epoch ~25.
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
