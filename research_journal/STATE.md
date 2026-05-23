# State

**Tick #**: 74
**Last updated**: 2026-05-23 ~12:00 (Asia/Hong_Kong)
**Current focus** (one sentence): EDA done + ImageNet-1K download started in
background; `data_v2.py` next tick.
**Last action**: Tick 74 — invoked **`dataset-inspect` SKILL**: ran MediaPipe
face detection on a streamed ImageNet-1K sample (twice — first attempt was
biased by HF sequential streaming hitting only class-0 "tench" / first-20
bird+fish classes, giving misleading 62%/1.6%). Cross-referenced literature:
[Yang et al. 2022 FAccT](https://dl.acm.org/doi/10.1145/3531146.3534615)
reports **~17% ImageNet face-containment rate** (MTCNN) — my biased samples
bracket this (62% high on people-rich classes, 1.6% on bird/fish). MediaPipe
pipeline verified working (confident detections, mean score 0.85 on faces).
Committed to **`evanarlian/imagenet_1k_resized_256`** (non-gated, 256-resized,
~70 GB) and started the HF download in background.
**Last action outcome**: face_mask design defensible (at batch 1024, expect
~170 face-samples/batch → template EMA will converge in <1 epoch). Dataset
acquisition path clear, download progressing (7/52 files in 2 min, ETA ~20 min).
**Running tasks** (on server, H100 80GB):
  - ImageNet download (background, PID 66574). ETA ~20 min.
**Stuck streak**: 0
**Planned next action** (tick 75): write `holo_net/data_v2.py` —
DINOv2-style multi-crop augmentation (2 global + 6 local crops) + MediaPipe
face-mask in DataLoader workers; verify against the downloaded dataset.
Then tick 76 = `train_v2.py` (DINO teacher-student + PC loss + EMA template
update); tick 77 = launch SSL training (invoke **`experiment-run` SKILL**).
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
