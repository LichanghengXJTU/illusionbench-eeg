# State

**Tick #**: 82 (completed)
**Last updated**: 2026-05-23 ~20:25 (Asia/Hong_Kong)
**Current focus** (one sentence): THINGS images downloaded + DINOv2 features
extracted (train 12.2s, test 1.4s, all on H100); EEG-decoder Stage 3 data
inventory fully populated (24/24 expected files present); ready for tick-83
ridge training + retrieval eval.
**Last action**: Tick 82 — discovered THINGS-EEG2 stimulus images are bundled
with the HF dataset (`images_set.tar.gz`, 0.78 GB) — saved a separate OSF
download trip. Downloaded (29s) + extracted (18,604 entries: 1654 train
concept dirs × 10 images + 200 test concept dirs × 1 image + metadata +
test_images_tensor pre-cooked). Verified directory schema matches the
ATM index convention (alphabetical 1..1654 outer, alphabetical 1..10 inner;
metadata.npy provides explicit `{train,test}_img_{concepts,files}` lists).
Wrote `eeg_decoder/extract_dinov2_things.py` (frozen DINOv2 ViT-S/14 +
`mean`-pooled patch tokens matching the FFA-layer `afp_pooled` convention
used in tick-80 IllusionBench eval, so the same feature space is used
across the EEG and IllusionBench measurements). SCP'd, ran on server:
test 200 imgs in 1.4 s (6.89 ms/img), train 16,540 imgs in 12.2 s (0.74
ms/img). Outputs saved to `data/things_features/dinov2_vits14_{test,train}.pt`
shape (200, 384) and (16540, 384). Sanity check via `data.py`:
**24/24 expected files present**; ATM↔DINOv2 alignment verified by index;
DINOv2 train feature norms 20.4–45.0 (mean 29.7) = healthy unnormalized
ViT-S features.
**Last action outcome**: all Stage-3 inputs locked in.
**Running tasks** (on server, H100 80GB): none — server idle.
**Stuck streak**: 0
**Planned next action** (tick 83, ~15 min — work is small, can pace fast):
write `eeg_decoder/stage3_ridge.py` — per-subject ridge ATM-EEG (16540×1024)
→ DINOv2 (16540×384), closed-form solution + λ sweep on a held-out fold;
evaluate top-K retrieval (K=1, 5, 10) on the 200 test stimuli using cosine
similarity in DINOv2 space; record per-subject distribution + summarize.
Compare to ATM's published CLIP retrieval as sanity. Then tick 84:
`illusionbench_transfer.py` — apply trained ridge to project Thatcher /
Composite / Part-Whole / Random-bbox DINOv2 features to EEG-decoded space,
measure ISI/CSI/PWI/random-bbox (the headline).

## Confidence in current best ideas
- **Idea-001** (IllusionBench-EEG): **9.5/10** — strongly reinforced. 3 distinct
  HOLO-Net instantiations now all fail §6 simultaneously, adding architecture-
  side corroboration to the training-objective thesis from 3 angles (identity
  loss, SSL-from-scratch-collapse, frozen-SSL+global-template). The benchmark
  itself is paper-ready; the v1+v2-collapse+v3 negatives are clean mechanistic
  evidence of why current vision SOTA can't satisfy it.
- **Idea-003** (HOLO-Net positive): **5.5/10** — slight up (was 5.0). v3
  (DINOv2-FTPC) gives **2/4 pass** vs v1's 1/4 (PWI flipped from 2.20 FAIL to
  0.446 PASS — large reversal in the right direction). Composite CSI 1.300 is
  within striking distance of the 1.5 threshold. The headline ambition is still
  alive if v2.2 (part-aware FTPC) lifts the Thatcher signal.

## The HOLO-Net result, two lines
v1 (CORnet + AdaFace identity): NO illusions (ISI/CSI/PWI all wrong direction).
v3 (frozen DINOv2 + global FTPC): PWI + random-bbox PASS; Thatcher ANTI-direction,
Composite close but FAIL. Diagnosis: global template can't isolate the
local-feature-orientation signal Psalta 2014 identifies as the Thatcher locus.

## Strategic note
The user's explicit #1 priority is the EEG decoder (主线 = EEG 关联). HOLO-Net
v3 makes the DINOv2 backbone production-ready for the EEG side; even if v2.2
fails, we now have a defensible Stage 3 EEG decoder built on a SOTA SSL prior
that already moves PWI from FAIL to PASS — that's a publishable IEEE-style
EEG-AI paper independent of the HOLO-Net headline.
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
