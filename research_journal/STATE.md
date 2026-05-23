# State

**Tick #**: 81 (completed)
**Last updated**: 2026-05-23 ~19:55 (Asia/Hong_Kong)
**Current focus** (one sentence): EEG-decoder Stage 3 scaffold landed —
design doc locked, ATM features (train+test, 10 subjects) + ViT-H-14 image
features downloaded (2.8 GB), `eeg_decoder/{__init__,data}.py` module
created, sanity ✓ (22 expected files present, 2 expected-missing for tick 82).
**Last action**: Tick 81 — server reachable / GPU idle / disk fine. Probed
THINGS-EEG2 data on server: ATM EEG _test_ features present from earlier
ticks, but no train EEG features, no raw EEG, no THINGS images, no DINOv2
features. Decided pragmatic Stage-3 path: skip raw EEG (100 GB, multi-day
download + ATM-style training); reuse ATM's pre-computed EEG features as
the EEG-side source (per-subject 1024-d, CLIP-H/14-aligned), train per-
subject ridge mapping ATM-EEG → DINOv2; this isolates the "EEG bottleneck
through DINOv2 substrate" question without an ATM-retrain confound.
Downloaded full ATM EEG features for all 10 subjects (train (66160, 1024)
+ test (200, 1024)) + ViT-H-14 train+test image features (74 MB + 1.6 MB)
from HF. Wrote `EEG_DECODER_STAGE3_DESIGN.md` (full pre-registration with
predictions). Wrote `eeg_decoder/__init__.py` + `data.py` (loaders +
sanity_check). SCP'd, ran sanity on server: 22 files present, 2 expected-
missing for DINOv2 features tick 82 will produce.
**Last action outcome**: scaffold complete; data inventory passes sanity;
design pre-registered with retrieval + IllusionBench-transfer predictions.
**Running tasks** (on server, H100 80GB): none — server idle.
**Stuck streak**: 0
**Planned next action** (tick 82, ~30 min): download THINGS images from
official OSF (~2 GB for 1854 concepts) + write `extract_dinov2_things.py`
(frozen DINOv2 ViT-S/14 forward, save dinov2_vits14_{train,test}.pt to
`data/things_features/`). Should complete in one tick.
Then tick 83: train ridge + evaluate retrieval. Then tick 84: IllusionBench
transfer (the headline).

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
