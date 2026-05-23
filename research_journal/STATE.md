# State

**Tick #**: 83 (completed)
**Last updated**: 2026-05-23 ~20:55 (Asia/Hong_Kong)
**Current focus** (one sentence): **Stage 3 EEG decoder built and validated**
— per-subject ridge ATM-EEG → DINOv2, top-1 retrieval **18.9% ± 3.8% (10
subjects, 200-way, chance 0.5%; 38× chance)**, top-5 43.6%, top-10 56.5%,
per-dim r_d mean 0.315 with 98.5% dims > 0.1; **DINOv2 target outperforms
CLIP-H/14 by ~63% relative under same ridge** (CLIP top-1 11.6% via same
pipeline) — new finding: target-space matters, refining E020's "uniform
low-pass" thesis. Tick 84 next = IllusionBench transfer (the headline).
**Last action**: Tick 83 — wrote `eeg_decoder/stage3_ridge.py` (per-subject
closed-form ridge + 5-fold λ-CV + top-K retrieval + per-dim r_d for tick-84).
Ran on server: 10 subjects × λ-CV + final fit + eval in <60s total. λ_best=100
for all subjects (CV consistent). **Top-1 = 0.189 ± 0.038** (200-way; chance
0.5%; 38× chance), **top-5 = 0.436 ± 0.070**, **top-10 = 0.565 ± 0.075**.
Per-dim r_d on DINOv2 test predictions: mean 0.315, std 0.108, 98.5% dims
>0.1. Pre-registered prediction (18-32% top-1) CONFIRMED. Ran sanity
comparator (same ridge, CLIP-H/14 target): top-1 = 0.116 ± 0.037 → **DINOv2
target outperforms CLIP-H/14 by ~63% relative under linear decoding** — a
new finding refining E020's "uniform low-pass" thesis (the low-pass was
target-specific). Wrote E046 with full results + sanity comparator + linked
ideas. NPZs at `/workspace/runs/2026-05-23_stage3-ridge_seed20260521/`.
**Last action outcome**: Stage-3 EEG decoder validated, ridge weights +
per-dim r_d saved → ready for tick-84 transfer.
**Running tasks** (on server, H100 80GB): none — server idle.
**Stuck streak**: 0
**Planned next action** (tick 84, ~30 min — the headline): write
`eeg_decoder/illusionbench_transfer.py` — for each (subject, layer, paradigm):
  apply per-dim r_d filter to IllusionBench DINOv2 features at FFA layer →
  feed through repo `analysis.compute_metrics` → record ISI/CSI/PWI/random-bbox.
Pre-registered prediction: PWI/random-bbox PASS likely preserved (raw 0.446,
0.654; per-dim attenuation should not flip sign); CSI 1.30 → unclear (close
to threshold); ISI 0.901 → likely stays anti-direction. Report per-subject
distribution + aggregate. Update IDEA_PIPELINE scores based on outcome.

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
