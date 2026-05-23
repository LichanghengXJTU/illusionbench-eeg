# State

**Tick #**: 84 (completed) — **THE HEADLINE TICK**
**Last updated**: 2026-05-23 ~21:20 (Asia/Hong_Kong)
**Current focus** (one sentence): **IllusionBench-EEG transfer DONE**. 0/10
subjects pass 4/4 §6 criteria; aggregate gives **1/4 PASS (only ISIrbox)**;
3/10 subjects scrape 2/4 (PWI + ISIrbox). **PWI flipped from raw DINOv2
PASS (0.397) to EEG-decoded FAIL (0.537 aggregate; 3 subjects with best
r_d preserve PWI).** ISI worsens slightly (0.811 vs 0.908 raw, more
anti-Thatcher). CSI ≈ unchanged (1.275 vs 1.282). ISIrbox preserved for all
10 subjects — control criterion robust. **THE BENCHMARK'S PRE-REGISTERED
SHARP PREDICTION IS VINDICATED**: even the best EEG decoder + best target
(DINOv2 + ridge) cannot deliver the human-aligned §6 PASS profile.
**Last action**: Tick 84 — wrote `eeg_decoder/illusionbench_transfer.py` —
per-dim r_d filter applied to IllusionBench DINOv2 features (raw mean-pooled
features at FFA layer = E045 baseline), per subject + aggregate + raw control;
then re-ran `analysis.compute_metrics` on each. Pixel-corrected.
**Result table (aggregate r_d; pixel-corrected at FFA = EEG-decoded DINOv2)**:
- ISI = 0.811 (raw 0.908; **anti-direction worse**; need ≥3.0 → FAIL)
- CSI = 1.275 (raw 1.282; essentially preserved; need ≥1.5 → FAIL)
- PWI = 0.537 (raw 0.397; **flipped PASS→FAIL**; need ≤0.5 → FAIL)
- ISIrbox = 0.603 (raw 0.662; control passes; need ≤1.5 → PASS)
- **ALL FOUR: FAIL (1/4 pass; only ISIrbox)**

Per-subject: 7/10 give 1/4 pass (ISIrbox only); 3/10 give 2/4 (PWI + ISIrbox);
**0/10 give 3+/4**. The 3 PWI-pass subjects (04, 08, 10) have the highest
per-dim r_d means (0.342-0.354), confirming a clean dose-response.
E047 written with full results + interpretation + idea-pipeline impact.
**Last action outcome**: the IllusionBench-EEG benchmark's pre-registered
sharp prediction is **VINDICATED**: even the best EEG decoder + best target
combination we found cannot deliver the §6 human-aligned PASS profile.
**Running tasks** (on server, H100 80GB): none — server idle.
**Stuck streak**: 0
**Planned next action** (tick 85, ~25 min): two-track —
  - **Track 1 (Paper consolidation)**: outline the IllusionBench-EEG paper
    with the 3 lines of evidence now in hand (image-side priors, HOLO-Net
    3× negatives, EEG-side transfer vindication). The story is complete
    enough to start drafting.
  - **Track 2 (Stage-3b sensitivity)**: try MLP mapping instead of ridge to
    test the [CONJECTURE] in E047 that non-linear decoders could recover
    PWI structure that linear-attenuation destroys. If MLP also fails, the
    "EEG bottleneck is fundamental" conclusion is much stronger.
  - **Track 3 (deferred)**: HOLO-Net v2.2 part-aware FTPC (image-side
    complement; lower priority now that the EEG-side headline is complete).

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
