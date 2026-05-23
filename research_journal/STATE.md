# State

**Tick #**: 86 (completed) — paper outline v2
**Last updated**: 2026-05-23 ~22:15 (Asia/Hong_Kong)
**Current focus** (one sentence): Wrote `PAPER_OUTLINE_v2.md` integrating all
post-tick-8 evidence (HOLO-Net 3-variant negatives + Stage 3 decoder + transfer
vindication + MLP sub-finding). v1 outline (tick 8) preserved as
`PAPER_OUTLINE.md`. v2 has updated abstract (270 words), 5 contributions
re-formulated, 10-section structure (intro / related / benchmark / image-side /
HOLO-Net / EEG / sub-finding / discussion / future / conclusion), figures and
tables inventories updated, 4 user-decision points flagged (single vs 2 papers,
companion PWI paper, run v2.2 first?, run Stage 4 first?).
**Last action**: Tick 85 — wrote `eeg_decoder/stage3_mlp.py` (2-layer MLP
1024→512→384 with GELU+dropout, cosine loss, AdamW, early-stop). Ran on H100:
10 subjects × ~7s each = 78s total. Retrieval: top-1 0.181 ± 0.039 (vs ridge
0.189 ± 0.038), top-5 0.430 (vs 0.436), top-10 0.566 (vs 0.565) — ESSENTIALLY
IDENTICAL to ridge on top-K. Per-dim r_d mean 0.324 (vs ridge 0.315) — also
≈ same. Re-ran `illusionbench_transfer.py` with MLP per-dim r_d.
**MLP IllusionBench §6 verdict (aggregate)**:
- ISI = 0.814 (vs ridge 0.811; raw 0.908) FAIL — unchanged
- CSI = 1.273 (vs ridge 1.275; raw 1.282) FAIL — unchanged
- **PWI = 0.479 PASS (vs ridge 0.537 FAIL; raw 0.397)** — MLP recovers PWI
- ISIrbox = 0.603 (vs ridge 0.603; raw 0.662) PASS — unchanged
- ALL FOUR: FAIL (2/4 pass — better than ridge's 1/4)
**Per-subject**: 7/10 pass PWI (vs ridge's 3/10); 0/10 achieve 4/4 (same).
Wrote E048 capturing: ISI/CSI bottlenecked decoder-family-invariantly (strong
result); PWI has non-linear EEG structure (sub-finding); 0/10 4/4 robust.
**Last action outcome**: headline strengthened and refined. Story:
"ISI/CSI fundamentally bottlenecked by EEG (decoder-family-invariant);
PWI partially recoverable with non-linear decoder (cognitive sub-finding);
4/4 PASS unreachable for any decoder we've tried (benchmark sharpness holds)."
**Running tasks** (on server, H100 80GB): none — server idle.
**Stuck streak**: 0
**Planned next action** (tick 87, ~20 min): per outline §D3 default
recommendation, **implement HOLO-Net v2.2 part-aware FTPC** (image-side
complement to the EEG-side headline). K=4 sub-templates {T_eyes, T_nose,
T_mouth, T_chin} at fixed sub-regions of the 16×16 patch grid; per-region
δ aggregation; quick template-only training on ImageNet face samples; eval
on IllusionBench. If v2.2 lifts ISI from 0.901 (anti) to ≥ 1.0 (no anti)
that's a meaningful image-side gain. ~1 tick to scaffold + run.

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
