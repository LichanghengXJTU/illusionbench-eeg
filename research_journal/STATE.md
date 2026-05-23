# State

**Tick #**: 91 (completed) — Gen 3 v4.0 LAUNCHED
**Last updated**: 2026-05-24 ~00:25 (Asia/Hong_Kong)
**Current focus** (one sentence): **Gen 3 v4.0 training STARTED on 1× H100**
(user approved single-card path); ViT-S/14 from scratch + DINOv2-style SSL
+ NO vflip aug + orient classification aux head. Wrote 3 new files
(gen3_data.py, gen3_model.py, gen3_train.py); fixed torch.amp→torch.cuda.amp
API; sanity bs=64 passed, bs=256 timing 1.2 s/step; launched 30 epochs in
background (ETA ~2.1 days). E053 written with pre-registered checkpoints
at steps 5000/25000/100000. Monitoring committed every 30 min.
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
**Planned next action** (tick 92, ~30 min from launch): **first monitor
checkpoint** of Gen 3 v4.0 training.
- SSH to server, check process alive
- Read last ~10 lines of train_log.jsonl
- Pre-registered step-5000 checkpoint: L_dino < 9.0, L_orient < 0.4,
  orient_acc > 0.8. If criteria fail → COLLAPSE detected, kill + diagnose.
  If healthy → continue.
- Report to user.

Subsequent ticks: monitor every 30-60 min until either (a) collapse +
restart with revised recipe, (b) step 100K healthy → schedule longer
intervals while training completes, or (c) training done → run §6 eval
on the new backbone via FTPC pipeline.

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
