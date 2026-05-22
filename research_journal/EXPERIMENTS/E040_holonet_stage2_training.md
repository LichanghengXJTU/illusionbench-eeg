# E040 — HOLO-Net Stage-2 face fine-tune: full-model training failure and minimal-mode fallback

**Status**: IN PROGRESS (training ~53% at tick 50) — no IllusionBench evaluation yet.
**Linked**: Idea-003 (HOLO-Net), Q014 (new), `IDEA_003_HOLO_NET_DESIGN.md`, ticks 41-50.

**Hypothesis** (Idea-003 pre-registered falsification): A HOLO-Net trained only on
Glint360K face identities — with NO illusion stimuli in training — will, at the FFA
layer, simultaneously satisfy Thatcher ISI ≥ 3.0, Composite CSI ≥ 1.5, Part-Whole
PWI ≤ 0.5, Random-bbox ISI ≤ 1.5. No prior in the 25-prior battery achieves this
combination.

**Method**: Stage-2 supervised face fine-tune. `holo_net/{model,data,losses,train}.py`.
Glint360K WebDataset (`gaunernst/glint360k-wds-gz`, 1385 shards, 130 GB, 112² aligned).
AdaFace quality-adaptive angular-margin identity head. Multi-task loss (identity +
predcode + orientation + face-detect + view-invariance + gist). H100 80GB.

**Stimuli (training)**: Glint360K faces only. NO Thatcher / composite / part-whole
perturbations (design constraint §0.4 of the design doc). IllusionBench is strictly
test-only.

**Models**: HOLO-Net v1.0 — 56.26M params (full); 53.63M params (minimal).

**Metric**: Stage-2 here = training loss curves only. Stage-2 *evaluation*
(ISI/CSI/PWI/random-bbox per layer) is PENDING — see Next steps.

**Pre-registered prediction**: see Hypothesis. Evaluation not yet run; no illusion
numbers exist for HOLO-Net.

## Result (numbers only, no interpretation) — from server logs

| Run | Config | Identity loss | Outcome |
|---|---|---|---|
| v1 (`holo_net_train_v1_failed.log`) | FULL model (all bio components), AdamW lr≈1e-4, 360K classes | flat **63-69** over steps 0-8750, no downward trend; aux pc≈3.22, ori≈2.7-3.5 | killed |
| v5 (`holo_net_train_v5_cpu.log`) | minimal mode, launched on CPU (misconfig) | — | killed (too slow) |
| v6 (`holo_net_stage2_v6_1epoch/`) | minimal mode, SGD lr 0.1, 360K classes | ≈24.7 at step 1250 | aborted ~1250 steps |
| current (`holo_net_stage2/`) | minimal mode, SGD lr 0.1 mom 0.9 wd 5e-4, num_classes=10000, batch 512, 30000 steps, warmup 1000 + cosine, AMP fp16 | **13.13 (step 0) → 5.78 (step 15850, elapsed 95.8 min)** | running, ~53% |

current-run aux losses: view-invariance 0.70 (step 0) → 0.088 (step 15850);
predcode / orientation / face-detect / gist = 0.000 throughout (components disabled).
step_t ≈ 0.35-0.39 s.

"minimal mode" disables, per `train.py` `setup_model_and_loss(minimal=True)`:
LGN-Magno, OFA, FFA, Orientation Gate, Predictive-Coding feedback, PFC-Gist.

## Interpretation

- **[CONFIRMED by run v1]** The full HOLO-Net architecture (all bio-fidelity
  components) under AdamW lr≈1e-4 does not optimize the identity objective:
  identity loss held 63-69 — roughly 5× the ≈13 starting loss of the minimal
  config — for 8750 steps with no downward trend. Tick 43's journal entry
  ("identity loss 65 ... expected for cold start") misread this as normal; it was not.
- **[CONFIRMED by current run]** The minimal configuration (CORnet-style
  LGN-Parvo→V1→V2→V4→IT(MFP)→AFP→ATL backbone + AdaFace head, all bio additions
  OFF) does optimize: identity loss 13→5.8 over ~16K steps.
- **[CONJECTURE]** The v1 failure is **not yet isolated**: v1→current differs on
  THREE axes at once (full→minimal components, AdamW→SGD, 360K→10K classes).
  Candidate causes: (a) predictive-coding reconstruction + 3-step recurrent FFA
  self-attention create a conflicting gradient landscape for the shared backbone;
  (b) 6-term multi-task loss with hand-set weights is imbalanced; (c) AdamW
  lr 1e-4 too low to escape init; (d) view-invariance aux without contrastive
  negatives → embedding collapse (a collapse at aux-weight 0.05 is flagged in a
  `train.py` comment). → Q014.
- **[CONJECTURE — IMPORTANT]** The current run, even when complete, does **NOT
  test the HOLO-Net thesis**. With FFA / OrientGate / PC / LGN-Magno / PFC
  disabled, the architecture reduces to ≈ Idea-003 **sub-path (a)** (CORnet-S
  anatomy + AdaFace + Glint360K face data), NOT sub-path (e) ("design our own").
  The pre-registered falsification criteria are defined *at the FFA layer*
  (design doc §6) — a layer that does not exist in the minimal model. The current
  run is therefore a sub-path-(a) datapoint, not a HOLO-Net evaluation.
- **[CONJECTURE]** Identity loss 5.8 after ~1 epoch on a 10K-identity subset is
  "learning but weak"; a converged AdaFace reaches far lower. Illusion-index
  estimates from this checkpoint will carry wide error bars.

## Replicability check

- Code: `holo_net/{model,data,losses,train}.py` at the tick-50 commit; local
  working tree verified byte-identical to server `/workspace/illusionbench-eeg/holo_net/`.
- ⚠ **NO explicit random seed**: `train.py` sets no `manual_seed` and the launch
  command passed no `--seed`. Re-runs are not bit-reproducible. Any number from
  this checkpoint that enters the paper must be re-run with a fixed seed
  (project convention: 20260521).
- Checkpoint: server `/workspace/holo_net_stage2/checkpoint.pt`;
  log `/workspace/holo_net_stage2/train_log.jsonl`.
- Launch: `python -m holo_net.train --mode train --minimal --num_classes 10000
  --optimizer sgd --lr 0.1 --warmup_steps 1000 --batch_size 512 --max_steps 30000
  --num_workers 32`

## Next steps

1. Let the current run finish (~85-90 min from tick 50).
2. **E041** — evaluate the minimal checkpoint on all 3 IllusionBench paradigms +
   random-bbox control. This is a clean **sub-path (a)** result (CORnet anatomy +
   AdaFace + Glint360K), directly comparable to E028 (CORnet-S ImageNet) and
   E032 (AdaFace-IR50-CASIA).
3. **Q014** — component-ablation ladder: starting from the known-trainable config
   (SGD 0.1, 10K classes), re-enable bio components one at a time (+PC, +FFA,
   +OrientGate, +OFA, +LGN-Magno, +PFC) to isolate which one(s) break training.
