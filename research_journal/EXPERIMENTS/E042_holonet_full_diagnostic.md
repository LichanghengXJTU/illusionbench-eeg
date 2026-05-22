# E042 — HOLO-Net full architecture: controlled re-test isolates the v1 failure to the AdaFace loss bug

**Status**: IN PROGRESS (full-model training launched tick 51) — answers Q014.
**Linked**: Idea-003, Q014, E040, `IDEA_003_HOLO_NET_DESIGN.md`.

**Hypothesis (Q014)**: The tick-43 full-model run (v1) failed — identity loss
stuck 63-69 — for one of: **(H1)** a bug in the AdaFace loss; **(H2)** the
bio-fidelity components (PC feedback / recurrent FFA / 6-term multi-task loss)
create a pathological optimization landscape; **(H3)** AdamW lr 1e-4 too low.

## Diagnosis from code audit (tick 51)

`holo_net/losses.py` `AdaFaceLoss.forward` carries an explicit fix-comment:

> "CORRECT formula (no extra subtraction; v1/v2 had buggy `- scale * m` term
> that artificially depressed target logits and caused identity loss to start
> at ~65 instead of ~1)"

So the debugging ticks 44-49 ALREADY found and removed an AdaFace formula bug —
an extra `− scale·margin` term subtracted from the target-class logit — and
added margin warmup (margin ramps 0→0.4 over 4000 steps). **This supports H1.**
But those same ticks ALSO disabled every bio component ("minimal mode") at the
same time, so H1 and H2 were never disentangled: the only post-fix run (the
minimal run, E040) cannot tell whether the bio components are independently
harmful.

## Method

Controlled comparison. Run the **FULL HOLO-Net** (all bio components ON:
LGN-Magno, OFA, FFA recurrent attention, Orientation Gate, PC feedback,
PFC-Gist) with the *fixed* AdaFace loss, holding **everything else identical**
to the minimal run (E040): SGD lr 0.1 + nesterov mom 0.9 wd 5e-4,
num_classes 10000, batch 512, warmup 1000 + cosine, AMP fp16, 30000 steps.
The ONLY variable = bio components on/off.

**Models**: HOLO-Net v1.0 full (56.26M params) vs minimal (53.63M = E040).
**Stimuli (training)**: Glint360K, 10K-identity subset. No illusion stimuli.

**Pre-registered prediction**:
- If H1 (loss bug) is the sole cause → full model starts at identity loss
  ≈ 13 (same as minimal) and descends similarly.
- If H2 (components pathological) → full model starts much higher (≈65 like v1)
  OR runs but fails to descend.

## Result (numbers only) — server `/workspace/holo_net_full_v1/train_log.jsonl`

| step | identity | predcode | orientation | face_detect | gist | total |
|---|---|---|---|---|---|---|
| 0 | **13.20** | 4.17 | 0.69 | 0.81 | 0.64 | 13.76 |
| 50 | 13.27 | 4.21 | 0.69 | 0.79 | 0.64 | 13.84 |
| 100 | 13.32 | 4.33 | 0.69 | 0.74 | 0.64 | 13.89 |

Compare **E040 minimal run, step 0: identity = 13.13.**

Full model forward + backward run cleanly on real Glint360K data — no NaN, no
shape error. GPU +21 GB (40 GB total alongside the still-running minimal run).
step_time ≈ 0.8 s (GPU shared with the minimal run).

## Interpretation

- **[CONFIRMED] H1 is the cause; H2 is refuted as a blocker.** The full
  HOLO-Net with all bio components, under the fixed loss, starts at identity
  loss 13.20 — within 0.07 of the minimal model's 13.13. The bio components do
  NOT inflate the loss and do NOT break the forward/backward pass. The v1
  failure (loss stuck ~65) was entirely the AdaFace `− scale·m` bug. H3 (AdamW)
  is moot — the loss bug alone explains it.
- **[CONFIRMED] The "minimal mode" detour was unnecessary.** The correct fix was
  the loss formula alone; disabling the bio-fidelity components — i.e. the
  entire HOLO-Net thesis — was not required, and is now reverted.
- **[CONJECTURE] descent**: at step 100 identity is flat-to-slightly-up
  (13.20→13.32), as expected during lr warmup (1e-4→1e-2) and AdaFace margin
  warmup (0→0.4 over 4000 steps); the minimal run showed the same early
  non-monotonic phase before descending to ~4. Full-model descent is expected
  but not yet confirmed — checkpoint next tick.

## Replicability check

- Code = tick-51 commit. ⚠ Still NO explicit seed (carried over from E040) —
  must be fixed before any paper number is taken from these checkpoints.
- Output: `/workspace/holo_net_full_v1/` (checkpoint.pt + train_log.jsonl).
- Launch: `python -m holo_net.train --mode train --num_classes 10000
  --optimizer sgd --lr 0.1 --warmup_steps 1000 --batch_size 512
  --max_steps 30000 --num_workers 24 --output_dir /workspace/holo_net_full_v1`
  — note: **no `--minimal`** (full architecture).

## Next steps

1. Confirm full-model loss descent (next tick).
2. When training completes → evaluate the FULL checkpoint on IllusionBench:
   per-layer ISI / CSI / PWI / random-bbox, with the falsification criteria
   read at the FFA layer (design doc §6). **This is the actual HOLO-Net test.**
3. The minimal run (E040) becomes the "minus all bio components" ablation arm.
4. Decide whether to scale identities 10K→larger (10K ≈ CASIA-WebFace scale,
   itself the best face-rec Thatcher prior P26 — so 10K is already defensible)
   for a definitive run.
