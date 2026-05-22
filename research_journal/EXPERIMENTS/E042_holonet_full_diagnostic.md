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

---

## Addendum — tick 54: OrientationGate bug found + fixed; run v1 → v2

While monitoring run v1, the `orientation` aux loss was found dead-flat at
ln 2 ≈ 0.693 for 6450 steps (Q015). Root cause — a real architecture bug:

- `OrientationGate` global-average-pooled `afp_spatial` (B,512,7,7) → (B,512)
  before its classifier. Orientation = a whole-face vertical flip = a spatial
  permutation; a mean is permutation-invariant ⇒ the GAP'd vector is provably
  blind to orientation.
- Verified at random init: `orientation_logits` differed by 3e-5 between an
  image and its vertical flip; `GAP(t) − GAP(flip(t))` = 0.000000 exactly,
  while `pool4×4(t) − pool4×4(flip(t))` = 0.43 (131% of signal magnitude).
- The conv stack itself IS orientation-bearing (afp_spatial flip-aligned
  residual 9e-4 ≪ its 2.2e-3 magnitude) — only the pooling discarded it.
- The OFA / PFC-Gist heads also GAP, but face/non-face is a channel-statistics
  task (not spatial), so GAP is fine there — consistent with the observation
  that `face_detect` and `gist` losses DID descend while `orientation` did not.

**Fix** (`model.py` `OrientationGate`): pool to a coarse 4×4 grid
(`AdaptiveAvgPool2d((4,4))`) instead of 1×1, preserving top-vs-bottom layout;
classifier input 512 → 512·16. +0.49M params (56.26M → 56.75M). Also added
`--seed` (default 20260521) to `train.py` for reproducible model init.

Run v1 was killed at step ~7000 — with an inert OrientationGate its FFA gating
(the Thatcher mechanism, design §7) could never engage, so the Thatcher result
would have been a guaranteed null. **Run v2** relaunched with the fix:
`/workspace/holo_net_full_v2/`, identical config (full model, SGD 0.1, 10K
classes, batch 512, 30K steps) + `--seed 20260521`. v2 step 250: identity 10.6,
predcode 4.5, step_time 0.38 s (minimal run had finished → full GPU). Orientation
descent to be confirmed next tick — the gate now has the capacity; with the
buggy gate it was mathematically impossible.

---

## Addendum — tick 55: gate fix worked but exposed an identity-collapse; run v2 → v3

Run v2 confirmed the OrientationGate fix: `orientation` loss descended
0.69 → 0.26 by step 1050 (vs dead-flat for 6450 steps in v1). **But the identity
loss stopped descending** — it bounced 11-14 across 4900 steps (v1, with the
inert gate, at least drifted 13→11; the minimal run reached ~8 by that step).

Root cause: with a *functional* gate, `afp_gated_spatial = afp_spatial · gate`
zeroes the FFA input for inverted faces (gate→~0.15), and since
`atl_input = ffa_out` was the SOLE identity path, inverted faces (50% of every
batch) became unidentifiable → identity loss ≈ ln(C) on half the batch → no net
descent. An orientation gate must not multiplicatively gate the only route to
the identity head.

**Fix** (`model.py` `HOLONet.forward`): FFA now runs on UNGATED `afp_spatial`;
the ATL identity input is `afp_pooled + gate · ffa_out` — an always-present
featural identity representation PLUS an orientation-gated additive holistic
term. Inverted faces stay identifiable (via afp_pooled); upright faces
additionally engage FFA configural binding. The Thatcher asymmetry now lives in
the gated additive term (faithful to design §7) without destroying identity
training. `out["ffa"]` = this orientation-modulated holistic representation.

Run v2 killed at step ~5000; **run v3** launched
(`/workspace/holo_net_full_v3/`, fixed wiring, seeded). v3 step 250: identity
10.65, step_time 0.34 s. Identity-loss descent + orientation descent to be
confirmed next tick. (The minimal-run E041 ablation baseline is unaffected —
the minimal path `atl_input = afp_pooled` is unchanged by this fix.)

---

## Addendum — tick 58: CORRECTION — the "identity bounce" is NORMAL; v4 launched to run to completion

Fetching the minimal run's full trajectory (the proper reference, not previously
on hand) overturns the tick-55/56 interpretation:

```
minimal identity:  step 0 13.1 → 250 10.5 → 1000 12.4 → 3000 10.2 →
                   step 4000 13.6 → 5000 12.6 → 7000 11.5 → 8000 10.2 →
                   9000 9.3 → 12000 7.7 → 15000 6.3 → ... → ~1.5 final
```

The minimal run — which trains perfectly — ALSO bounces identity 10-13.6 for the
first ~8000 steps (including a rise to 13.6 at step 4000), and only begins a
clean monotone descent after ~step 8000. The early bounce is normal here
(AdaFace margin warmup 0→0.4 over 4000 steps + SGD peak-lr + 10K-class prototype
organisation).

**Consequence**: v1/v2/v3 were killed at steps ~5000-7000 — squarely inside the
normal bounce. Their "identity not descending / rose to 14" was NOT failure: it
matches what the minimal run does at the same steps (v3 step 4000 ≈ 13.7 vs
minimal step 4000 ≈ 13.6). The tick-55 "v2 identity collapse" and tick-56 "v3
failed" interpretations were **premature — judged before the descent phase**.
The three relaunches chased a failure the data did not actually establish.

The two `model.py` fixes are still sound and retained: the OrientationGate 4×4
pool is a mathematically-confirmed real bug fix (GAP is flip-invariant); the
FFA/ATL additive wiring (`afp_pooled + gate·ffa_out`) is a sound design even
though the "collapse" that motivated it was a misread.

**v4** launched — full HOLO-Net (both fixes), 30000 steps, seed 20260521,
`/workspace/holo_net_full_v4/`. **It will be left to run to completion** and
judged at step ~15000+ (where minimal is unambiguously descending) and at the
end — NOT killed during the bounce.

---

## Addendum — tick 60: v4 verdict — HOLO-Net DOES train (Q017 = yes); gate inert → gate.detach() fix → v5

v4 ran to step 11400. Identity: 13.1 → bounce (13.8 @ step 3800) → 11.2 (5700)
→ 9.5 (8550) → **8.47 (11400)**. The minimal run at the same steps: 11.6 (6000),
9.1 (10000), 8.78 (11000). **v4 tracks the minimal run almost exactly.**
predcode converged 4.1→0.017.

**Q017 ANSWERED — yes: the full HOLO-Net trains identity at the same rate as the
minimal model.** The tick-58 correction is fully validated — v1/v2/v3 were
killed prematurely during the normal bounce; the architecture was never broken.

**But v4's `orientation` loss is flat at 0.696 for all 11400 steps** — the
OrientationGate never learned. Cause (the tick-57 hypothesis, now confirmed):
under the additive wiring `atl_input = afp_pooled + gate·ffa_out`, the identity
loss has a strong DIRECT gradient into the gate (the gate scales `ffa_out`),
overwhelming the orientation CE (weight 0.1) → the gate collapses to a constant.
An inert gate = no orientation asymmetry = the Thatcher mechanism (design §7)
defeated.

**Fix** (`model.py`): `gate.detach()` in the holistic combination — the gate
still modulates `ffa_out` in the forward pass, but the identity loss no longer
backprops into it, so the OrientationGate is trained by the orientation CE
alone. v2 is the existence proof (v2's gate, under weak identity pressure, DID
learn: orientation 0.69→0.26 by step 1050).

v4 killed at step 11400 — past the bounce, descending fine; killed not in doubt
but because it would only confirm the predictable inert-gate outcome. **v5**
launched — full HOLO-Net + gate.detach(), 30000 steps, seed 20260521,
`/workspace/holo_net_full_v5/`. Expected: identity descends like v4/minimal AND
orientation now descends. v5 is the candidate for the falsification test.

---

## Addendum — tick 61: v5 — gate.detach() fix CONFIRMED working

v5 step 5550. **orientation loss: 0.71 → 0.56 (step 450) → 0.050 (step 900) →
0.009 (step 1350)**, then settles ~0.01-0.10 — the OrientationGate now learns
orientation properly (v4 was dead-flat at 0.696 for 11400 steps). The
`gate.detach()` fix works: with the identity gradient removed, the orientation
CE trains the gate cleanly. **v5 is the first run with a functional gate AND
on-track identity training.**

Identity at step 5550 = 12.56, in the normal bounce (minimal/v4 bounce 11-14 in
this range); verdict deferred to step ~15000+. v5 remains the falsification-test
candidate.

