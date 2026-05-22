# E043 — HOLO-Net component-isolation ablation: which component breaks identity training?

**Status**: IN PROGRESS (diag-A, diag-B launched tick 56). Answers Q016.
**Linked**: Idea-003, E042, Q016, E040 (minimal baseline).

## Background

The minimal HOLO-Net (all bio components OFF) trains identity cleanly
(E040: loss 13 → 1.8). The full HOLO-Net does NOT — across three runs:

- **v1** (buggy AdaFace loss + inert OrientationGate): identity drifted 13→11,
  very slow.
- **v2** (gate fixed): identity bounced 11-14, no net descent.
- **v3** (gate + FFA/ATL wiring fixed): identity descended 13→10.6 by step 3000,
  then ROSE to 14.1 by step 4550; orientation went flat at ln 2 again.

Three patch-and-relaunch attempts have not produced a trainable full model.
Switching from guessing to systematic component isolation.

## Hypothesis (Q016)

Some specific bio component — or the multi-task aux losses — breaks the full
model's identity optimisation. Two candidate groups:

- **(A)** the FFA holistic term added into the identity representation
  (`atl_input = afp_pooled + gate·ffa_out`);
- **(B)** the auxiliary losses (predcode, face-detect, gist) + the LGN-Magno
  dual stream, which reshape the shared backbone.

## Method

From the known-good minimal base (which trains), re-enable component groups one
at a time via the new `train.py --enable` flag. Two parallel runs, everything
else identical to the minimal run (SGD 0.1, num_classes 10000, batch 512,
seed 20260521, 8000 steps):

- **diag-A** `--enable ffa`: minimal backbone + FFA holistic term, NO aux
  losses. Isolates group (A). identity path = `afp_pooled + ffa_out`.
- **diag-B** `--enable pc,ofa,gist,magno`: minimal backbone + LGN-Magno + the
  3 aux losses, NO FFA. Isolates group (B). identity path = `afp_pooled`.

## Pre-registered prediction

- If diag-A's identity fails to descend → the FFA holistic term corrupts the
  identity representation.
- If diag-B's identity fails → the aux losses / LGN-Magno corrupt the backbone.
- If both descend like the minimal run → the culprit is the Orientation Gate
  or a component interaction (next ablation: `+orient`, then pairs).

## Result (numbers only)

PENDING — diag-A at step 550 (identity 12.1), diag-B at step 100 (identity
12.9). Read the descent trend next tick (~step 4000+, past the AdaFace margin
warmup that completes at step 4000 — the point where v3's identity blew up).

## Replicability

`holo_net/train.py --enable` at the tick-56 commit; seed 20260521.
Outputs: `/workspace/holo_net_diagA_ffa/`, `/workspace/holo_net_diagB_aux/`,
`/workspace/holo_net_diagD_ffa_orient/`.

---

## Addendum — tick 57: diag-A/B preliminary (inconclusive); diag-D added

At step ~3000 both diagnostics look **similar and not catastrophic**:
- diag-A (+ffa): identity 13.2 → 10.6 (step 3050), descending with an early
  bounce.
- diag-B (+aux): identity 13.0 → 10.6 (step 2600), same shape; predcode
  4.1→0.5, face_detect→0.02, gist 0.9→0.18 (the aux losses themselves descend).

So neither the FFA holistic term alone nor the aux-loss group alone breaks
identity in the first ~3000 steps — both track roughly like the minimal run.
**But the decisive window is step 3000-5000**: that is where v3's identity blew
up (10.6 → 14.1), coinciding with the AdaFace margin warmup completing at step
4000. diag-A/B have not reached it yet — verdict deferred to next tick.

**diag-D launched** `--enable ffa,orient` — FFA holistic term + a *functional*
Orientation Gate, i.e. `atl_input = afp_pooled + gate·ffa_out` with a real gate
(v3's holistic mechanism, minus pc/ofa/gist/magno). If diag-A trains but diag-D
does not, the gate×FFA interaction is the culprit. diag-D step 50, identity 13.3.
