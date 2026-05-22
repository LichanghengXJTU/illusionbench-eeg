# E044 — HOLO-Net v5: pre-registered falsification test — RESULT

**Status**: COMPLETE — the decisive Idea-003 experiment.
**Linked**: Idea-003, E042 (v5 training), E041 (minimal ablation floor),
`IDEA_003_HOLO_NET_DESIGN.md` §6.

**Hypothesis (pre-registered, design §6)**: a HOLO-Net trained only on Glint360K
identities (no illusion stimuli) will, at the FFA layer, SIMULTANEOUSLY satisfy
Thatcher ISI ≥ 3.0, Composite CSI ≥ 1.5, Part-Whole PWI ≤ 0.5, Random-bbox
ISI ≤ 1.5 — a combination no prior in the 25-prior battery achieves.

**Method**: v5 = full HOLO-Net (all bio components; all three model.py fixes —
AdaFace loss, OrientationGate 4×4 pool, FFA/ATL additive wiring + gate.detach()),
trained to step 30000 on Glint360K-10K (final identity loss 0.97, orientation
0.16 — both converged). Evaluated via `eval_falsification.py` over the 4
IllusionBench stimulus sets, per brain-region layer, pixel-corrected, seed 20260521.

## Result (pixel-corrected; numbers only)

| layer | ISI | CSI | PWI | ISI_rbox |
|---|---|---|---|---|
| v1  | 1.020 | 1.147 | 1.261 | 1.017 |
| v2  | 1.036 | 1.182 | 0.741 | 0.998 |
| v4  | 0.997 | 1.127 | 0.747 | 1.025 |
| mfp | 0.993 | 1.085 | 0.752 | 1.115 |
| afp | 1.018 | 0.967 | 2.068 | 1.073 |
| ffa | 1.002 | 0.991 | 2.203 | 1.076 |
| atl | 0.996 | 1.103 | 0.813 | 1.100 |

**FFA-layer verdict vs design §6**:
- Thatcher ISI = 1.002 — need ≥ 3.0 → **FAIL**
- Composite CSI = 0.991 — need ≥ 1.5 → **FAIL**
- Part-Whole PWI = 2.203 — need ≤ 0.5 → **FAIL**
- Random-bbox ISI = 1.076 — need ≤ 1.5 → PASS (control)
- **ALL FOUR SIMULTANEOUSLY: FAIL.**

**HOLO-Net v1.0 does NOT pass its pre-registered falsification.**

## Interpretation

- **[CONFIRMED]** v5 — despite training identity well (loss 0.97) with a
  functional Orientation Gate — shows NO Thatcher effect (ISI ≈ 1.0 at *every*
  layer) and NO composite effect (CSI ≈ 1.0). Versus the minimal ablation floor
  (E041: ISI 0.975, CSI 1.192): the full bio architecture did NOT lift Thatcher
  or composite above the no-bio-components baseline. The bio-fidelity components
  do not, on their own, emerge configural-illusion sensitivity.

- **[CONFIRMED]** The FFA module is NOT inert — it dramatically restructures the
  part-whole geometry: PWI rises mfp 0.75 → afp 2.07 → ffa 2.20 (minimal floor
  ffa = 0.78). The FFA holistic binding has a strong, specific effect on the
  part-whole paradigm — but OPPOSITE to the ≤0.5 criterion: it *amplifies* the
  eye-swap distance in whole-face context (the criterion, set from DINOv2,
  wanted dampening). PWI 2.20 is the single largest paradigm deviation HOLO-Net
  produces — the architecture changes something real, just not as predicted.

- **[CONJECTURE — connects to Idea-001]** Thatcher/composite do not emerge
  because HOLO-Net is trained with a FACE-IDENTITY objective (AdaFace).
  Idea-001's E004 already established face-identity-trained models (FaceNet)
  give ISI ≈ 1.0, while image-text contrastive training emerges ISI 4-7
  (E002/E005). v5 — a maximally bio-faithful face architecture trained on
  identity — giving ISI ≈ 1.0 is fully consistent: **the bio-architecture does
  not substitute for the training objective. The HOLO-Net negative result
  CONFIRMS Idea-001's central thesis from the architecture side — configural-
  illusion emergence is driven by the training objective, not architectural
  bio-fidelity.**

- **[CONJECTURE]** The design §7 Thatcher mechanism is conceptually flawed: the
  Orientation Gate distinguishes WHOLE-FACE upright vs inverted, but Thatcher is
  a LOCAL-feature inversion within an upright face — to the gate, both V1
  (upright-normal) and V2 (upright-thatched) are "upright", so the gate gives no
  Thatcher-specific signal. The architecture has no mechanism that specifically
  responds to local-feature-inversion-conditioned-on-upright.

## Replicability

- v5 checkpoint `/workspace/holo_net_full_v5/checkpoint.pt` (step 30000, seed
  20260521); `eval_falsification.py` at the tick-62 commit.
- Tables: `outputs/tables/HOLONET_v5_{thatcher,composite,partwhole,randombbox}.csv`.

## Bottom line

Idea-003's pre-registered hypothesis is **REFUTED** — HOLO-Net v1.0 is not the
"first paradigm-consistent model". But the result is rigorous and coherent, and
it is load-bearing for Idea-001 (it confirms the training-objective thesis
architecturally). Strategic fork → `NOTES_FOR_USER.md` FLAG-002.
