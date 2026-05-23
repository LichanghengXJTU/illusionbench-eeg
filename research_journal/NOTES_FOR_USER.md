# Notes for user

Append-only async channel from the autonomous research loop to the user. The loop
writes here when it discovers a NEED that requires user action (new resource,
paid access, lab data, etc.). The loop does NOT stop — it picks the next-best
action and keeps going.

User: to resolve a NEED, edit this file and add a `RESOLVED:` line under that
entry (or commit/push from your phone via GitHub). The loop scans this file each
tick.

---

## NEED-001 — 2026-05-22 00:50 — Harmonized priors (Q004) blocked on TF/Keras compat

- **Why needed**: Q004 — perception-aligned visual prior comparison (Serre lab harmonized models trained with ClickMe-attention loss). Tests whether perceptual alignment training boosts/reduces face-Thatcher ISI vs vanilla CLIP/ViT counterparts.
- **What I tried first**:
  1. `pip install harmonization` ✓ installed
  2. Added missing deps: `xplique`, `efficientnet`, `vit_keras` ✓ installed
  3. `load_ResNet50()` downloads weights OK but fails on `tf.keras.models.load_model(.h5)` — new Keras 3 (default with TF 2.16+) refuses to load old TF 2.x .h5 format. Error path: `keras/src/saving/saving_api.py:218`.
- **Estimated cost / time / access**: 30 min if working tensorflow+keras 2.15 wheels exist for Python 3.12; otherwise needs downgrade to Python 3.11 + tf-2.15.
- **What blocks if not provided**: Q004 cannot be tested with the official Harmonization checkpoints.
- **What I am doing in the meantime**: Pivoting to Q005 EEG-side feasibility — cloning ATM repo, scoping what's needed to run their full inference pipeline on Thatcher stimuli. This is higher-value for Idea-001 anyway.
- **Awaiting**: User suggestion — options: (a) approve a Python 3.11 + TF 2.15 secondary venv, (b) approve porting Harmonized weights to PyTorch (significant effort), (c) drop Harmonized and use DINOv2+THINGS-similarity-finetune as the perception-aligned fallback per skills-roadmap, (d) defer Q004 indefinitely. The loop continues regardless.

---

## FLAG-001 — 2026-05-22 12:15 — HOLO-Net full architecture does not train; decision fork ahead (not a resource NEED — the loop is proceeding)

- **What happened**: The full bio-fidelity HOLO-Net (Idea-003, sub-path e) failed
  to optimize in Stage-2 face fine-tuning — identity loss stuck at 63-69 for 8750
  steps, no learning (E040 run v1). It only trains after disabling **every**
  bio-fidelity component (LGN-Magno, OFA, FFA, Orientation Gate, Predify PC,
  PFC-Gist) → "minimal mode". Minimal mode ≈ Idea-003 sub-path (a) (CORnet
  anatomy + AdaFace + Glint360K), NOT the HOLO-Net thesis. The run currently on
  the H100 (step ~16K/30K) is therefore a sub-path-(a) experiment.
- **Why it matters**: the pre-registered HOLO-Net falsification criteria are
  measured at the FFA layer — which is disabled in the only config that trains.
  Idea-003's headline novelty is, as of tick 50, untested.
- **The fork (the loop's default is option 1; tell it if you want otherwise)**:
  1. **[default]** Let the minimal run finish → evaluate it on IllusionBench
     (E041) as a clean sub-path (a) datapoint → then run Q014 component-ablation
     ladder to isolate which bio component breaks the full model. Keeps Idea-003
     alive at low cost.
  2. **Debug-first**: stop the minimal run now and go straight to the Q014
     ablation ladder on the full architecture.
  3. **De-scope**: treat HOLO-Net as future work; Idea-001 (the IllusionBench-EEG
     paper, score 9.3, paper-ready) is the deliverable. Idea-003 does not gate it.
- **What blocks if not decided**: nothing — the loop proceeds on option 1. This
  flag exists so you can redirect from your phone (edit this file with a
  `RESOLVED:` line) if you prefer option 2 or 3.
- **Idea-001 status**: unaffected, paper-ready, score 9.3/10.

**RESOLVED: 2026-05-22 12:48 (tick 51)** — User chose the debug-first fork and
directed full focus on Idea-003. Diagnosed within the same tick: the full-model
failure was an AdaFace loss-formula bug, not the bio architecture. E042
controlled re-test confirms the FULL HOLO-Net trains (starts at identity loss
13.20 ≈ minimal's 13.13). HOLO-Net is unblocked; the loop is now training the
full architecture. No GPU shortage yet — the H100 hosts both the full run and
the minimal-ablation run concurrently (40/80 GB). Will raise a NEED here if a
larger run (full 360K identities, or parallel ablations) needs more GPU.

---

## FLAG-002 — 2026-05-23 05:30 — HOLO-Net v5 was REFUTED by its own pre-registered falsification; strategic fork (loop is proceeding on a synthesis tick — not a resource NEED)

- **What happened**: v5 — the fully-trained full HOLO-Net (identity loss 0.97,
  functional Orientation Gate, all 3 bug fixes) — was evaluated on the
  pre-registered falsification (E044). FFA-layer result: Thatcher ISI 1.00,
  Composite CSI 0.99, Part-Whole PWI 2.20, Random-bbox 1.08 → **fails 3 of 4
  criteria; ALL FOUR: FAIL.** HOLO-Net v1.0 is not the "first paradigm-consistent
  model" the design hoped for. The eval pipeline is validated (it gave the
  expected ablation-floor result for the minimal model) — this is a real
  negative result, not a measurement artifact.

- **Why (best current reading)**: HOLO-Net is trained with a face-IDENTITY
  objective (AdaFace). IllusionBench-EEG (Idea-001) already established that
  face-identity-trained models (FaceNet) show ISI ≈ 1.0, while image-text
  contrastive training emerges ISI 4-7. So a bio-faithful architecture trained
  on identity giving ISI ≈ 1.0 is consistent: **the training objective, not
  architectural bio-fidelity, is the lever.** The HOLO-Net negative result
  *confirms Idea-001's central thesis from the architecture side.*

- **The fork (your call; the loop continues meanwhile on a synthesis tick)**:
  1. **Reframe & consolidate (loop's tentative lean)** — make it ONE paper:
     IllusionBench-EEG (the benchmark + dissociation) + the HOLO-Net result as
     a rigorous mechanistic negative ("a maximally bio-faithful face model,
     trained on identity, does NOT develop configural-illusion sensitivity →
     the effect is objective-driven"). Honest, coherent, publishable — but
     HOLO-Net is a *supporting* negative result, not the headline positive.
  2. **Re-aim HOLO-Net at the objective** — retrain the HOLO-Net architecture
     with an image-text contrastive objective (the lever Idea-001 identifies)
     instead of / alongside AdaFace identity. If configural illusions then
     emerge, HOLO-Net *can* be a positive headline. Cost: needs face
     image-text/caption data or a CLIP-distillation target (HOLO-Net currently
     trains on Glint360K identity labels only — no captions); ~1 day to set up
     + train. This is the scientifically-motivated next experiment.
  3. **Design-§8 architecture revisions** (6-step FFA, Capsule/GLOM). The loop's
     honest assessment: likely the WRONG lever — if the objective drives
     emergence (per 1 above), more architecture under the same identity
     objective will still give ISI ≈ 1. Not recommended as the first move.

- **What blocks if not decided**: nothing immediately — the loop will spend the
  next tick(s) on a synthesis (per-layer diagnosis, the Idea-001 connection,
  integrated write-up) which is useful under ANY of the three options. But the
  direction (especially option 2, which needs a data decision) wants your
  steer. Edit this file with a `RESOLVED:` line, or reply.

- **Idea-001 status**: unaffected and strengthened — the HOLO-Net result is
  corroborating evidence for it. Idea-001 score 9.3/10.

**RESOLVED: 2026-05-23 09:30 (tick 71)** — User picked a 4th option:
**diagnostic-first**. Hold off on a new training run; deeply analyse whether
v5 failed because the brain-science mechanism wasn't well imitated, or for
some other reason. The loop is now writing `HOLONET_DIAGNOSTIC.md` — a
neuroscience-grounded per-component critique. First-pass finding (lit-grounded):
HOLO-Net's design §7 Thatcher mechanism is **not faithful** to the actual
neural mechanism (Psalta et al. 2014 attribute Thatcher to orientation-
sensitive LOCAL feature detectors, not a global up/down gate; the
illusion-specific signal lives in fSTS, which HOLO-Net does not have at all).
Architecture-implementation flaws appear primary. Detailed write-up + a
proposed v2.0 redesign coming this tick / next.
