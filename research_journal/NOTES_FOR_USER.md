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
