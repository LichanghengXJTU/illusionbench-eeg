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
