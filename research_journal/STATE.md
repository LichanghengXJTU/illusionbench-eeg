# State

**Tick #**: 3 (E005 image-text training family)
**Last updated**: 2026-05-22 00:35 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-001 score now 8.0 (publishable threshold). Next: test Harmonized perception-aligned priors (Q004) to see if they exhibit the same Thatcher signature as CLIP, then plan Q005 EEG-side feasibility.
**Last action**: E005 added SigLIP-base (ISI=3.51), SigLIP-SO400M (ISI=5.59), MetaCLIP-H/14 (ISI=6.08) to model zoo. All consistent with CLIP family (ISI 4-7) on face Thatcher, with same ~70% face-feature-specificity ratio (random-bbox drop). Q007 partially answered: family-general not CLIP-specific.
**Last action outcome**: Confirmed image-text-training-family-general phenomenon; Idea-001 → 8.0 (novelty 8 + feasibility 8 + evidence 8).
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 4): Q004 — add Harmonized priors (Serre lab: harmonized-ResNet50 / ViT-B / EfficientNet via `pip install harmonization` or git clone). Tests whether ClickMe-aligned perceptual training boosts/reduces Thatcher ISI vs vanilla CLIP/ViT counterparts.
**Confidence in current best idea (Idea-001)**: 8.0/10

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (H100 80GB, 2TB NVMe, 2TB RAM, 224 CPUs)
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
