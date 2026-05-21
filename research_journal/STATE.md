# State

**Tick #**: 2 (E004 face-trained baselines)
**Last updated**: 2026-05-22 00:15 (Asia/Hong_Kong)
**Current focus** (one sentence): Q007 — what specifically about CLIP-class training drives the orientation-emergent face-Thatcher sensitivity, given Q003 just refuted "face-recognition training is the cause"?
**Last action**: E004 added FaceNet (VGGFace2 + CASIA-Webface variants). Result: ISI=1.02-1.12 (face Thatcher), 0.38-0.46 (random-bbox). Face-recognition training does NOT produce Thatcher ISI; the signature is orientation-emergent in image-text contrastive training (CLIP) but is actively suppressed by pose-invariant identity training (FaceNet).
**Last action outcome**: Q003 ANSWERED (refuted); spawned Q007; Idea-001 score 7.0→7.7 with reframed pitch v2.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 3): Q007 — add SigLIP (image-text, larger pre-training data) + EVA-CLIP + MetaCLIP as image-text-training family comparison. Discriminates "any large image-text training emerges Thatcher" vs "CLIP-specific". Then Q004 Harmonized priors in tick 4 to test perception-aligned training.
**Confidence in current best idea (Idea-001)**: 7.7/10

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (H100 80GB, 2TB NVMe, 2TB RAM, 224 CPUs)
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
