# State

**Tick #**: 1 (E003 random-bbox control)
**Last updated**: 2026-05-21 23:50 (Asia/Hong_Kong)
**Current focus** (one sentence): With Q001 now confirmed face-feature-specific, identify which model class most strongly encodes the configural Thatcher signature — face-recognition-trained (FaceNet/ArcFace) or large-scale CLIP-bigG.
**Last action**: E003 random-bbox FFHQ control completed; CLIP-bigG ISI dropped 6.76 → 1.97 (−71%); other CLIP variants 59-74% reduction; DINOv2-large/giant fell below 1.0.
**Last action outcome**: confirmed Q001 — face-feature-specificity dominates; small residual general orientation bias (~20-30%) remains in CLIP class.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 2): Q003 — add FaceNet (facenet-pytorch, VGGFace2-trained Inception-Resnet) + ArcFace to model zoo; extract on FFHQ face Thatcher; compute ISI. Hypothesis: ISI(face-trained) ≥ ISI(CLIP-bigG) if face-feature-specific component is from explicit face-recognition learning; ≤ if CLIP emergent property is the main driver.
**Confidence in current best idea (Idea-001)**: 7/10 (up from 5/10) — face-specificity confirmed, now narrowing the model-class story.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (H100 80GB, 2TB NVMe, 2TB RAM, 224 CPUs)
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
