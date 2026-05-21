# State

**Tick #**: 31 (E033 RSA cross-prior taxonomy + CKA)
**Last updated**: 2026-05-22 (Asia/Hong_Kong)
**Current focus** (one sentence): E033 ran pairwise RSA across all 25 priors on the 800-stimulus Thatcher set + CKA between CLIP-H/14 and AdaFace-CASIA. Pivoted from tick 30's planned ATM-training (stimulus mismatch + 2-4h GPU budget) to a more tractable representational geometry analysis.
**Last action**: E033 — computed 25×25 Spearman RSA matrix + 7-pair CKA + hierarchical clustering at k=4/6/8. Generated heatmap figure.
**Last action outcome**: (1) 5-6 clean RSA clusters (CLIP+DINOv2+SigLIP merge into one "semantic" cluster; angular-margin face-rec is its own cluster; FaceNet-triplet separate; CORnet-S alone; MAE/VAE/untrained reconstructive cluster). (2) CKA(CLIP-H/14, AdaFace-CASIA) = 0.45 — AdaFace adds ~55% novel info over CLIP. (3) Paradigm-level dissociations (Thatcher vs Composite/Part-Whole) are fine-grained effects WITHIN the big semantic cluster — argues that the benchmark detects subtle structure missed by global RSA.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 32): Two candidates: (a) integrate E033 figure into PAPER_DRAFT as new Figure 3; (b) extend RSA to Composite + Part-Whole stimuli (separately) to see if cluster structure changes by paradigm. (a) is paper-progress; (b) is empirical extension. Will likely do (b) first since it's a 5-min job, then (a).
**Confidence in current best idea**: 8.9/10 for Idea-001 (mature + now backed by RSA taxonomy); 8.2/10 for Idea-003 (sub-path (h) motivated by AdaFace CKA novelty)

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- CORnet repo: `/workspace/eeg_repos/CORnet/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
