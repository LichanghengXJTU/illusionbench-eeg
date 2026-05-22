# State

**Tick #**: 35 (Refresh Figure 2 + Figure 5 with 25 priors)
**Last updated**: 2026-05-22 (Asia/Hong_Kong)
**Current focus** (one sentence): Continued figure-refresh tick. Updated `figures/face_vs_randombbox_polished.py` and `figures/scaling_law.py` with new prior classes (Bio-inspired, Face-AM, Face-triplet), regenerated `_v2.{png,pdf}` for both.
**Last action**: Edited PRIOR_CLASS / CLASS_ORDER / CLASS_COLOR / PRETTY_NAME / IMAGE_ENCODER_PARAMS_M / FAMILY tables; added Face-AM/Face-triplet/Bio-inspired scatter to scaling_law; regenerated figures.
**Last action outcome**: Figure 2 v2 shows AdaFace-IR50-CASIA face Thatcher ISI ≈ 3 vs random-bbox ≈ 1, while CORnet-S is at baseline for both — visually demonstrates Claim 3 generalizes to new priors. Figure 5 v2 shows Face-AM cluster does NOT lie on the CLIP/DINOv2 scaling curve — they're at log10(params)≈1.7 with widely varying ISI 1.2-3.0 driven by training data, not scale (E032 data-inversion finding visualized).
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 36): Two options: (a) download a small subset of THINGS-EEG2 test images for the AdaFace-target Route A (Claim 4 extension), (b) do a "weakest claim audit" — explicitly identify which claim has the lowest evidence quality and design an experiment to strengthen it. (b) is meta-work; (a) is empirical extension. Will likely do (a) but with a small budget (~10 min) to fetch images.
**Confidence in current best idea**: 9.0/10 for Idea-001 (paper-ready, all figures now reflect 25 priors); 8.2/10 for Idea-003.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- CORnet repo: `/workspace/eeg_repos/CORnet/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
