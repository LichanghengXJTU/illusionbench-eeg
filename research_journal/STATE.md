# State

**Tick #**: 34 (PIVOT — 25-prior headline figure refresh)
**Last updated**: 2026-05-22 (Asia/Hong_Kong)
**Current focus** (one sentence): Original tick 34 plan (EEG-side AdaFace Route A) infeasible without THINGS-EEG2 image download. Pivoted to refreshing the headline figure (Figure 1) with all 25 priors instead of the legacy 17. Now paper Figure 1 reflects the full updated benchmark including bio-inspired (CORnet-S), face-AM (7 ArcFace/AdaFace variants).
**Last action**: Updated `figures/three_paradigm_polished.py` with extended PRIOR_CLASS / CLASS_ORDER / CLASS_COLOR / PRETTY_NAME tables, regenerated `figures/exports/three_paradigm_polished_v2.{png,pdf}` on server, pulled to local. Updated §4.1 paper reference. PAPER_DRAFT now references v2.
**Last action outcome**: Figure 1 (v2) clearly shows AdaFace-IR50-CASIA as ~3 ISI face-rec outlier; CORnet-S as PWI 0.21 anomaly; full 25-prior class coloring (CLIP red, DINOv2 orange, Face-AM dark green, Bio-inspired purple, etc.).
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 35): Two candidates: (a) update other polished figures (face_vs_randombbox, route_a, scaling_law) with the new priors where applicable, (b) actually download a small subset of THINGS-EEG2 test images (e.g., 200 representative images from OSF) to enable the AdaFace-target Route A in tick 36. (a) is paper-finishing; (b) extends Claim 4. Will likely do (a) first.
**Confidence in current best idea**: 9.0/10 for Idea-001 (paper-ready, figures updating); 8.2/10 for Idea-003.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- CORnet repo: `/workspace/eeg_repos/CORnet/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
