# State

**Tick #**: 4 (Q004 blocked, Q005 scoped via E006)
**Last updated**: 2026-05-22 01:00 (Asia/Hong_Kong)
**Current focus** (one sentence): Q005 EEG-side feasibility — Route A (use ATM trained checkpoint on THINGS-EEG2 test, compute per-dimension EEG preservation × Thatcher loading correlation) is the cleanest first experiment because we lack EEG of Thatcher stimuli.
**Last action**: Attempted Q004 Harmonized priors — blocked on TF/Keras 3 incompatibility loading .h5 saved-model files. Wrote NEED-001 to NOTES_FOR_USER. Pivoted to Q005 scoping: cloned ATM repo at `/workspace/eeg_repos/EEG_Image_decode`, confirmed ATM uses identical CLIP-ViT-H/14-LAION anchor as our P04 (ISI=5.50), refined Q005 into Route A experiment for tick 5. No new ISI numbers this tick — scoping artifact only.
**Last action outcome**: blocked on Q004 (notified user); Q005 now has a precise actionable plan.
**Running tasks** (on server): none
**Stuck streak**: 0 (productive pivot, not stuck)
**Planned next action** (tick 5): E020 Route A — (a) download ATM trained checkpoint from HF `LidongYang/EEG_Image_decode`, (b) download THINGS-EEG2 preprocessed test set (gasparyanartur/things-eeg2), (c) run ATM inference on test EEG, (d) compute per-CLIP-dim EEG-preservation correlation, (e) compute Thatcher-dim loading from existing FFHQ E002 CLIP embeddings, (f) test whether Thatcher-dims are preserved or destroyed by the EEG bottleneck.
**Confidence in current best idea (Idea-001)**: 8.0/10

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (H100 80GB, 2TB NVMe, 2TB RAM, 224 CPUs)
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
