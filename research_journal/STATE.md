# State

**Tick #**: 24 (Scaling law)
**Last updated**: 2026-05-22 06:15 (Asia/Hong_Kong)
**Current focus** (one sentence): Q008 fully answered — scale and language contributions decompose cleanly: same slope (~1.75 per log10 params) for image-text and SSL families, but image-text adds a +2.9 ISI constant offset.
**Last action**: E026 scaling-law analysis. Fit per-family linear regressions on log-params: image-text ISI ≈ 1.71 × log10(M) + 0.82; DINOv2-SSL ISI ≈ 1.78 × log10(M) − 2.06. Generated `scaling_law.png` figure. Wrote E026 with predictive consequences.
**Last action outcome**: A clean decomposition publishable result. Idea-001 8.7/10 maintained.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 25): The remaining high-value items from the Future Work menu are blocked or larger: (A) cross-decoder Route A is blocked on no released embeddings; (B) multi-prior decoder needs training; (C) 2AFC verification is doable but adds a few ticks; (D) non-face illusions extend benchmark substantially. Given that the project deliverable is now feature-complete + with this scaling-law refinement, consider whether to (a) implement 2AFC verification, (b) start non-face illusions, or (c) wrap up.
**Confidence in current best idea (Idea-001)**: 8.7/10 + clean scaling decomposition supplementary

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
