# State

**Tick #**: 30 (AuraFace RGB sanity + AdaFace IR-50 dataset sweep)
**Last updated**: 2026-05-22 (Asia/Hong_Kong)
**Current focus** (one sentence): E032 resolved AuraFace's PWI inversion as a BGR-preprocessing artifact (RGB-correct PWI = 0.99, baseline), AND discovered AdaFace IR-50 trained on small CASIA dataset gives the highest Thatcher ISI of any face-rec model (**2.91 [2.63, 3.22]**, comparable to mid-tier CLIP).
**Last action**: E032 — extracted P23rgb (AuraFace RGB control) + P26 (AdaFace IR-50 CASIA) + P28 (IR-50 WebFace4M) + P29 (IR-50 MS1MV2); computed all 4 paradigm metrics.
**Last action outcome**: AdaFace IR-50 CASIA is the strongest face-rec Thatcher signal (ISI 2.91), demonstrating inverted data-scaling (small noisy data → more Thatcher). 10+ distinct dissociation patterns now across 24 priors. AuraFace PWI inversion is a preprocessing artifact, not a real finding.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 31): Sub-path (h) — replace ATM's CLIP target with P26 AdaFace-CASIA embeddings; train ATM mini-version (1 subject, few epochs) on THINGS-EEG2 and measure: (1) does training converge? (2) does the EEG-side preserve the AdaFace Thatcher signal better than CLIP-side? Estimated 2-4h GPU.
**Confidence in current best idea**: 8.7/10 for Idea-001 (mature); **8.2/10 for Idea-003** (raised — AdaFace-CASIA gives a concrete proof-of-concept that face-rec can match modest CLIP Thatcher)

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- CORnet repo: `/workspace/eeg_repos/CORnet/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
