# State

**Tick #**: 40 (PIVOT — HOLO-Net design doc for Idea-003 sub-path e)
**Last updated**: 2026-05-22 (Asia/Hong_Kong)
**Current focus** (one sentence): User pivoted from RSA robustness check to substantive Idea-003 design — propose a NEW bio-fidelity model architecture (HOLO-Net) where each layer maps to a specific brain region.
**Last action**: Wrote `IDEA_003_HOLO_NET_DESIGN.md` (12 sections, ~3500 words). Brain-region layer mapping: LGN(Magno+Parvo) → V1/V2/V4/IT (CORnet-S backbone) + OFA branch + AFP view-invariance + FFA 3-step recurrent self-attention with Magno-gist query + Orientation Gate + Predify-style top-down predictive coding feedback + ATL AdaFace identity head. Web-search lit review for face patches (Tsao & Freiwald 2010), magno gist (Bar 2003), predictive coding (Rao & Ballard 1999, Predify package), CORnet-S (Kubilius 2019), OFA/FFA/ATL hierarchical face processing (Collins & Olson 2014). Pre-registered falsification: at FFA layer must simultaneously satisfy ISI≥3, CSI≥1.5, PWI≤0.5, Random-bbox≤1.5 (no current prior achieves this). Training plan: ~155GB Glint360K + 10% ImageNet on H100 for 3-5 days.
**Last action outcome**: User has detailed design doc to review. Stopped previous tick-40 monitor (RSA robustness check) since it's superseded.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 41): AWAITING user decisions on §12 of design doc (falsification thresholds OK? Predify vs custom PC? Calendar budget? Capsule/GLOM fallback?). After greenlight: tick 41 = PyTorch architecture skeleton, tick 42 = Predify integration + dual-stream sanity test, tick 43 = Glint360K dataset acquisition (~140GB), tick 44+ = training + evaluation.
**Confidence in current best idea**: 9.3/10 for Idea-001 (paper-ready, stable); **8.5/10 for Idea-003** (raised — concrete architectural design with falsification criteria, replaces the recipe-level sub-path (a)).

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- CORnet repo: `/workspace/eeg_repos/CORnet/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
