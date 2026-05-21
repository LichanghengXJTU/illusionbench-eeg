# State

**Tick #**: 0 (pre-launch)
**Last updated**: 2026-05-21 22:50 (Asia/Hong_Kong)
**Current focus** (one sentence): Determine whether the CLIP-class ISI hierarchy observed on FFHQ Thatcher stimuli is face-specific configural processing or general orientation-dependent perturbation sensitivity (Q001).
**Last action**: Computed ISI on FFHQ 1024 stimuli for 12 visual priors; observed CLIP-bigG14 ISI=6.76 with N03_pixel=1.000 sanity intact.
**Last action outcome**: confirmed Q-no-trivial-bug + revealed Q001 must be resolved before any interpretation
**Running tasks** (on server `103.207.149.173:11022`):
  - none
**Stuck streak**: 0
**Planned next action** (tick 1): Build non-face Thatcher control stimulus battery (E003) → extract embeddings on all 12 priors → compute ISI → compare with FFHQ face Thatcher ISI to discriminate Q001.
**Confidence in current best idea**: 5/10 (need C2 result before committing)

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (H100 80GB, 2TB NVMe, 2TB RAM, 224 CPUs)
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub remote: <pending — awaiting user PAT>
