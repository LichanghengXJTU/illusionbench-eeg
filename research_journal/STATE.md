# State

**Tick #**: 25 (closed) → entering Idea-003 phase (bio-inspired vision prior for EEG decoding)
**Last updated**: 2026-05-22 06:35 (Asia/Hong_Kong)
**Current focus** (one sentence): User raised Idea-003 — flip the causal direction from "model behavior → speculate brain" (unscientific) to "brain mechanism → forward-modeled bio-inspired vision prior → test paradigm-consistency on IllusionBench-EEG + competitive retrieval". This is a sharper publishable contribution; cheapest go/no-go is testing CORnet-S.
**Last action**: (a) Idea-003 added to IDEA_PIPELINE with full critique + failure modes + downstream-application arguments. (b) Q010 spawned. (c) Tick 25 E027 2AFC sanity check (Spearman ρ 0.93-0.97 across 3 paradigms) committed. Loop continues into tick 26.
**Last action outcome**: Pivot decision confirmed — bio-inspired direction is high-value. Need go/no-go on CORnet-S immediately.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 26): E028 — install CORnet-S checkpoint, add to model registry, extract on all 3 paradigms (Thatcher + Composite + Part-Whole), compute ISI/CSI/PWI, judge paradigm-consistency. If positive, expand to CORnet-S 10 time settings + Hybrid CORnet (from user's Harvard preliminary).
**Confidence in current best idea**: 8.7/10 for Idea-001 (mature); 8.0/10 for Idea-003 (provisional, depends on tick 26 result)

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
