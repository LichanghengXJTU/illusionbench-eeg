# State

**Tick #**: 29 (Angular-margin loss-family ablation, AdaFace surprise)
**Last updated**: 2026-05-22 (Asia/Hong_Kong)
**Current focus** (one sentence): E031 tested AdaFace IR-101 MS1MV2 (P24) and ArcFace IR-101 WebFace4M (P25); falsified "angular-margin loss → Thatcher" broad hypothesis, but discovered AdaFace IR-101 MS1MV2 is the most paradigm-consistent face-recognition model (ISI 1.24, CSI 1.60, PWI 0.59 — all in expected direction across 3 paradigms).
**Last action**: E031 — extracted CVLFace AdaFace IR-101 + ArcFace IR-101 WebFace4M; computed ISI/CSI/PWI on 3 paradigms + random-bbox.
**Last action outcome**: 7 distinct dissociation patterns now identified across 21 priors. AdaFace IR-101 is best-balanced face-rec model (all 3 indices deviate from baseline in expected direction). AuraFace's Thatcher (1.70) + inverted PWI (1.43) is NOT replicated on IR-101 ArcFace (ISI 1.17, PWI 0.50) — confounded by backbone/data, not loss alone.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 30): Sanity check — re-extract AuraFace with RGB preprocessing to rule out BGR confound; also test AdaFace IR-50 variants on CASIA/VGG2/WebFace4M to disentangle data scale effect from architecture effect. 1-2h work.
**Confidence in current best idea**: 8.7/10 for Idea-001 (mature, publishable); 7.5/10 for Idea-003 (sub-path (a) refined toward AdaFace-style training)

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- CORnet repo: `/workspace/eeg_repos/CORnet/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
