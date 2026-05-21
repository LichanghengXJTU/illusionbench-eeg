# State

**Tick #**: 26 (CORnet-S go/no-go)
**Last updated**: 2026-05-22 06:50 (Asia/Hong_Kong)
**Current focus** (one sentence): E028 reveals that bio-inspired anatomy alone (CORnet-S, ImageNet) does NOT produce Thatcher illusion (ISI 0.99 = baseline), but DOES produce strongest Part-Whole effect (PWI 0.214, lowest of all 18 models). Idea-003 needs refinement — path forward is face-tuned CORnet + multi-anchor decoders.
**Last action**: E028 CORnet-S on 3 paradigms + random-bbox control. Pre-registered prediction partially refuted: paradigm-consistency NOT achieved by anatomy alone. But Part-Whole result is a distinct positive contribution.
**Last action outcome**: Idea-003 score 8.0 → 7.3 (provisional). Sub-paths (a) face-tuned CORnet and (d) multi-anchor decoder now leading candidates.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 27): Two parallel directions to test —
  (a) **Face-tuned CORnet**: train CORnet-S architecture on face-recognition objective. Or look for a pretrained face-CORnet. Tests whether bio-anatomy + face-data combination gives the missing Thatcher signal.
  (d) **Multi-anchor decoder simulation**: simulate a "fused embedding" by concatenating CLIP + CORnet features, compute ISI/CSI/PWI on the fused space. Tests whether the fusion has paradigm-consistent illusion sensitivity (CLIP gives Thatcher, CORnet gives Part-Whole).
  Pick (d) for tick 27 — cheaper, only needs concatenation + reuse of existing NPZs.
**Confidence in current best idea**: 8.7/10 for Idea-001 (mature); 7.3/10 for Idea-003 (provisional, refined sub-paths)

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- CORnet repo: `/workspace/eeg_repos/CORnet/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
