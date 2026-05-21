# State

**Tick #**: 27 (Fusion simulation negative result)
**Last updated**: 2026-05-22 07:05 (Asia/Hong_Kong)
**Current focus** (one sentence): E029 showed simple multi-anchor fusion = distance averaging with no emergent paradigm-consistency; Idea-003 sub-path (d) ruled out. Next: explore off-the-shelf face-tuned bio-inspired models (sub-path a candidate is to check whether FaceCORnet exists or can be quickly trained).
**Last action**: E029 CLIP+CORnet fusion at 6 weight points. All produce weighted-average ISI/CSI/PWI — no fusion configuration achieves paradigm-consistency.
**Last action outcome**: Idea-003 sub-path (d) refuted. Sub-path (a) remains most promising but requires custom training. Idea-003 score held at 7.3 pending experimental tractability.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 28): Two-step exploration —
  (1) Literature: confirm no off-the-shelf face-CORnet exists; check NSD-aligned PyTorch encoders availability
  (2) If face-CORnet doesn't exist as checkpoint, attempt quick fine-tuning: CORnet-S last 1-2 layers on a face-classification head with VGGFace2 subset. Time-budget 1-2 hours of GPU work, see if Thatcher signal emerges.
**Confidence in current best idea**: 8.7/10 for Idea-001 (mature); 7.3/10 for Idea-003 (provisional, paths becoming clearer)

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- CORnet repo: `/workspace/eeg_repos/CORnet/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
