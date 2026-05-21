# State

**Tick #**: 28 (Literature scan + ArcFace surprise)
**Last updated**: 2026-05-22 (Asia/Hong_Kong)
**Current focus** (one sentence): E030 closed sub-paths (b)(c) (no off-the-shelf bio-inspired face encoder), confirmed FaceCLIP variants don't actually fine-tune the CLIP encoder, AND discovered ArcFace (angular-margin loss) gives ISI 1.70 on Thatcher (vs FaceNet 1.12) — refining sub-path (a) to "face-CORnet trained with angular-margin loss".
**Last action**: Added P23_arcface_auraface to registry, extracted on 3 paradigms + random-bbox. Updated E030 + IDEA_PIPELINE Idea-003 sub-paths.
**Last action outcome**: ArcFace **shows Thatcher** (ISI 1.70 [1.64, 1.77], 77% face-specific) — falsifying the broad "face-data alone is insufficient" hypothesis from E028. Also inverts Part-Whole (PWI 1.43) — fourth distinct failure mode in our taxonomy. **Five distinct dissociation patterns confirmed across 19 priors**.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 29): Sub-path (f) — loss-function ablation. Test CosFace / MagFace / SphereFace (additional angular-margin face-recognition models) on 3 paradigms to confirm "angular-margin loss → Thatcher signal" pattern. If confirmed, design face-CORnet + ArcFace-loss custom training plan for tick 30+ (multi-hour). Estimated tick 29: 1-2 hours of work.
**Confidence in current best idea**: 8.7/10 for Idea-001 (mature, publishable); 7.5/10 for Idea-003 (provisional, sub-path (a) revived by E030 finding)

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- CORnet repo: `/workspace/eeg_repos/CORnet/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
