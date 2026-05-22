# State

**Tick #**: 36 (E035 per-subject Route A — Claim 4 strengthening)
**Last updated**: 2026-05-22 (Asia/Hong_Kong)
**Current focus** (one sentence): Weakest-claim audit identified Claim 4 (EEG uniform low-pass) as the weakest. E035 strengthens it by showing the result holds INDIVIDUALLY for each of 10 THINGS-EEG2 subjects — not a subject-averaging artifact.
**Last action**: Web search confirmed no public pre-computed EEG embeddings beyond ATM-S exist for THINGS-EEG2 (NICE-EEG releases weights only; AVDE/ENIGMA/ViEEG no releases). Pivoted to per-subject consistency analysis. Computed per-dim Pearson r + Spearman(r, Thatcher-loading) per subject.
**Last action outcome**: 8/10 subjects show negative Spearman(r, loading) < 0; 9/10 have p > 0.1; mean r per subject in [0.126, 0.212]. Pooled distribution of 10240 (dim, subject) r values is roughly Gaussian centered at 0.158, no bimodality. Claim 4 raised from MEDIUM-STRONG to STRONG.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 37): Three candidates: (a) audit Claim 6 (face-rec data inversion) by checking additional small/noisy face datasets (e.g., LFW, AgeDB); (b) integrate E035 figure into paper Figure 4 or 4b; (c) update ABSTRACT to reflect 7 claims including E035 strengthening. (c) is paper-finishing, fastest.
**Confidence in current best idea**: 9.2/10 for Idea-001 (Claim 4 raised STRONG); 8.2/10 for Idea-003.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- CORnet repo: `/workspace/eeg_repos/CORnet/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
