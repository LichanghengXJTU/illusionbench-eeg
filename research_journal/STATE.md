# State

**Tick #**: 10 (E025 bbox-size sensitivity — Claim 3 strengthened)
**Last updated**: 2026-05-22 02:35 (Asia/Hong_Kong)
**Current focus** (one sentence): Claim 3 now STRONG with E003 + E025 jointly demonstrating the face-feature-specificity gap is robust to random-bbox size (CLIP random-bbox ISI 1.2-2.0 across 0.5×-1.5× scales while face-Thatcher 5-7); ready to begin writing actual paper sections.
**Last action**: E025 — added `--bbox_scale` to random-bbox generator; ran 0.5× and 1.5× variants; extracted on 6 key priors; computed ISI; demonstrated robust ~3-4× gap. CLIP-bigG14 RB ISI: 1.63 (0.5×) → 1.97 (1.0×) → 1.47 (1.5×). Face-Thatcher gap stable across scales.
**Last action outcome**: All 5 paper claims now have STRONG evidence. Idea-001 score held at 8.7/10 but every claim has been audited for robustness.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 11): Either (a) begin drafting actual Introduction prose for the paper, or (b) generate the polished publication-quality combined figure (claim-3 sensitivity panel + three-paradigm panel side by side). Both are write-phase tasks. Plan to do (a) Introduction first because it forces clarity of pitch.
**Confidence in current best idea (Idea-001)**: 8.7/10

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
