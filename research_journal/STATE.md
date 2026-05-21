# State

**Tick #**: 8 (E024 Part-Whole PWI — three-paradigm benchmark complete)
**Last updated**: 2026-05-22 02:00 (Asia/Hong_Kong)
**Current focus** (one sentence): Three holistic-face paradigms done; clean dissociation visible across model classes. Time to consolidate: write a paper outline, generate publication-quality figures, and begin draft.
**Last action**: E024 part-whole PWI on 17 priors (N=100). All trained models PWI < 1 (whole-face context dampens eye-swap discriminability); DINOv2 family lowest at 0.31-0.40 confirming it as the "most spatially-holistic" model class. FaceNet still baseline. Generated `three_paradigm_panel.png` showing all 3 paradigms side-by-side per prior.
**Last action outcome**: Three-paradigm dissociation now formally documented. Pitch v6 written: CLIP wins Thatcher; DINOv2 wins composite + part-whole; FaceNet wins nothing. Idea-001 score held at 8.7.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 9): WRITE phase begins. Three sub-tasks:
  (a) Generate publication-quality combined figure (three-paradigm panel polished)
  (b) Sketch paper outline in research_journal/PAPER_OUTLINE.md
  (c) Identify the strongest 5 claims and their supporting experiments + write a Methods skeleton
This consolidates 8 ticks of results into a paper-shaped artifact.
**Confidence in current best idea (Idea-001)**: 8.7/10 (publishable; refinement remaining is sharpness + writing)

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (H100 80GB, 2TB NVMe, 2TB RAM, 224 CPUs)
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- ATM emb: `/workspace/illusionbench-eeg/data/atm_emb_eeg/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
