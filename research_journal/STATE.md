# State

**Tick #**: 32 (E034 paradigm-specific RSA)
**Last updated**: 2026-05-22 (Asia/Hong_Kong)
**Current focus** (one sentence): E034 extended RSA to Composite + Part-Whole, revealing cluster structure CHANGES by paradigm — CORnet-S, AdaFace-CASIA, FaceNet all migrate between clusters across paradigms. Dissociation is a (model × paradigm) interaction, not a fixed model-identity property.
**Last action**: E034 — computed RSA per paradigm (Thatcher 800, Composite 400, Part-Whole 400 stimuli); hierarchical clustering at k=6; cross-paradigm Spearman agreement.
**Last action outcome**: Cross-paradigm RSA Spearman: Thatcher↔Composite 0.86, Thatcher↔PartWhole **0.54**, Composite↔PartWhole 0.73. Part-Whole is most distinct. CORnet-S migrates "alone→semantic→reconstructive"; AdaFace-CASIA migrates "angular-margin→triplet→semantic". On Part-Whole, FaceNet/AuraFace/AdaFace-CASIA all collapse into the big CLIP+DINOv2 semantic cluster.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 33): SYNTHESIS (per loop protocol, ≥3 experiments since last synthesis at tick 25). Integrate E028-E034 into PAPER_DRAFT: (a) update §3 with new findings, (b) generate combined 3-panel RSA figure (Thatcher + Composite + Part-Whole) for paper Figure 3, (c) update CLAIMS_SKELETON.md with new claims about (model × paradigm) interaction.
**Confidence in current best idea**: **9.0/10 for Idea-001** (paradigm-conditional migration is a strong new positive finding); 8.2/10 for Idea-003.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- CORnet repo: `/workspace/eeg_repos/CORnet/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
