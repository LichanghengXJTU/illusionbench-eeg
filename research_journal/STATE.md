# State

**Tick #**: 7 (E022 lit gap check + E023 composite-face CSI)
**Last updated**: 2026-05-22 01:35 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-001 score now **8.7** — composite-face paradigm reveals a SHARP dissociation: CLIP dominates Thatcher but DINOv2 dominates composite. The dissociation is a major novelty boost. Next: part-whole illusion as the third paradigm completes the holistic-face benchmark trio.
**Last action**: (a) Phillips & White 2026 BJP review confirmed orthogonal scope (DCNN face-identity vs psych models; no CLIP / Thatcher / EEG) ⇒ Idea-001 prior-art-safe. (b) Generated composite-face battery (100 FFHQ identities × 4 conditions = 400 stimuli) with shifted-bottom misalignment. CSI computed across 17 priors. SURPRISE: paradigm-specific dissociation — CLIP-family weak on composite (corrected CSI 1.10-1.20), DINOv2-family strong (1.41-1.53), MetaCLIP-H14 and CLIP-L14 outlier high (1.56, 1.60). Sanity check note: pixel-baseline CSI = 1.099 not 1.000 because composite paradigm's V1-V2 vs V3-V4 are not pixel-exact-matched; we report both raw and pixel-corrected CSI.
**Last action outcome**: SHARPER pitch v5 + spawned Q009 (CLIP-L14 cross-paradigm outlier).
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 8): Part-Whole illusion battery. For each identity, generate:
  PW V1: whole face with target feature (eyes OR mouth) intact
  PW V2: whole face with feature replaced from another identity
  PW V3: feature-only (target) on gray background
  PW V4: feature-only (different identity's feature) on gray
PWI_pert = d(V1, V2) / d(V3, V4). Human prediction: > 1 (whole context aids discrimination). Then extract all 17 priors. Look for further paradigm dissociation pattern.
**Confidence in current best idea (Idea-001)**: 8.7/10

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (H100 80GB, 2TB NVMe, 2TB RAM, 224 CPUs)
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- ATM emb: `/workspace/illusionbench-eeg/data/atm_emb_eeg/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
