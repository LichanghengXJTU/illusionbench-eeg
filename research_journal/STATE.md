# State

**Tick #**: 5 (E020 Route A complete)
**Last updated**: 2026-05-22 01:05 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-001 score now 8.3 — Route A produced first direct EEG-side evidence: ATM EEG bottleneck uniformly attenuates CLIP space to r≈0.158 per dim, with NO dimension-specific structure (Thatcher dims not selectively destroyed). Next: extend the analysis with AVDE / ENIGMA if their pretrained embeddings are available, and refine the prediction of EEG-side ISI by checking face-category-restricted preservation.
**Last action**: Downloaded ATM pre-computed EEG embeddings (10 subjects × 200 concepts × 1024 CLIP-H/14 dims) + CLIP target embeddings. Computed per-dim preservation correlation across subjects; computed Thatcher loading per dim from E002 P04 CLIP-H/14 FFHQ embeddings. Spearman ρ(preservation, loading) = -0.029, perm p = 0.35 — NULL relationship. Scatter figure generated.
**Last action outcome**: NULL on preservation × loading correlation; UNIFORM low preservation (mean r 0.158) ⇒ predicted EEG-side ISI ~1.5-2 (vs CLIP-side 5.5). Idea-001 8.0 → 8.3.
**Running tasks** (on server): none
**Stuck streak**: 0
**Planned next action** (tick 6): Extend Route A. Two options:
  (a) Restrict preservation analysis to FACE categories of THINGS-EEG2 only (~7% of concepts) — sees if face-category EEG carries more face-feature information per dim than the general analysis
  (b) Add AVDE / ENIGMA pre-computed embeddings if available on HuggingFace
  (c) Search literature for prior empirical estimates of EEG-CLIP preservation
Pick the cheapest-and-most-informative for tick 6.
**Confidence in current best idea (Idea-001)**: 8.3/10

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (H100 80GB, 2TB NVMe, 2TB RAM, 224 CPUs)
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- ATM emb: `/workspace/illusionbench-eeg/data/atm_emb_eeg/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
