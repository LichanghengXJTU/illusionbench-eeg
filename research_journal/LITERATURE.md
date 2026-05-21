# Literature

Append-only chronological notes. Each entry: title, URL, read-depth, what it claims,
how it relates to our project, threat-or-opportunity.

---

## The seven seeded papers (user-provided, 2026-05-21)

### LaBraM (Jiang et al., ICLR 2024 spotlight) — arxiv 2405.18765
- URL: https://github.com/935963004/LaBraM
- Read: structured skim (subagent report)
- Claims: BERT-style foundation model for raw multi-channel EEG, pretrained on ~2,500 h via
  masked patch prediction over learned neural codebook. No visual prior at pretraining.
- Relates: candidate EEG encoder for E020+ (Q005). Used as backbone by AVDE.
- Status: code + checkpoint available.

### CBraMod (Wang et al., ICLR 2025) — arxiv 2412.07236
- URL: https://github.com/wjq-learning/CBraMod
- Read: structured skim
- Claims: criss-cross EEG transformer, pretrained on TUEG, SOTA across 12 BCI datasets.
  Again, no visual prior.
- Relates: alternative EEG encoder for the Q005 factorial.
- Status: code released.

### CSBrain (Zhou et al., preprint 2025) — arxiv 2506.23075
- URL: https://arxiv.org/abs/2506.23075
- Read: abstract + intro
- Claims: cross-scale spatiotemporal brain FM; multi-scale CST + Structured Sparse Attention.
  "Code and model will be released" — no URL yet.
- Relates: latest EEG FM in 2025-2026; potential Q005 backbone.
- Status: code NOT yet released → blocked unless we contact authors.

### ATM (Li et al., 2024) — arxiv 2403.07721
- URL: https://github.com/dongyangli-del/EEG_Image_decode
- Read: structured skim
- Claims: EEG-to-image with CLIP-ViT image embedding alignment + 2-stage diffusion.
- Relates: primary EEG-to-image baseline for Q005. CLIP is the load-bearing visual prior.
- Status: code + HuggingFace preprocessed data available.

### AVDE (ICLR 2026) — arxiv 2602.22555
- URL: https://github.com/ddicee/avde
- Read: structured skim
- Claims: replaces diffusion with autoregressive VAR-style next-scale prediction over VQ tokens;
  uses LaBraM encoder + CLIP target.
- Relates: cleanest factorial split (LaBraM encoder × CLIP/VQ-VAE priors).
- Status: code released.

### HVF / "Hierarchical Visual Embeddings" (Zheng et al., ICLR 2026) — arxiv 2602.07495
- URL: https://openreview.net/forum?id=IEq71qS8B7
- Read: structured skim
- Claims: brain embedding aligned to fused multi-encoder visual representation (CLIP RN50 +
  CLIP ViT-B/32 + SDXL VAE); Fusion Prior trained on ImageNet.
- Relates: most-relevant EEG-to-image baseline; author Jiawen Zheng is at HKUST-GZ (same uni).
- Status: code NOT in paper; HKUST proximity could unblock via email.

### ViEEG (Liu et al., May 2025 v3) — arxiv 2505.12408
- URL: https://arxiv.org/abs/2505.12408
- Read: structured skim
- Claims: three-stream EEG encoder aligned to CLIP-ViT-H/14 of three image decompositions
  (contour, foreground object, raw scene); 40.9% top-1 retrieval on THINGS-EEG.
- Relates: explicit "biological hierarchy" claim — most attackable Q005 target.
- Status: no GitHub URL in paper.

---

## Foundational psychophysics for the project

### Thompson 1980 — Margaret Thatcher: a new illusion
- URL: https://anstislab.ucsd.edu/files/2012/11/2009-Thatcher-1980-paper.pdf
- Read: abstract via web search
- Claims: original Thatcher illusion description — invert eyes + mouth in face; upright
  grotesque, inverted appears normal.
- Relates: defines the canonical stimulus paradigm. Our generate_thatcher.py implements
  this with MediaPipe-driven bbox detection.

### Carbon et al. 2005 — Thatcher faces ERP study
- URL: https://www.experimental-psychology.de/ccc/docs/pubs/CarbonSchweinbergerKaufmannLeder2005.pdf
- Read: abstract
- Claims: human ISI ≈ 4-5 in 2AFC; ERP/N170 evidence that early visual cortex detects
  Thatcher in both orientations but late conscious report fails inverted.
- Relates: source of our "human reference" range. **CAUTION**: their ISI is from
  behavioral 2AFC d', not embedding distance — quantitative comparison with our
  ISI_pert is qualitative only.

### Jacob et al. 2021 (Nat Comm) — Qualitative similarities and differences in visual object representations between brains and deep networks
- URL: https://www.nature.com/articles/s41467-021-22078-3
- Read: abstract + search results
- Claims: many human visual phenomena (Thatcher among them) reproduce in deep nets trained
  on faces; many others (global advantage, parts-whole) do not.
- Relates: **prior art most threatening to novelty**. We must demonstrate clear delta from
  this work. Their work was image-only and pre-CLIP-bigG. Our angle: (a) systematic across
  modern priors (CLIP-bigG, DINOv2-giant, MAE), (b) EEG-decoder image-side, (c) ISI
  metric (their metric was identification-confidence drop, different).

### Phillips & White 2026 (BJP) — The state of modelling face processing in humans with deep learning
- URL: https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/bjop.12794
- Read: abstract only — full read still needed
- Claims: reviews how DNN face models compare to psychological models. DNNs surpass humans
  in identification; qualitative aspects partially modeled.
- Relates: **must read in full** to know what's already covered.
- Status: action item for loop tick 2 or 3.

### Fel et al. 2022 (NeurIPS) — Harmonizing the object recognition strategies of deep neural networks with humans
- URL: https://arxiv.org/abs/2211.04533
- Read: search summary
- Claims: harmonization training routine aligns DNN saliency with ClickMe human attention.
- Relates: source of our planned Q004 perception-aligned baseline.

---

## Datasets / benchmarks

### THINGS-EEG2 (Gifford et al. 2022 NeuroImage)
- 10 subjects, 64-ch EEG, 16,740 image conditions. Standard EEG-to-image benchmark.
- HF: `gasparyanartur/things-eeg2`. OSF: project 3jk45.

### AllJoined-1.6M (arxiv 2508.18571)
- 20 subjects, 32-ch consumer-grade EEG, 1.6M trials.
- HF: `Alljoined/Alljoined-1.6M` (136 GB).

### FFHQ-1024 (Karras et al. 2019)
- 70k 1024×1024 face PNGs. We use as canonical-quality stimulus source.
- HF tar shard: `gaunernst/ffhq-1024-wds/00000.tar`.

### ClickMe (Linsley et al. NeurIPS 2017)
- Human click maps for ImageNet images. Used by Harmonization training.
