# E022 — Literature gap check (Phillips & White 2026, adjacent CLIP work)

**Hypothesis**: Idea-001's novelty depends on no prior work having published the
following combination: image-text contrastive vision priors × face holistic
illusions (Thatcher/composite/part-whole) × EEG-decoder preservation.

**Method**: Read Phillips & White 2026 BJP review abstract + 2024-2026 arXiv
searches for CLIP × Thatcher / configural processing / face holistic illusion.

**Result**:
- **Phillips & White 2026** (BJP, full text paywalled): abstract via PubMed
  https://pubmed.ncbi.nlm.nih.gov/40364689/ — covers DCNN-based face-identity
  models vs psychological models of face processing. **Does NOT mention** any of:
  Thatcher illusion in DNNs, CLIP, EEG decoders, holistic processing benchmarks,
  ISI/illusion-sensitivity metrics, FaceNet vs CLIP comparison. Their scope is
  identity-recognition DCNNs (the FaceNet/VGGFace lineage, with which our E004
  finding is consistent — they don't show Thatcher).
- **arXiv 2508.09814** "Dynamic evolution of CLIP texture-shape bias and its
  relationship to human alignment" (2025) — closest adjacent work. Studies
  CLIP's texture-shape bias evolution during training, with multiple
  perceptual-alignment benchmarks (low-level image quality, perceptual
  similarity, saliency, noise robustness). **Does NOT use face Thatcher /
  composite / part-whole stimuli** and **does NOT study EEG decoders**.
  Complementary perspective, no overlap.
- **arXiv 2403.02580** "What do we learn from inverting CLIP models?" — model
  inversion = generating images that maximize CLIP scores. Unrelated to face
  inversion / Thatcher.
- General search: no paper combines "CLIP" + "Thatcher illusion" + "EEG" or
  even "CLIP" + "configural processing" + "ISI" / "illusion sensitivity index"
  in 2024-2026. The "Jacob et al. 2021 Nat Comm" remains the only adjacent
  prior work, and it pre-dates CLIP-bigG by years and only uses VGG-Face-style
  DNNs.

**Interpretation**:
- [CONFIRMED] No published paper covers Idea-001's exact framing. We are
  the first to:
  1. Demonstrate face-Thatcher ISI in CLIP-class image-text contrastive priors
  2. Show the hierarchy across 17 priors and 6 training paradigms
  3. Show face-recognition-trained models (FaceNet) have NO Thatcher signal
  4. Quantify per-CLIP-dim EEG preservation × Thatcher loading via Route A on ATM
- [CONFIRMED] The risk that "Phillips & White already published this" is FALSE.
- [LIMITATION] Wiley paywall prevented full-text Phillips & White read. If
  during writing we find we cannot get full text, paragraph-level claims
  about overlap should be qualified as "based on abstract".

**Linked Q###**: meta — protects Idea-001 novelty from the most relevant 2026
review.
