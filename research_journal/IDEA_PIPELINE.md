# Idea pipeline

Each idea ranked by **score = (novelty × feasibility × evidence) / 30**, on 0-10 each.
Ideas must have ≥ 2 prior-art citations before being added.

---

## Idea-001 (current strongest): EEG-decoding "Thatcher-signal preservation"

- **One-line pitch**: Current SOTA EEG-to-image decoders inherit a face-feature-specific configural Thatcher signature from their CLIP visual priors (ISI 4-7, ~70% face-specific per E003); we measure whether the EEG signal preserves this signature through encoder compression.
- **Why now**: We have evidence (E002 + E003) that the CLIP visual prior — universal across ATM / AVDE / ENIGMA / HVF — shows ISI 4-7 on FFHQ Thatcher, and that this is ~70-80% face-feature-specific (E003 random-bbox control drops ISI by 60-74%). The downstream question is whether EEG signal under current decoders carries this face-configural information.
- **Cheapest discriminating experiment**: Once face-trained / Harmonized baselines are added (Q003/Q004), run E020: extract EEG-conditioned image embeddings via ATM/AVDE/ENIGMA pipelines on THINGS-EEG2 test stimuli; substitute the Thatcher battery; measure ISI of EEG-decoded representations.
- **Prior art**:
  1. Jacob et al. 2021 Nat Comm: face-DNN reproduces Thatcher behaviorally. https://www.nature.com/articles/s41467-021-22078-3
  2. Li et al. 2024 (ATM): EEG-to-image via CLIP guidance. arxiv 2403.07721
  3. Phillips & White 2026 (BJP review): face-DNN alignment without EEG. https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/bjop.12794
  - **Gap claim**: no prior work measures whether EEG-conditioned visual representations preserve face-configural illusion signatures.
- **Risks**:
  - ~~If Q001 reveals general orientation bias~~ — RESOLVED, predominantly face-specific (E003).
  - EEG signal may have insufficient SNR to carry configural information; result could be "EEG destroys it" — still publishable but a different narrative.
- **State of evidence**: E001-E005 establish image-side ISI hierarchy across 17 priors, face-feature-specific (E003), not from face-training (E004), family-general to image-text training (E005). **E020 (Route A) adds the first EEG-side data point**: ATM-decoded EEG embeddings have uniform per-dim preservation r ≈ 0.158 in CLIP-H/14 space; Thatcher-loaded dimensions are NEITHER specially preserved nor specially destroyed (Spearman ρ = -0.029, p = 0.35). Predicted EEG-side ISI on hypothetical Thatcher-EEG: ~1.5-2 (collapsed from CLIP-side 5.5).
- **Score**: novelty 9 + feasibility 8 + evidence 9 = **8.7** (held; E024 reinforces dissociation across 3 paradigms)
- **Reframed pitch v6 (post 3-paradigm dissociation)**: Across 17 modern visual priors and THREE holistic-face illusion paradigms (Thatcher, composite-face, part-whole), we identify a **clean dissociation between training objectives**: **CLIP-family dominates Thatcher** (orientation × local-feature ISI 4-7) while **DINOv2-family dominates composite-face** (spatial-integration CSI 1.4-1.7) AND **dominates part-whole context-dampening** (lowest PWI 0.31-0.40). Face-identity-trained models (FaceNet) sit at baseline on ALL three paradigms — face-recognition training systematically suppresses the holistic configural signals other training paradigms emerge. **Direct analysis of ATM's pre-computed EEG embeddings (E020-E021) reveals the EEG bottleneck imposes uniform per-dim attenuation (mean r ≈ 0.158 across 10 subjects × 1024 CLIP-H/14 dims), with NO dimension-specific or face-category-specific structure**. The three-paradigm dissociation + the uniform-low-pass EEG result together demonstrate two main contributions for the field: (a) a model-class taxonomy by holistic-illusion axis (Thatcher = image-text contrastive, composite/part-whole = SSL global), (b) a sharp prediction that current EEG visual decoders will FAIL on all three holistic axes due to non-specific bottleneck. Architecturally, this implies dimension-fine perceptual structure (not just CLIP-cluster semantic anchor) is required for human-aligned EEG visual decoding.

---

## Idea-002: Illusion-based EEG decoder benchmark suite

- **One-line pitch**: A reproducible benchmark (face Thatcher + composite + part-whole) for evaluating whether EEG decoders preserve image-side illusion sensitivity, released alongside metrics and 12+ baseline models.
- **Why now**: Same evidence as Idea-001; benchmark framing is broader and more D&B-track-friendly.
- **Cheapest discriminating experiment**: Same as Idea-001 plus expand stimulus battery to composite + part-whole.
- **Prior art**:
  1. Jacob et al. 2021 used 9 visual phenomena but no EEG.
  2. Phillips & White 2026 reviews many DNN-face alignment studies but no benchmark released for EEG decoders.
- **Risks**: D&B benchmarks need a strong "useful for the field" justification, which depends on Q005 result.
- **Score**: novelty 6 + feasibility 7 + evidence 5 = **6.0**

---

(More ideas will be added by the loop. Score gates: ≥ 6.5 to be considered actionable; ≥ 8 to claim "publishable.")
