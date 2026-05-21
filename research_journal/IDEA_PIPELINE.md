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
- **Score**: novelty 8 + feasibility 8 + evidence 9 = **8.3** (up from 8.0; we now have direct EEG evidence not just inference)
- **Reframed pitch v4**: Across 17 modern visual priors representing 6 training paradigms, image-text contrastive training (CLIP/SigLIP/MetaCLIP) exhibits a strong face-Thatcher ISI (4-7) absent from face-identity training (FaceNet, ISI≈1.0) and pixel-statistic baselines (≈1). All current SOTA EEG-to-image decoders (ATM, AVDE, ENIGMA, HVF) anchor to this image-text-contrastive family. **Direct analysis of ATM's pre-computed EEG embeddings (E020) reveals the EEG bottleneck imposes a uniform per-dim attenuation (mean Pearson r ≈ 0.158 across 10 THINGS-EEG2 subjects, 1024 CLIP dims), with NO dimension-specific structure: Thatcher-loaded dimensions are neither selectively preserved nor destroyed.** This implies the visual-prior-driven Thatcher signal is washed out non-specifically by EEG decoding — current EEG visual decoders recover cluster-level semantic information but not fine perceptual asymmetries. The result reframes "human-aligned EEG visual decoding" not as "matching CLIP's Thatcher" (the prior already has it) but as "designing EEG architectures that preserve dimension-fine perceptual structure".

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
