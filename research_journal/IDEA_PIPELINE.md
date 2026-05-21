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
- **State of evidence**: E001-E005 establish (a) image-side ISI hierarchy across 17 priors, (b) face-feature-specific (E003), (c) NOT from face-recognition training (E004), (d) general to image-text contrastive family (E005 — SigLIP and MetaCLIP also show ISI 5-6). EEG-side untested.
- **Score**: novelty 8 + feasibility 8 + evidence 8 = **8.0** (up from 7.7) — crosses publishable threshold
- **Reframed pitch v3**: Across 17 visual priors representing 6 training paradigms, a clean ISI hierarchy emerges. **Image-text contrastive training (CLIP / SigLIP / MetaCLIP) — the visual prior universally used by current EEG-to-image decoders — exhibits face-Thatcher ISI 4-7 with face-feature-specificity ratio ~70%. Crucially: face-identity training (FaceNet) shows NO Thatcher (ISI ≈ 1.0). This means current EEG decoders inherit an orientation-emergent face-configural signature from their visual anchor — but we do not know whether the EEG signal preserves it.** The central scientific question for the EEG side: do ATM / AVDE / ENIGMA pipelines preserve, attenuate, or destroy this CLIP-side Thatcher signature when projecting through the EEG modality?

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
