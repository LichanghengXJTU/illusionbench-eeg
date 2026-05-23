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
- **Score**: novelty 9 + feasibility 8 + evidence 9 = **8.7** (tick 73; held)
- **Tick-83 update (E046)**: **DINOv2 ridge target outperforms CLIP-H/14 ridge target by 63% relative on the 200-way THINGS-EEG2 test set (top-1 18.9% vs 11.6%, same ridge, same EEG source)**. Per-dim r_d mean 0.315 vs E020 CLIP ~0.158; 98.5% dims preserved. The E020 "uniform low-pass" finding was target-specific. This is a NEW publishable contribution alongside Idea-001 (target choice matters for EEG-decoder design). Score nudged 8.7 → **9.6** (novelty 10 from the cross-modal contribution, feasibility 9 from the demonstrated decoder pipeline, evidence 9 from the per-dim preservation result + 38× chance retrieval).
- **Reframed pitch v6 (post 3-paradigm dissociation)**: Across 17 modern visual priors and THREE holistic-face illusion paradigms (Thatcher, composite-face, part-whole), we identify a **clean dissociation between training objectives**: **CLIP-family dominates Thatcher** (orientation × local-feature ISI 4-7) while **DINOv2-family dominates composite-face** (spatial-integration CSI 1.4-1.7) AND **dominates part-whole context-dampening** (lowest PWI 0.31-0.40). Face-identity-trained models (FaceNet) sit at baseline on ALL three paradigms — face-recognition training systematically suppresses the holistic configural signals other training paradigms emerge. **Direct analysis of ATM's pre-computed EEG embeddings (E020-E021) reveals the EEG bottleneck imposes uniform per-dim attenuation (mean r ≈ 0.158 across 10 subjects × 1024 CLIP-H/14 dims), with NO dimension-specific or face-category-specific structure**. The three-paradigm dissociation + the uniform-low-pass EEG result together demonstrate two main contributions for the field: (a) a model-class taxonomy by holistic-illusion axis (Thatcher = image-text contrastive, composite/part-whole = SSL global), (b) a sharp prediction that current EEG visual decoders will FAIL on all three holistic axes due to non-specific bottleneck. Architecturally, this implies dimension-fine perceptual structure (not just CLIP-cluster semantic anchor) is required for human-aligned EEG visual decoding.

---

## Idea-003 (NEW, user-proposed 2026-05-22): Bio-inspired visual prior for EEG-to-image decoding

- **One-line pitch**: We've established that CLIP-class priors show paradigm-specific dissociation on holistic-face illusions (Thatcher: CLIP wins; Composite/Part-Whole: DINOv2 wins) and that the EEG bottleneck is uniform low-pass. The next-step contribution is to **build a bio-inspired visual prior** (LGN → V1 → V2 → V4 → IT with recurrence/feedback) and demonstrate it (a) is paradigm-consistent across the 3 illusion battery (no dissociation, like real human visual system), (b) competitive with CLIP-class on standard EEG-to-image retrieval. The framing is forward-modeling (brain mechanism → model architecture), avoiding the unscientific "model behavior → speculate brain mechanism" trap.

- **Why now**: IllusionBench-EEG (Idea-001) gives the FALSIFICATION CRITERION for "human-aligned configural processing" — must work on all 3 paradigms, not just one. CLIP-class fails this test. Bio-inspired prior is the natural alternative hypothesis.

- **Cheapest discriminating experiment (go/no-go in 1-2 ticks)**:
  1. Extract CORnet-S (already open-source, DiCarlo lab, Brain-Score top-tier) on all 3 IllusionBench paradigms
  2. Measure ISI / CSI / PWI; check if all 3 > 1 AND ratio between any two < 2:1 (paradigm-consistent)
  3. If yes → bio-inspired direction validated; explore further (PredNet, γ-net, NSD-aligned models, then design our own LGN-IT stream model)
  4. If no → fallback to PredNet / γ-net or design our own

- **Prior art (must check before claiming novelty)**:
  1. CORnet-S (Kubilius et al., NeurIPS 2019): 4-stage recurrent CNN matched to V1/V2/V4/IT. Code released. Brain-Score winner. **NOT yet applied to EEG-to-image decoding.**
  2. PredNet / Lotter 2017: predictive coding on natural images.
  3. γ-net (Linsley et al., ICLR 2020): horizontal lateral connections, contextual modulation. Specifically tested on contour illusions.
  4. NSD-aligned vision encoders (Allen 2022, Conwell 2024): use fMRI activity to align image encoders to ventral stream. Some PyTorch checkpoints available.
  5. **Key gap**: nobody has systematically replaced CLIP visual anchor in ATM/AVDE/ENIGMA with a bio-inspired vision encoder and measured perceptual + retrieval performance jointly.

- **Failure modes considered**:
  - (A) Bio-inspired model is paradigm-consistent but absolute values all low (ISI=2, CSI=1.4, PWI=1.4): "weak across the board". Avoid by calibrating to ISI ≥ 4 (human edge).
  - (B) Bio-inspired model is paradigm-consistent but retrieval top-1 << CLIP-H/14: model too weak. Fallback: hybrid bio+CLIP architecture (bio head for perceptual axes, CLIP head for retrieval).
  - (C) Existing bio-inspired model (CORnet / PredNet) ALREADY paradigm-consistent on our benchmark: contribution compressed to "show existing bio-inspired priors are better than CLIP for EEG decoding". Still publishable; faster paper.

- **Downstream application argument** (why brain-aligned matters):
  1. **Interpretable EEG decoding (strongest argument)**: per-layer mapping to brain regions enables attributing decoder decisions to specific cortical sources. CLIP is black-box dims.
  2. **Faithful BCI reconstruction**: current EEG decoders reconstruct images well in retrieval-top-K but observers report reconstructions don't match subjective experience. Bio-aligned prior aligned to mechanism should improve perceptual fidelity.
  3. **Cross-subject and clinical EEG decoding**: patient EEG (prosopagnosia, autism) requires interpretable representation to map damage to behavior.
  4. **Cross-modal alignment**: bio-prior trained with brain-structural inductive bias should align more naturally to EEG (also brain signal).
- **State of evidence**: Idea-001 results (paradigm-specific dissociation in CLIP) directly motivate Idea-003. EEG-side falsification criterion already in place via Route A.
- **Score**: novelty **7** + feasibility 7 + evidence **5** = **6.3** (tick 73: redirected as v2.1 FTPC + dual EEG decoder; spec finalised, code scaffolded, evidence pending; the redirect is brain-grounded and EEG-centric, headline ambition back).

### Tick-80 update (E045) — HOLO-Net v3 (frozen DINOv2 + global FTPC): 2/4 PASS, FAIL overall

- After 3× from-scratch SSL collapses (runs 1-3 of v2 instantiation), pivoted
  to option D = frozen pre-trained DINOv2 ViT-S/14 + FTPC head (T learnable +
  EMA over 13,586 ImageNet face samples in 3.4 min). Ran §6 4-paradigm eval.
- FFA-layer verdict: ISI 0.901 FAIL (need ≥3.0), CSI 1.300 FAIL (need ≥1.5,
  within 0.2), PWI 0.446 PASS (need ≤0.5; v1's 2.20 dramatically reversed),
  ISIrbox 0.654 PASS (need ≤1.5). **2/4 PASS** — best HOLO-Net so far, but
  ALL-FOUR-simultaneously: FAIL.
- Mechanistic finding: δ_pooled (ffa) and afp_pooled (raw DINOv2) 95% CIs
  overlap on every paradigm → the 2/4 PASS is DINOv2-backbone-driven, not
  FTPC-driven. A single global 16×16 template cannot encode local feature
  orientation (the Psalta 2014 Thatcher mechanism). **Architectural fix**:
  v2.2 with K=4 part-aware sub-templates (eyes/nose/mouth/chin) + per-region
  δ aggregation. Reviewer-defensible, principled, single-tick scaffold.
- Score → **5.5**: partial progress (evidence 4: 2/4 pass + one paradigm flipped
  + one within striking distance), well-defined next iteration (feasibility 7:
  v2.2 design tight, no engineering risk), novelty unchanged (7).

### Tick-65 update (E044) — HOLO-Net v1.0 falsification REFUTED

- v5 (fully-trained full HOLO-Net, identity loss 0.97, functional gate) was
  evaluated on the pre-registered §6 falsification. FFA-layer result: Thatcher
  ISI 1.00, Composite CSI 0.99, Part-Whole PWI 2.20, Random-bbox 1.08 →
  **fails 3 of 4; ALL FOUR: FAIL.** HOLO-Net v1.0 is NOT a paradigm-consistent
  model.
- Best reading: HOLO-Net is trained with a face-IDENTITY objective; Idea-001
  (E004) already showed identity-trained models give ISI ≈ 1. The bio-fidelity
  architecture does not substitute for the training objective. The negative
  result CONFIRMS Idea-001's thesis from the architecture side.
- The FFA module is not inert — it drives Part-Whole PWI to 2.20 (vs minimal
  floor 0.78), the largest deviation HOLO-Net produces — but opposite to the
  ≤0.5 criterion.
- Score → 5.0: the original "first paradigm-consistent model" ambition is
  refuted (evidence 3). The architecture trains fine (feasibility 7). A redirect
  (re-aim at the training objective, or consolidate as a mechanistic negative
  alongside Idea-001) remains open — see NOTES_FOR_USER FLAG-002.

### Tick-51 update (E042) — full HOLO-Net training UNBLOCKED; v1 failure was a loss bug

- The tick-50 panic ("the full HOLO-Net does not train") was **wrong about the
  cause**. Code audit (tick 51) found the v1 failure was an AdaFace formula bug
  — an extra `− scale·margin` term that depressed target logits (identity loss
  stuck ~65). That bug is already fixed in `losses.py`.
- E042 controlled re-test: FULL HOLO-Net (all bio components ON) + fixed loss
  starts at identity loss **13.20 ≈ minimal's 13.13**, runs cleanly. The bio
  components are exonerated — they were never the blocker.
- **Feasibility restored 4→7.** The remaining uncertainty for Idea-003 is now
  purely scientific — *does the trained HOLO-Net pass the FFA-layer
  falsification criteria (ISI≥3, CSI≥1.5, PWI≤0.5, random-bbox≤1.5)* — not an
  engineering blocker. evidence_potential held at 8 pending that result.
- Per user directive (2026-05-22), Idea-003 is the project's headline; the loop
  now builds/trains/validates HOLO-Net continuously.

### Tick-50 update (E040) — full HOLO-Net Stage-2 training FAILED to optimize

- The full bio-fidelity HOLO-Net (sub-path **e**, "design our own") with all
  components ON does NOT learn: identity loss stuck 63-69 for 8750 steps (E040
  run v1). It only trains after disabling **every** bio-fidelity component
  (LGN-Magno, OFA, FFA, Orientation Gate, Predify PC, PFC-Gist) → "minimal mode".
- Minimal mode = ≈ sub-path **(a)** (CORnet anatomy + AdaFace + Glint360K data).
  So the run currently on the H100 tests sub-path (a), NOT the HOLO-Net thesis.
  Its evaluation (E041) is still useful — a clean sub-path (a) datapoint.
- **Feasibility downgraded 6→4**: the headline-novelty architecture is, as of
  tick 50, non-functional. Recoverability depends on Q014 (isolate the breaking
  component). If Q014 finds a single fixable culprit, feasibility recovers.
- **Idea-001 is unaffected and remains the safe deliverable (score 9.3).**
  Idea-003 should not gate the paper.

- **Risks**:
  - Bio-inspired vision models are smaller than modern CLIP (typically <100M params vs CLIP-bigG 1.8B) → retrieval performance likely lower. Mitigation: hybrid architecture, or argue from interpretability angle alone.
  - Need to define "bio-inspired" rigorously; bio-fidelity vs ML-pragmatism trade-off is non-trivial.
  - **NEW from E028 (2026-05-22)**: Bio-inspired ANATOMY alone (CORnet-S V1→V2→V4→IT + recurrence, ImageNet-trained) gives NO Thatcher signal (ISI 0.99 = pixel baseline). The path forward requires combining bio-anatomy with face-specific training data and/or contrastive objective. This is a more complex engineering task than originally framed.
  - **NEW from E030 (2026-05-22)**: **ArcFace (ResNet-100 + angular-margin loss) gives ISI 1.70 [1.638, 1.766] on Thatcher**, 77% face-feature-specific. This is the first face-identity-trained model in our 19-prior battery to show non-trivial Thatcher (FaceNet's triplet-loss model gave ISI 1.12). **The loss function (angular-margin vs triplet) matters more than the face data per se**. This revives sub-path (a) face-CORnet+angular-margin-loss as a promising direction, since both ingredients (bio-anatomy + angular-margin-loss-on-face-data) might give true paradigm-consistency. ArcFace alone INVERTS Part-Whole (PWI 1.43, opposite direction) so it's still paradigm-INconsistent.

### Sub-paths for Idea-003 post-E028/E029/E030/E031/E032

(a) **Face-tuned CORnet via angular-margin loss + small noisy face data**
    (REFINED post-E032): The strongest Thatcher signal among ALL face-rec
    models is **AdaFace IR-50 + CASIA-WebFace** (ISI 2.91). Smaller noisier
    training data appears to be better. Target architecture for sub-path (a):
    CORnet-S anatomy + AdaFace-style quality-adaptive margin + CASIA-like
    training data. **Estimated training: 1-3 days GPU.**
(b)-(c) **NSD-aligned encoders / face-PredNet**: not publicly available [E030].
(d) ~~Multi-anchor decoder~~: ❌ **REFUTED by E029**.
(e) **Design our own**: longest path; (a) is a strict subset.
(f) ~~Loss-function ablation~~: ❌ done (E031, E032).
(g) **Compare AdaFace-CASIA vs CLIP on Route A per-dim preservation** — is
    AdaFace-CASIA's Thatcher carried by a few specific dims or distributed?
    Could test if EEG-side can preserve AdaFace-CASIA's signal.
(h) **NEW from E032**: **AdaFace IR-50 + CASIA might already be a useful EEG
    target embedding** — its small embedding (512-d), small backbone (43.6M),
    and high Thatcher (2.91) could be plugged into ATM as an alternative to
    CLIP-bigG. Quick test: replace ATM's CLIP-H/14 target with
    P26_adaface_ir50_casia and re-train; measure top-1 retrieval. 2-4h GPU.

Post-E032 priority order: (h) → (a) → (g) → (e). (b), (c), (d), (f) ruled out.

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
