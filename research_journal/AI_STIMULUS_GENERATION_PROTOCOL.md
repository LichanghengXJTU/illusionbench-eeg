# AI image-generation protocol for IllusionBench-EEG stimuli

**Purpose**: Specify rigorous and ethical procedures for generating Thatcher,
composite, and part-whole face stimuli using GPT-image-2 (or equivalent
multimodal generators), such that the resulting benchmark is publishable at
NeurIPS D&B / TMLR without methodology objections from reviewers.

**Author**: drafted 2026-05-24 in response to user direction "你需要设计对应
的方案规避不严谨性和伦理问题".

**Status**: design spec — execution gated on user confirmation + API key
re-supply.

---

## Why AI-generated stimuli at all?

Real-photo face benchmarks (FFHQ, LFW, CFD) bring chronic problems for
illusion paradigms:

- **Identity-pair selection bias**: matching gender / age / pose across
  100 pairs requires hand-curation that is impossible at scale.
- **Skin-tone / luminance mismatch** in composite stimuli is a known
  shortcut feature (Murphy & Cook 2017).
- **Bbox-shaped paste seams** in Thatcher stimuli are visible artifacts
  (confirmed in our QC; also present in Jacob 2021 Nat Comm OSF release).
- **Consent issue**: hand-cut feature transforms of real identifiable
  people may violate model release.
- **Limited paradigm coverage**: no public dataset spans all 3 paradigms.

AI generation can in principle solve all five. But it introduces new
problems that this protocol explicitly addresses.

---

## P1. Generator transparency (reproducibility floor)

For every stimulus released:

- Record `generator_model: gpt-image-2`, `model_version_string` (the
  exact version reported by API at call time, NOT just "gpt-image-2"),
  `api_call_timestamp_utc`, `prompt_text` (full string), `seed` if the
  API supports it, `negative_prompt` (if any), `revision_chain` (list
  of edit prompts if iterative), and `output_image_sha256`.
- Store this metadata in `data/stimuli_gpt/manifest.csv` per row.
- **Generated images themselves are version-locked artifacts**: even
  if OpenAI deprecates the model, the image bytes stay reproducible
  via the open dataset release. Re-generation is NOT replication —
  the released image IS the canonical artifact.
- Pin the OpenAI SDK version in `requirements.txt` and capture in
  generation logs.

## P2. Human validation panel (no skipping)

Before any stimulus enters the benchmark, run a **small online
human-validation study** (e.g. n=30-50 MTurk / Prolific, ~1h each):

- **Realism check**: 4-AFC "which of these faces is real?" — mix
  generated upright neutral faces with FFHQ real faces. If raters
  identify > 80% of AI faces as fake, the generation quality is too
  low; iterate prompts.
- **Paradigm validation**:
  - **Thatcher**: 2AFC "which face looks grotesque?" on (V1, V2)
    upright pairs. Humans should pick V2 (thatched upright) at > 90%.
    On (V3, V4) inverted pairs, humans should be at chance (~50%).
  - **Composite**: same/different judgment on top half, aligned vs
    misaligned conditions. Standard composite-effect interference
    (~10-15% accuracy drop for aligned-different vs misaligned-different)
    should replicate.
  - **Part-Whole**: study-then-test paradigm matched to Tanaka 1997.
    Accuracy in whole context should exceed isolated parts by
    ~5-15%.
- **Pass criterion**: if any paradigm fails to elicit human effect
  in the predicted direction, that paradigm's stimuli are revised
  (different prompts / different transforms) until they do.
- This **anchors §6 thresholds to human data on OUR stimuli**, not
  to legacy literature numbers from different stimulus sets. Solves
  the "back-derived from CLIP" critique.
- Budget: ~$300-500 on Prolific for n=50 × 3 paradigms × 30 min each.
  This is the single most important rigor investment.

## P3. Demographic balance via prompt-grid (not free-text)

- Construct an explicit factorial: `{gender: M/F/NB} × {age:
  young-adult/middle-age/older} × {ethnicity: 6 categories from
  CFD: Black/White/Asian/Latinx/Middle-Eastern/Indigenous} ×
  {neutral expression: yes-only}`.
- N per cell = ceil(target_N / cell_count). For 200 identities,
  3 × 3 × 6 = 54 cells → ~4 identities per cell.
- Document the realized distribution and report any cell with < 2
  successful generations as a known limitation.
- This addresses the well-documented Western-Caucasian bias of
  text-to-image generators (Bianchi et al. 2023, Cho et al. 2023).
- **Do NOT** use free-text prompts like "a generic face" or
  "a typical person" — they will produce biased outputs.

## P4. Generator artifact QC pipeline

Each generated face must pass:

1. **MediaPipe FaceMesh** detection with confidence ≥ 0.9 (stricter
   than our 0.7 for FFHQ — AI faces are typically less anatomically
   regular).
2. **IOD ≥ 200 px** at 1024² (stricter than FFHQ's 120 px).
3. **Eye-line tilt ≤ 10°** (stricter than FFHQ's 15°).
4. **Symmetry score** ≥ threshold (left/right landmark distance to
   midline within 5%).
5. **No duplicate face detection** (single-face only).
6. **Visual sanity automated check**: count fingers / teeth /
   eyes via a separate detector — reject if abnormal.
7. **Real-vs-fake classifier check** (P5 below): the AI face must
   be classified "real" with confidence ≥ a threshold by a face
   anti-spoofing model (e.g. CelebA-Spoof-trained ResNet).
8. **Manual review of first 50** before generating the remaining
   batch. Spot-check 5% of the final set.

Failure handling: regenerate with same prompt + new seed up to 3
tries; if still failing, skip that cell and document.

## P5. Real-vs-AI similarity bridge

Before declaring the AI stimuli are usable, demonstrate that **on
the dimensions we care about**, they behave like real faces:

- Train a small ResNet-50 binary classifier on FFHQ-real (N=2000)
  vs our generated set (N=2000) with held-out cross-validation.
  If test accuracy is much > 80%, the AI faces are clearly
  distinguishable from real — DNN-based EEG decoders may pick up
  this shortcut. Iterate prompts.
- Compute per-prior cosine-distance distributions on AI faces vs
  FFHQ faces. If a single CLIP variant gives qualitatively
  different ISI hierarchy on the two sources, that's evidence
  the AI stimuli are not transferable.
- **Pass criterion**: across our 25-prior battery, the rank
  correlation (Spearman) between AI-stimuli ISI and FFHQ-stimuli
  ISI should be > 0.85.

## P6. Hybrid validation (the safety net)

**Run all final metrics on BOTH** AI-generated and FFHQ
(contour-mask-fixed) stimulus sets. Report:

- **Two §6 verdict tables** — one per stimulus source.
- **A "Stimulus-source consistency" supplementary section** that
  shows the dissociation pattern (CLIP vs DINOv2 vs FaceNet on
  3 paradigms) is preserved across sources.
- If they agree: the benchmark is robust; report AI as the
  primary stimulus (better controlled) with FFHQ as a robustness
  check. If they disagree: AI generator artifacts are driving
  signal; demote AI to supplementary or scrap.

## P7. Replicability + open dataset release

- Release on Hugging Face Datasets + Zenodo (for DOI):
  - All generated images (PNG, 1024²)
  - Full per-image manifest (P1) including prompts + seeds
  - Generation code (Python module that calls the API)
  - Human-validation behavioral data + analysis notebooks
  - License: CC-BY 4.0 for images + behavioral data;
    MIT for code
- Pin to a specific GPT-image-2 model snapshot via API
  parameter; document model deprecation policy. The benchmark
  remains usable indefinitely via the released image bytes
  even if the generator goes away.

## P8. Ethics audit checklist

- **Consent**: synthetic faces don't correspond to real
  individuals → no consent issue. **But** explicitly prompt the
  generator NOT to produce likenesses of real people: prepend
  every prompt with "Generate a fully synthetic non-identifiable
  face. Do not reproduce the likeness of any real person, public
  figure, or celebrity."
- **Likeness check**: face-recognition cross-search the generated
  faces against the LFW + VGGFace2 + CelebA databases via a face
  embedding model (FaceNet); reject any AI face with cosine
  similarity > 0.7 to any real-person embedding.
- **Bias audit**: report demographic distribution; identify any
  cell where the generator failed (e.g. older Indigenous women);
  acknowledge in paper.
- **Misuse risk**: face illusions are low-risk content; release
  is appropriate. Include a model card noting that the images
  are NOT to be used for face-recognition training (which would
  inherit the generator's biases).
- **IRB**: if a human-validation panel (P2) is run, file an IRB
  protocol at HKUST for the behavioral data collection. Likely
  expedited review (~2-4 weeks).

## P9. Limitation acknowledgment in paper

The Limitations section must include:

- "Stimuli are AI-generated (GPT-image-2, snapshot $X$,
  $Y$ identities); generalization to in-the-wild real-photo
  face stimuli is supported by the FFHQ robustness check (§S$Z$)
  but not guaranteed in all conditions."
- "The generator's training data composition is not publicly
  disclosed; demographic representation may be skewed in ways
  our prompt-grid only partially controls."
- "Human-validation panel n=$X$ is small; per-stimulus human
  ratings have wide CIs."
- "AI-generated face benchmarks are a young methodology; we
  recommend follow-up replication using laboratory-collected
  real face stimuli (e.g., CFD pairs hand-curated for matched
  demographics) once an EEG dataset for those paradigms exists."

---

## Execution roadmap (proposed)

| Phase | Deliverable | Cost | Time |
|---|---|---|---|
| Phase 0 | Prompt template design + 50-face pilot | API $5 | 1 day |
| Phase 1 | Pilot batch (P1 + P3 + P4) | API $50 | 2 days |
| Phase 2 | Human validation panel (P2) | Prolific $400 | 1 week (IRB + collection) |
| Phase 3 | Full generation (200 IDs × 4 conds × 3 paradigms) | API $200-400 | 1 week |
| Phase 4 | Real-vs-AI bridge (P5) + hybrid validation (P6) | $0 GPU | 3 days |
| Phase 5 | Release artifacts (P7) + paper limitations write-up | $0 | 2 days |

Total: ~3 weeks, ~$700-900, **including IRB time**.

---

## Hard contingencies (when to abort)

- P2 human validation fails: the human Thatcher/composite/PW
  effect does NOT replicate on our AI stimuli. → Stop. Iterate
  prompts. If 3 rounds of prompt iteration still fail, conclude
  GPT-image-2 cannot produce face stimuli that elicit the
  configural illusion in humans, and **fall back to FFHQ +
  contour-mask refactor + acknowledge in paper that the AI
  pathway was attempted and found insufficient**.
- P5 real-vs-AI bridge fails (classifier > 90% accuracy): AI
  faces too obviously fake. → Stop. Iterate.
- P8 likeness check fails (any AI face matches a real LFW
  identity): drop those AI faces; regenerate with stricter
  "no real likeness" prompt.

---

## Cross-references

- `stimuli/generate_thatcher_ffhq.py`, `stimuli/generate_composite_ffhq.py`,
  `stimuli/generate_partwhole_ffhq.py` — existing FFHQ generators to be
  ported / adapted for AI source images.
- `analysis/section6_verdict.py` — produces the §6 table for both stimulus
  sources (P6).
- `research_journal/NOTES_FOR_USER.md` MILESTONE-007 — context for why
  this protocol exists.
- Suggested supplementary statistical reference: Bianchi, F. et al.
  (2023). "Easily accessible text-to-image generation amplifies
  demographic stereotypes at large scale." FAccT 2023.

---

## Open questions for user (to resolve before Phase 0)

1. **API key**: previous agent did not commit yours; please re-supply
   securely (e.g. via env-var instructions in a paste-to-shell snippet
   rather than committed to repo).
2. **IRB**: do we file at HKUST or skip the human-validation panel
   in favor of just publishing the dataset without per-stimulus human
   anchoring? Skipping P2 saves ~$400 and a week but weakens the §6
   threshold defense significantly.
3. **Scope**: 200 identities at 4 conditions × 3 paradigms = 2400 images
   ≈ $200-400 at GPT-image-2 list price. If you'd like fewer (e.g. 100
   identities = $100-200) or more (e.g. 500 identities = $500-1000),
   say so.
4. **Hybrid vs AI-only**: do we keep the FFHQ (contour-mask-fixed)
   pathway running in parallel for robustness, or commit AI-only?
   Recommendation: hybrid (the FFHQ refactor is cheap).
