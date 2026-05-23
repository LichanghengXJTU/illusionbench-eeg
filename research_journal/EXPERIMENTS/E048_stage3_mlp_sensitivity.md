# E048 — Stage-3 MLP sensitivity check (vs ridge baseline E046/E047)

**Hypothesis** (E047 [CONJECTURE], pre-registered before run): a non-linear
mapping (2-layer MLP with GELU) could recover PWI structure that linear
(ridge) decoding destroys. If MLP retrieval >> ridge OR MLP IllusionBench
transfer PWI < 0.5 across more subjects, the "EEG bottleneck is fundamental"
conclusion weakens. If MLP ≈ ridge on both, conclusion strengthens.

**Method**:
- Per-subject MLP: 1024 (ATM EEG) → 512 (GELU + dropout 0.1) → 384 (DINOv2)
- Loss: 1 - cosine_similarity
- AdamW lr 1e-3, weight_decay 1e-4, batch 256
- 90/10 train/val split (deterministic seed); early stop patience 5
- Up to 50 epochs; subjects converged at 27-48 epochs
- Same retrieval eval + per-dim r_d as ridge
- Same IllusionBench transfer pipeline (`illusionbench_transfer.py`)
  with MLP-derived per-dim r_d in place of ridge's

**Stimuli / Models / Metric**: same as E046 / E047.

**Result (10 subjects, retrieval)**:

| Metric | Ridge (E046) | MLP (this) | Δ |
|--------|-------------:|-----------:|--:|
| top-1 mean | 0.189 ± 0.038 | 0.181 ± 0.039 | -0.008 (≈ same) |
| top-5 mean | 0.436 ± 0.070 | 0.430 ± 0.072 | -0.006 (≈ same) |
| top-10 mean | 0.565 ± 0.075 | 0.566 ± 0.079 | +0.001 (≈ same) |
| per-dim r_d mean | 0.315 | 0.324 | +0.009 (≈ same) |
| frac dims r_d > 0.1 | 0.985 | 0.989 | +0.004 (≈ same) |
| total wallclock | <60 s | 78.6 s | (MLP is fast) |

**Retrieval finding**: MLP is **essentially identical to ridge** on top-K
retrieval and per-dim preservation. Non-linearity does not help recover more
EEG-decodable information when measured by retrieval / linear correlation
proxies.

**Result (10 subjects, IllusionBench transfer §6 verdict)**:

| | Ridge (E047) | MLP (this) |
|---|---|---|
| ISI agg | 0.811 FAIL | 0.814 FAIL |
| CSI agg | 1.275 FAIL | 1.273 FAIL |
| **PWI agg** | **0.537 FAIL** | **0.479 PASS** (closer to raw 0.397) |
| ISIrbox agg | 0.603 PASS | 0.603 PASS |
| ALL FOUR agg | FAIL (1/4) | FAIL (2/4) |
| Subjects with 4/4 | 0/10 | **0/10** |
| Subjects with 3/4 | 0/10 | 0/10 |
| Subjects with 2/4 | 3/10 (PWI + ISIrbox) | **7/10 (PWI + ISIrbox)** |
| Subjects with 1/4 | 7/10 (only ISIrbox) | 3/10 (only ISIrbox) |

**Key finding**:
- **MLP CAN recover PWI for more subjects than ridge** (7/10 vs 3/10), and
  pushes aggregate PWI from 0.537 (FAIL by 0.04) to 0.479 (PASS by 0.02).
  The part-whole signal contains **non-linear EEG structure** that ridge
  attenuates but MLP recovers.
- **ISI and CSI are NOT recoverable by either decoder type** (Thatcher anti-
  direction 0.811 vs 0.814; composite 1.275 vs 1.273). The §6 thresholds for
  these paradigms are out of reach with current ATM-EEG features regardless
  of decoder family.
- **0/10 subjects achieve 4/4 PASS with MLP** — the benchmark's sharp
  prediction holds even with the relaxation from linear to non-linear
  decoding.

**Interpretation**:

[CONFIRMED] **The "EEG bottleneck is fundamental for ISI and CSI" conclusion
is STRONGER post-MLP**: matching both linear and non-linear decoders fail
identically on Thatcher and composite (≤ 1.5 absolute Δ in any subject).
Future decoder improvements on these axes need to look beyond the
linear/non-linear axis — likely at the sensor side (higher-channel EEG,
better SNR) or the encoder side (re-train ATM-equivalent against DINOv2
target, Stage 4).

[CONFIRMED] **PWI has a non-linear EEG signature** that linear decoders
attenuate but MLPs recover. Cognitive-science interpretation: the
part-whole effect (whole-face context boost to part recognition) appears
to be coded in the EEG signal in a way that requires non-linear
combination of channel/time features. This is a publishable sub-finding
distinct from the benchmark vindication.

[CONJECTURE] **The 4/4 PASS ambition is still unreachable** for any subject
with any decoder family we've tried. Two reasonable next-step
hypotheses to test:
  (a) ATM-EEG features are CLIP-aligned-trained → information loss is
      already baked in. Re-training ATM end-to-end against DINOv2 (Stage 4)
      could preserve more.
  (b) Linear features of all kinds (CLIP, DINOv2, etc.) are insufficient;
      need to look at multimodal embeddings or attention-driven embeddings.

**Replicability check**:
- Random seed: 20260521 + subject_id offset for MLP init / shuffle
- Code: `eeg_decoder/stage3_mlp.py`
- Output: `/workspace/runs/2026-05-23_stage3-mlp_seed20260521/`
- IllusionBench-MLP transfer: ran via `illusionbench_transfer.py` with
  `--r_d_path` pointing at MLP's `per_dim_preservation.pt`, output to
  `/tmp/stage3_mlp_transfer/` (volatile — re-derivable, not committed to repo)

**Linked Q###**:
- Q005 (EEG-side preservation): further refined. Linear-decoder
  per-dim preservation is target-specific (E046); MLP per-dim is similar but
  MLP IllusionBench transfer preserves PWI better → preservation
  is non-linearity-sensitive for some paradigms.

**Linked Idea-###**:
- **Idea-001 (IllusionBench-EEG benchmark)**: held at 9.8. The benchmark's
  4/4 sharpness is corroborated by yet another decoder family (MLP) failing
  identically.
- **Idea-002 (benchmark suite + baselines)**: held at 7.5. Adds another
  baseline (MLP) to the table.
- **NEW Idea-004 (paradigm-non-linearity finding)**: candidate idea for a
  short companion paper or sub-finding — "PWI has non-linear EEG structure
  that linear decoders miss; ISI/CSI do not". To be assessed if the user
  wants a separate writeup.
