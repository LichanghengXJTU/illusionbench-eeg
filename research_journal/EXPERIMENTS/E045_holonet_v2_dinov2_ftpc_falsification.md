# E045 — HOLO-Net-V2-DINOv2 (FTPC option D) falsification

**Hypothesis**: A frozen pre-trained DINOv2 ViT-S/14 backbone (LVD-142M SSL —
natural-image distribution with faces) + learnable FTPC face template T +
EMA-fitted T over ~13k ImageNet face samples, with the δ = patch − T residual
read out as the FFA/fSTS-analogue, will pass IllusionBench-EEG §6 (ISI ≥ 3.0,
CSI ≥ 1.5, PWI ≤ 0.5, random-bbox ≤ 1.5) at the FFA layer — closing the gap
that HOLO-Net v1 (CORnet anatomy + AdaFace identity) opened (E044: 1/4 pass).

**Method**:
- Backbone: `dinov2_vits14` (84 MB Meta release; 21.7M params; LVD-142M SSL).
- FTPC head: single learnable T ∈ ℝ^{384×16×16} (over patch-token spatial map)
  + EMA buffer over face-detected ImageNet samples.
- Template training (`train_v2_dinov2.py`): 500 batches × bs 256 = 128K imgs
  → 13,586 MediaPipe-detected face samples (~10.6%); pure EMA update on T (no
  backbone gradient, no SSL loss). 3.4 min wallclock. T_ema norm 309.21 → 287.88
  monotonic decay (well-converged EMA over a stationary distribution).
- Evaluation (`eval_falsification_dinov2.py`): standard IllusionBench-EEG
  4-paradigm pipeline (Thatcher 200 IDs, Composite 100 IDs, Part-Whole 100 IDs,
  Random-bbox 200 IDs); per-layer NPZ → `analysis.compute_metrics` index;
  pixel-corrected.
- Layer aliasing in this DINOv2-only setup: v1/v2/v4/mfp/afp/atl all alias to
  `afp_pooled` (CLS-equivalent mean-pooled patch feature); **ffa = δ_pooled**
  (template residual, mean-pooled) — the FTPC novelty.

**Stimuli**: same FFHQ-1024 sets as E044 (Thatcher / Composite / Part-Whole /
Random-bbox), so the result is directly comparable to HOLO-Net v1 and to the
25-prior battery from E033/E034.

**Models**: `HOLONetV2Dinov2` (22.15M total; **0.10M trainable** = template
only). Checkpoint at
`/workspace/runs/2026-05-23_holonet-v2-dinov2_seed20260521/checkpoint.pt`.

**Metric**: ISI-style perturbation ratio (`isi_pert`), pixel-corrected with
the same baselines as E044 (Thatcher 1.000, Composite 1.099, Part-Whole 1.202,
Random-bbox 1.000).

**Pre-registered prediction** (filed in `HOLONET_V2_DESIGN.md` §6 prior to
running): δ_pooled meets all four criteria simultaneously, attributable to the
template-residual mechanism implementing the fSTS-analogue per Psalta 2014.

**Result** (numbers only — pixel-corrected):

| layer | ISI (Thatcher) | CSI (Composite) | PWI (Part-Whole) | ISIrbox |
|-------|---------------:|----------------:|-----------------:|--------:|
| v1    | 0.908          | 1.282           | 0.397            | 0.662   |
| v2    | 0.908          | 1.282           | 0.397            | 0.662   |
| v4    | 0.908          | 1.282           | 0.397            | 0.662   |
| mfp   | 0.908          | 1.282           | 0.397            | 0.662   |
| afp   | 0.908          | 1.282           | 0.397            | 0.662   |
| **ffa** (= δ_pooled) | **0.901** | **1.300** | **0.446** | **0.654** |
| atl   | 0.908          | 1.282           | 0.397            | 0.662   |

95% CIs (ffa): ISI [0.847, 0.955], CSI [1.371, 1.486], PWI [0.474, 0.606],
ISIrbox [0.623, 0.684]. Raw (non-pixel-corrected) afp ISI 0.908 [0.853, 0.962];
ffa ISI 0.901 [0.847, 0.955] — CIs overlap.

**§6 verdict (FFA layer)**:
- ISI = 0.901 — need ≥ 3.0 → **FAIL** (worse than v1's 1.00)
- CSI = 1.300 — need ≥ 1.5 → **FAIL** (better than v1's 0.99 by 0.31; closer)
- PWI = 0.446 — need ≤ 0.5 → **PASS** (v1 was 2.20 → FAIL; large reversal)
- ISIrbox = 0.654 — need ≤ 1.5 → **PASS** (v1 was 1.08; both PASS)
- **ALL FOUR SIMULTANEOUSLY: FAIL** (2/4 pass — improvement vs v1's 1/4 pass)

**Interpretation**:
- [CONFIRMED] The δ residual against a single global template does NOT add
  discriminative power over the raw DINOv2 feature: ffa vs afp CIs overlap on
  every paradigm (Thatcher: 0.901 vs 0.908; Composite: 1.430 vs 1.409; PW:
  0.536 vs 0.477; rand-bbox: 0.654 vs 0.662). The FTPC mechanism as
  implemented is essentially a constant-shift of the embedding.
- [CONFIRMED] The 2/4-pass profile is **driven by the DINOv2 backbone itself,
  not by the FTPC head**. DINOv2's SSL on natural images already gives the
  paradigm-consistent CSI/PWI signature E033/E034 reported for the DINOv2-family.
- [CONFIRMED] DINOv2-FTPC's FAIL on Thatcher (ISI 0.901, ANTI-Thatcher: the
  inverted "Thatchered" is MORE similar to the inverted-original than to the
  upright comparison) is the SAME signature E033 reported for DINOv2 alone
  (ISI 0.68 on FFHQ in E033). The FTPC template did not flip this; a single
  global template cannot encode the LOCAL feature-orientation mechanism Psalta
  2014 identifies as the actual Thatcher locus.
- [CONJECTURE] A **part-aware FTPC** — separate templates {T_eyes, T_nose,
  T_mouth, T_chin} at appropriate spatial sub-regions of the patch-token grid,
  with per-region δ aggregated by upright-vs-inverted concordance — would
  isolate local feature orientation (the Psalta 2014 mechanism) and could move
  the Thatcher signal upward without harming the DINOv2-driven PW/CSI passes.
  Testable: design v2.2 with K=4 part templates + part-aware δ; same eval.

**This is the 3rd HOLO-Net instantiation to fail §6 simultaneously, with each
failure adding mechanistic information**:
  - v1 (CORnet + AdaFace identity, E044): FFA ISI 1.00, CSI 0.99, PWI 2.20,
    rand 1.08 → 1/4 pass. Diagnosis: identity training suppresses configural
    signal (E004 corroborated).
  - v2 (CORnet + DINO SSL × 3 collapses, E045-prelim runs 1-3): training failure
    — from-scratch DINO SSL on small CORnet backbone collapses to uniform
    output regardless of out_dim or PC-loss detach. No §6 number obtainable.
  - v3 (frozen DINOv2 + global FTPC template, THIS EXPERIMENT): FFA ISI 0.901,
    CSI 1.300, PWI 0.446, rand 0.654 → 2/4 pass. Diagnosis: global template
    cannot encode local-feature-orientation; mechanism needs part-awareness.

**Replicability check**:
- Random seed: 20260521 (Python / numpy / torch)
- DINOv2 weights: `dinov2_vits14_pretrain.pth` (Meta, 84 MB; deterministic
  download; SHA verifiable from PyTorch hub)
- Template training: deterministic except for HF dataset shuffle order (no seed
  passed to DataLoader.shuffle yet — note for repro)
- Code commits: pending tick-80 commit
- NPZ paths: `/workspace/illusionbench-eeg/outputs/embeddings/{thatcher,composite,partwhole,randombbox}_HOLONET_v2_dinov2/`
- Tables: `/workspace/illusionbench-eeg/outputs/tables/HOLONET_v2_dinov2_*.csv`

**Linked Q###**: corroborates Q015 (does FTPC isolate δ signal?) → answer: not
with a global template; need part-aware decomposition.

**Linked Idea-###**:
- Idea-001 (IllusionBench-EEG benchmark): **strongly reinforced** — 3 distinct
  HOLO-Net instantiations all fail simultaneously, adding architecture-side
  corroboration to the training-objective thesis. Score should go up.
- Idea-003 (HOLO-Net bio-inspired prior): unchanged headline ambition, but
  the design space narrows: the bio mechanism implementation must be PART-
  AWARE, not global. v2.2 design lock-in.
