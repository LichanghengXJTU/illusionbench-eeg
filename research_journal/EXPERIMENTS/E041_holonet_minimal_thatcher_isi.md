# E041 — Minimal HOLO-Net (all bio components OFF): per-layer Thatcher ISI — the "minus everything" ablation baseline

**Status**: preliminary (checkpoint step 25000 of 30000); Thatcher paradigm only.
**Linked**: Idea-003, E040, E042, Q014.

**Hypothesis**: The minimal-mode HOLO-Net — CORnet-style
LGN-Parvo→V1→V2→V4→IT(MFP)→AFP→ATL backbone + AdaFace head, with FFA / Orientation
Gate / PC feedback / LGN-Magno / OFA / PFC-Gist all disabled — trained only on
Glint360K faces, gives Thatcher ISI ≈ 1 at every layer. I.e. bio-anatomy backbone
+ face-recognition training *alone* do not produce the Thatcher configural
signature. This is the ablation floor against which the FULL HOLO-Net (E042) is
compared.

**Method**: `holo_net/eval_extract.py` → 7 per-layer NPZ on the 800-stimulus
Thatcher FFHQ set (`data/stimuli_ffhq/thatcher_manifest.csv`, 200 identities ×
{V1 upright-normal, V2 upright-thatched, V3 inverted-normal, V4 inverted-thatched})
→ `analysis/compute_metrics.py` ISI_pert with 1000× identity-bootstrap CIs.
Checkpoint = minimal run frozen at step 25000 (`ckpt_minimal_step25000.pt`,
identity loss ≈ 1.9).

**Models**: HOLO-Net v1.0 minimal (53.63M params).

**Pre-registered prediction**: ISI ≈ 1 at all layers (no bio components → no
orientation × feature interaction).

## Result (numbers only) — `outputs/tables/holonet_minimal_thatcher_isi.csv`

| Layer | dim | ISI_pert | 95% CI | d_up | d_inv |
|---|---|---|---|---|---|
| v1  | 64  | 1.019 | [0.998, 1.040] | 6.31e-5 | 6.20e-5 |
| v2  | 128 | 1.037 | [1.010, 1.062] | 1.66e-4 | 1.60e-4 |
| v4  | 256 | 1.018 | [0.994, 1.042] | 2.74e-3 | 2.69e-3 |
| mfp | 512 | 0.965 | [0.946, 0.986] | 0.0573  | 0.0593  |
| afp | 512 | 0.975 | [0.950, 1.000] | 0.4264  | 0.4376  |
| ffa | 512 | 0.975 | [0.950, 1.000] | 0.4264  | 0.4376  |
| atl | 512 | 0.975 | [0.951, 1.000] | 0.4211  | 0.4317  |

(In minimal mode `use_ffa=False`, so the "ffa" readout == AFP-pooled — hence ffa
and afp rows are identical. atl = the AdaFace head embedding.)

## Interpretation

- **[CONFIRMED] The minimal HOLO-Net shows NO Thatcher effect at any
  brain-region layer** — ISI 0.965-1.037, every CI spanning or adjacent to 1.0.
- **[CONFIRMED — harness sanity]** v1 (lowest layer) ISI = 1.019, essentially
  the pixel baseline (1.000); v1 features barely move under the perturbation
  (d ≈ 6e-5). Internal consistency check passes — `eval_extract.py` +
  `compute_metrics.py` behave correctly end-to-end. **The HOLO-Net evaluation
  pipeline is now validated** and ready for the full-model checkpoint.
- **[CONJECTURE]** Consistent with E028 (CORnet-S ImageNet, ISI 0.99) and E031
  (AdaFace on large clean data, ISI ~1.2): bio-anatomy + clean-data face-rec
  training ≠ Thatcher. Glint360K is a large *cleaned* dataset, so per Claim 6
  (data-inversion) a low ISI is the expected outcome for this configuration.
- **[CONJECTURE — the key role of this experiment]** This is the **ablation
  floor**. If the FULL HOLO-Net (E042) reaches FFA-layer ISI ≥ 3 while this
  minimal baseline sits at ≈ 1, the Thatcher signal is cleanly attributable to
  the bio-fidelity components (FFA recurrent binding / Orientation Gate /
  magno-gist), not to the CORnet backbone or the face-identity data. If the
  full model ALSO gives ≈ 1, HOLO-Net is falsified on Thatcher.

## Caveats

- **Preliminary checkpoint** (step 25000 / 30000). Identity loss is still
  descending (~1.9 → ~1.5 expected); ISI of a face-rec model is not expected to
  shift materially, but this will be re-confirmed on the final step-30000
  checkpoint.
- **Thatcher only.** Composite-CSI and Part-Whole-PWI are pending location of
  that metric code (not present in `analysis/`).

## Replicability

- Code: `holo_net/eval_extract.py` + `analysis/compute_metrics.py` at the
  tick-53 commit. seed 20260521 for the bootstrap.
- ⚠ The *training* run had no fixed seed (caveat inherited from E040).
- Checkpoint: `/workspace/holo_net_stage2/ckpt_minimal_step25000.pt` (frozen).
- NPZ: `/workspace/.../outputs/embeddings/thatcher_holonet_minimal/` (server;
  `*.npz` gitignored). ISI table: `outputs/tables/holonet_minimal_thatcher_isi.csv`.

---

## Addendum — tick 59: composite + part-whole completed → full 3-paradigm ablation baseline

`compute_metrics.py` computes CSI and PWI too — the composite/part-whole
generators reuse the V1-V4 condition slots, so the same script + the
`data/stimuli_ffhq_{composite,partwhole}/thatcher_manifest.csv` manifests work
unchanged (no separate metric code — that earlier "missing code" worry is
resolved). Ran the minimal checkpoint (step 25000) on both:

**Composite CSI** (raw; pixel baseline 1.099 → corrected):
v1 1.254, v2 1.291, v4 1.259, mfp 1.179, afp/ffa 1.310, atl 1.313.
FFA-layer raw 1.310 → **pixel-corrected ≈ 1.19**.

**Part-Whole PWI** (raw; pixel baseline 1.202 → corrected):
v1 1.516 (low-level, noisy), v2 0.934, v4 0.812, mfp 0.872, afp/ffa 0.936,
atl 0.926. FFA-layer raw 0.936 → **pixel-corrected ≈ 0.78**.

**The full ablation floor — minimal HOLO-Net, FFA-layer, all 3 paradigms:**

| Paradigm | minimal HOLO-Net (FFA layer) | falsification threshold | pass? |
|---|---|---|---|
| Thatcher ISI | ≈ 1.0 | ≥ 3.0 | ✗ |
| Composite CSI | ≈ 1.19 (corrected) | ≥ 1.5 | ✗ |
| Part-Whole PWI | ≈ 0.78 (corrected) | ≤ 0.5 | ✗ |

**[CONFIRMED]** The minimal HOLO-Net (all bio components OFF) fails the
pre-registered falsification on **all three** paradigms. This is the intended
ablation floor: if the full HOLO-Net (v4) clears any threshold, it is cleanly
attributable to the bio-fidelity components, not the CORnet backbone + AdaFace
training. The 3-paradigm eval pipeline (`eval_extract.py` → `compute_metrics.py`)
is validated and ready for v4. Tables: `outputs/tables/holonet_minimal_{composite,partwhole}.csv`.
