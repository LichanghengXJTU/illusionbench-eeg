# E006 — Q005 EEG-side feasibility scoping (no new numbers this tick)

**Status**: scoping document. No new ISI values produced. Output is a refined
experimental plan for Q005 to guide ticks 5+.

## Sharper formulation of Q005

After E001-E005, the picture on the IMAGE SIDE is:
- Image-text contrastive priors (CLIP / SigLIP / MetaCLIP) all show face-Thatcher
  ISI 4-7 in their image embedding space.
- ATM uses CLIP-ViT-H/14 LAION as its visual anchor — IDENTICAL to our P04
  (which gave ISI=5.50 in E002). Confirmed by reading
  `EEG_Image_decode/Retrieval/eegdatasets_leaveone.py`:
  ```python
  model_type = 'ViT-H-14'
  vlmodel, ..., feature_extractor = open_clip.create_model_and_transforms(
      model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
  ```
- AVDE, ENIGMA, HVF use the same family of CLIP image priors.

**So the question is not "does the image side have Thatcher ISI" (it does, 5.5).
The question is "does the EEG-decoded image embedding inherit that signature".**

## What "EEG-decoded image embedding" means concretely

ATM at inference time: given test EEG signal `e`, ATM's encoder `f` produces
`f(e)` ∈ R^1024 (CLIP-H/14 space). The training objective minimizes
distance(f(e), CLIP(x)) where x is the image the subject viewed.

After training, `f(e)` is approximately CLIP(x) — but with NON-TRIVIAL ERROR because
EEG is low-SNR. The error is the EEG bottleneck.

For our Thatcher question:
1. The CLIP-space ISI = 5.5 represents an embedding-space property of face Thatcher.
2. ATM's `f(e)` produces a noisy estimate of CLIP(x) when x is a THINGS test image.
3. The Thatcher signature in CLIP space is in CERTAIN DIRECTIONS of that space.
4. If those directions are EEG-recoverable, ATM-decoded embeddings will preserve
   the Thatcher ISI signature.
5. If those directions are EEG-destroyed (e.g., they correspond to noise in EEG),
   ATM output will collapse the signature.

## Hard problem identified

**We have NO EEG data of subjects viewing Thatcher stimuli.** ATM was trained on
THINGS-EEG2 which is natural object images (1854 concepts, ~half non-face).
Even the test set has no Thatcher pairs. So we cannot directly query "what does
ATM output for V1 vs V2".

## Three viable experimental routes (tick 5+ candidates)

### Route A — Direct preservation analysis (highest fidelity)

Take ATM's pretrained encoder. On THINGS-EEG2 test set:
1. For each test trial (image, EEG): compute `clip_target = CLIP(image)` and
   `eeg_pred = ATM(EEG)`.
2. Build a tensor `T` of "preservation per CLIP dimension": correlation between
   `clip_target[:, d]` and `eeg_pred[:, d]` across all trials, for d=1..1024.
3. Separately, on the FFHQ Thatcher battery: compute the V1/V2 difference vector
   `Δ = CLIP(V1) − CLIP(V2)` and its components in d=1..1024.
4. Compute "Thatcher dimensions" — the top-K dimensions where Δ has high
   magnitude consistently across identities.
5. Check whether these dimensions are well-preserved (T[d] large) or
   destroyed (T[d] small).

**Cost**: cheap. Only requires ATM trained checkpoint + THINGS-EEG2 test data.

**Sanity prediction**: ATM was trained on THINGS, NOT face Thatcher. The
Thatcher dimensions are face-feature-specific and likely live in dimensions
that EEG-of-objects does not cover well. We predict POOR preservation → "EEG
destroys it" narrative.

**This is the cleanest first experiment. Tick 5 candidate.**

### Route B — Train our own minimal EEG encoder (medium effort)

Train a small EEG→CLIP encoder on THINGS-EEG2 using only ~30 face categories
from THINGS-1854. See if face-trained EEG decoder preserves the Thatcher
signature on natural face stimuli (proxy).

**Cost**: 1-2 days of training compute on H100. Manageable.

### Route C — Probe the AVDE/ENIGMA pipelines too (broader claim)

After Route A on ATM, replicate on AVDE and ENIGMA for paper-level breadth.

## Resources needed for Route A (tick 5 work)

1. ATM trained checkpoint: HF `LidongYang/EEG_Image_decode`. Verify exists.
2. THINGS-EEG2 test set (10 subjects × 200 test concepts × ~80 reps): preprocessed
   version on HuggingFace `gasparyanartur/things-eeg2` or original OSF `3jk45`.
3. Server already has the open_clip dependency for CLIP-ViT-H/14 LAION (matches
   our P04).
4. ATM repo cloned at `/workspace/eeg_repos/EEG_Image_decode`.

## What I am doing in this tick (no new experimental data)

Just this scoping document + repo clone. Next tick (5) attempts Route A.

**Linked Q###**: Q005.
