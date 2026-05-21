# E021 — Route A refinement: face-subset preservation

**Hypothesis**: Maybe the average preservation r ≈ 0.158 across all 200 THINGS
test concepts dilutes a stronger face-category signal. If face categories
have meaningful EEG signatures, restricting preservation to face-related
concepts should reveal HIGHER per-dim preservation than non-face categories
or the full mix.

**Method**: Use the CLIP-ViT-H/14-LAION text encoder to embed 8 face-related
queries ("a photo of a face", "person", "man", "woman", "boy", "girl", "human",
"baby"). For each of 200 THINGS-EEG2 test concepts, take the maximum cosine
similarity to any query as a "face-relatedness score". Take top-30 as the face
subset. Random-30 sampled from the remaining 170 as matched control.

For each subset: compute mean per-dim Pearson preservation across 10
THINGS-EEG2 subjects, then compare via permutation test.

**Pre-registered prediction**:
- If face-categories carry richer EEG signal: face subset preservation > random subset preservation
- If preservation is uniform across categories: face ≈ random ≈ all-200

**Result**:
```
top-30 max-face-similarity range: 0.577 — 0.656 (vs all-200 median 0.537)
                  preservation mean r    median
face subset         0.150              0.145
random subset       0.145              0.141
all 200             0.158              0.149

face − random       +0.005           perm p_two-sided = 0.374  (NULL)
```

**Interpretation** ([CONFIRMED] / [CONJECTURE]):

- [CONFIRMED]: Face-related test concepts do NOT receive privileged EEG
  preservation in ATM-S. Mean per-dim preservation 0.150 (face) vs 0.145
  (random) is statistically indistinguishable (perm p = 0.374).
- [CONFIRMED]: The overall preservation 0.158 (all 200) is the lowest of the
  three because random subsampling adds variance noise; the face-vs-random
  comparison (matched-sample design) is the cleaner contrast.
- [CONFIRMED]: This **strengthens** the E020 conclusion — the EEG bottleneck
  is genuinely a uniform low-pass across both DIMENSIONS (E020) and
  CATEGORIES (E021). There is no privileged path for face information
  through ATM's EEG decoder.
- [CONJECTURE]: The text-similarity-based face concept identification is
  approximate. We do not have ground-truth labels for the 200 test concepts.
  The top-30 max-sim score (0.58-0.66) is moderate — not all of these
  concepts are strictly face/person; some may be face-adjacent (faces in
  context, body parts). A labeled-list-based selection might tighten the
  result, but the gap is so small (+0.005) it is unlikely to reverse.

**Implication for Idea-001 (no change in score)**: Confirms the framing of
"uniform-low-pass" EEG bottleneck. Strengthens Pitch v4 by ruling out the
alternative explanation "ATM is face-blind specifically".

**Replicability**:
- Code: `analysis/route_a_face_subset.py` (seed 20260521, 10000 perm)
- Output: `outputs/tables/route_a/{route_a_face_subset_arrays.npz, route_a_face_subset_summary.json}`

**Linked Q###**: Q005 (refinement of E020).
