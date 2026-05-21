# E027 — 2AFC verification: ISI_pert vs ROC-AUC sanity check

**Hypothesis**: ISI_pert (embedding-distance ratio) should rank-correlate
strongly with an independent ROC-AUC measure that directly tests "can the
model's cosine distance separate orientation-A-modified pairs from
orientation-B-modified pairs?" If they agree across 17 models on 3 paradigms,
ISI_pert is a valid behavioral analog.

**Method**: For each model + paradigm, compute AUC = P(d(pair_A) > d(pair_B))
via vectorized Mann-Whitney U statistic. Spearman + Pearson correlation with
ISI_pert across all 17 models per paradigm.

**Result** (across 17 models per paradigm):
```
Thatcher:    Spearman ρ(ISI_pert, AUC) = +0.966  p < 0.0001
Composite:   Spearman ρ                = +0.931  p < 0.0001
Part-Whole:  Spearman ρ                = +0.929  p < 0.0001
```

**Interpretation**:
- [CONFIRMED] ISI_pert is essentially equivalent to ROC-AUC under
  rank-ordering across 17 priors on all three paradigms (ρ = 0.93-0.97).
- [CONFIRMED] Claim 1 (training-objective × paradigm dissociation) is robust
  under behavioral-style framing: it does not depend on the specific
  embedding-distance-ratio formulation.
- [CONFIRMED] FaceNet AUC on Thatcher = 0.53-0.58 — only marginally above
  chance, consistent with no Thatcher illusion.
- [CONFIRMED] CLIP-bigG14 AUC on Thatcher = 0.963 — near-perfect separation
  of upright vs inverted modified pairs.
- [CONFIRMED] On Part-Whole, AUC is mostly < 0.5 (= "orientation A modified
  pairs have smaller distance than orientation B"), consistent with the
  PWI < 1 finding (whole-context dampens vs part-isolated amplifies).

**Replicability**:
- Code: `analysis/verification_2afc.py`
- Output: `outputs/tables/verification/verification_2afc_{thatcher,composite,partwhole}.csv`
- Summary JSON per paradigm with correlation values.

**Linked**: Strengthens Claim 1 to "robust under multiple operationalizations".
