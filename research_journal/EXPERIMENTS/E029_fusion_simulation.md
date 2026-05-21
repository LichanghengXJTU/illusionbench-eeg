# E029 — Multi-anchor fusion simulation (CLIP × CORnet)

**Hypothesis** (sub-path d of Idea-003): If CLIP captures Thatcher but lacks
Part-Whole, and CORnet-S captures Part-Whole but lacks Thatcher, then a fused
embedding space — concatenating both priors — should be paradigm-consistent:
high ISI on Thatcher (from CLIP component), low PWI on Part-Whole (from
CORnet component), reasonable CSI on Composite.

**Method**: Theory: for L2-normalized embeddings a (D_a-dim) and b (D_b-dim),
the concatenation [a, b] has norm √2, and cos([a1,b1]/√2, [a2,b2]/√2) =
(cos_a + cos_b) / 2, so fused distance is the average of component distances.

We tested:
1. Pure CLIP-bigG14
2. Pure CORnet-S
3. 50/50 unit-concat (equal magnitude)
4. Magnitude-balanced (scale CORnet so its mean cross-identity distance
   matches CLIP's)
5. Weighted concat with w_cornet ∈ {0.3, 0.5, 0.7, 0.9} (after rebalancing)

All computed on the existing P06 and P22 NPZs for the three paradigms.

**Result**:
```
Paradigm    Pure CLIP   Pure CORnet   50/50    Mag-bal   w_corn=0.3  w_corn=0.7
Thatcher    6.760       0.991         4.461    3.025     5.305       1.515
Composite   1.325       1.151         1.281    1.240     1.299       1.179
Part-Whole  0.704       0.214         0.541    0.416     0.602       0.270
```

**Interpretation** ([CONFIRMED]):

- [CONFIRMED] **Fusion = distance averaging, no emergent capability**. Fused
  ISI/CSI/PWI is essentially a weighted average of the two pure-prior values.
  This is mathematically forced by the L2-normalization + concatenation.
- [CONFIRMED] **Cannot achieve paradigm-consistent illusion sensitivity via
  simple multi-anchor fusion**. To preserve Thatcher (CLIP's strength) AND
  amplify Part-Whole (CORnet's strength), the two priors must contribute
  oppositely, and they **dilute each other** in cosine-distance space.
- [CONFIRMED] No fusion weight gives "all three paradigm indices favorable
  (Thatcher > 4, Composite > 1.5, Part-Whole < 0.5)". The Thatcher-vs-Part-Whole
  tension is intrinsic to the fusion formulation.

**Implication for Idea-003**:
- ❌ Sub-path (d) multi-anchor fusion alone is **not sufficient** to realize
  the bio-inspired paradigm-consistent vision. The two component priors
  must each be paradigm-consistent themselves; fusing two paradigm-specific
  priors yields a weighted-average paradigm-specific result.
- ✅ Sub-path (a) **face-tuned CORnet** remains the right candidate: train
  CORnet-S architecture on face-recognition objective so it acquires
  face-feature tuning, while retaining recurrent IT integration. This is
  a multi-day training effort.
- ⏭ Alternative: explore **NSD-aligned encoders** (Allen 2022, Conwell 2024)
  which use human ventral stream fMRI to supervise pretraining. These may
  naturally combine bio-anatomy with face-feature tuning.
- ⏭ Alternative: **predictive coding networks (PredNet, etc.)** which have
  hierarchical bottom-up + top-down structure that may yield different
  illusion sensitivity profiles than CORnet.

**Replicability**:
- Script: `analysis/fusion_simulation.py`
- Inputs: P06 CLIP-bigG14 + P22 CORnet-S NPZs across 3 paradigms
- Output: `outputs/tables/fusion/fusion_{thatcher,composite,partwhole}.csv`

**Linked**: refines Idea-003 understanding — sub-path (d) shown insufficient;
sub-paths (a), (b), (c) remain candidates.
