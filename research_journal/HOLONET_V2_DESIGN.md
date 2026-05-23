# HOLO-Net v2.0 — Face-Template Predictive Coding (FTPC)

Written tick 72, in response to the user's directive: bigger redesign (not
patches), no gates, must capture the *illusion-transmission mechanism*
explicitly, must also be strong on the normal identity task.

**TL;DR**: replace the entire "Orientation Gate + recurrent-FFA-binding"
mechanism of v1.0 with a single coherent framework — **Face-Template
Predictive Coding**: the system learns a canonical UPRIGHT face template
(face-specific top-down prior), and the prediction-error between bottom-up
local features and the template *is the illusion signal*. No gates anywhere.
The signal propagates through an explicit dorsal/fSTS-like "anomaly stream"
that is the locus of the upright-specific illusion response. The ventral
identity stream is left clean, so normal-task performance is preserved.

---

## 1. The user's design constraints (binding, tick 71→72)

1. **No gates.** v1's Orientation Gate was the wrong primitive; same for any
   per-feature gate fix.
2. **Illusion-transmission mechanism, explicitly modelled.** Not just "build
   an architecture that happens to produce ISI"; *propose where the illusion
   signal lives and how it propagates* through the visual system.
3. **Strong on normal tasks too.** The model must be a credible face-identity
   model independent of the illusion. No illusion-only-good toys.
4. **Holistic-processing grounded.** The mechanism must respect the brain's
   actual holistic face computation, not piecewise local tweaks.
5. **A coherent framework, not small revisions.**

## 2. The framework — Face-Template Predictive Coding (FTPC)

Three coupled streams in one model:

```
                       ┌──────────────────────────────────────────────┐
                       │            CANONICAL FACE TEMPLATE           │
                       │  (learnable per-region spatial codebook of   │
                       │   expected features for an upright face)     │
                       └────────────────────┬─────────────────────────┘
                                            │ top-down prior
                                            ▼
INPUT ─► LGN-Parvo ─► V1 ─► V2 ─► V4 ─► IT(MFP) ─► AFP ─►─ Δ(bottom-up, template)
                                            │                        │
                                            ▼                        ▼
                                        ATL (AdaFace identity)   ANOMALY MAP
                                            │                  (per-region
                                            │                   mismatch field)
                                            │                        │
                                            │            ┌───────────┴────────────┐
                                            │            ▼                        ▼
                                            │     anomaly-stream head      back-modulation
                                            │      (fSTS-analogue:           into AFP/FFA
                                            │     scalar grotesqueness        (additive,
                                            │     + per-region map)            never gate)
                                            ▼                        
                                  identity output (ATL)               
```

- **Ventral (identity)** — LGN-Parvo → V1 → V2 → V4 → IT(MFP) → AFP → ATL.
  Identical to v5's minimal-mode backbone (which already trains identity well
  to loss ≈ 1.5 on Glint360K-10K). **No gates, no recurrent FFA self-attention,
  no orientation gating.** This is the "normal task" stream and is the
  guarantee of strong identity performance.

- **Face template (top-down)** — a learnable per-spatial-location feature
  codebook representing the canonical upright-face configuration: a tensor
  `T ∈ R^{D × 7 × 7}` with the same shape as AFP_spatial. Learned either as
  a slow EMA of AFP_spatial over canonical-face training batches OR as a
  parameter trained with a "reconstruct expected face" objective.

- **Anomaly stream (dorsal / fSTS analogue)** — the prediction error
  `δ = AFP_spatial(x) − T` is the illusion signal. It is a spatial map; a
  small head (3 conv layers + GAP + MLP) reduces it to (i) a scalar
  grotesqueness score and (ii) a per-region anomaly map. **This stream is
  the locus of illusion sensitivity.**

- **Back-modulation (optional, additive only)** — the anomaly map can
  additively perturb AFP_spatial before it enters ATL (`AFP' = AFP + α·δ`),
  giving the identity head an "abnormality signal" without ever multiplying
  or gating. α is a learnable scalar (initialised at 0 → at init the
  identity path is unaffected → guaranteed not to hurt identity training).

## 3. Why this explains the illusions

**Thatcher (V1 upright-normal → V2 upright-thatched)**: V1 is a canonical
upright face → AFP_spatial(V1) matches the upright template → tiny δ → no
anomaly. V2 has feature-orientation-inverted regions on an otherwise upright
face → AFP_spatial(V2) mismatches the template specifically at eye/mouth
regions → large δ at those locations → large grotesqueness signal →
embedding (with back-modulation) is shifted strongly. d(V1,V2) is large.

V3 (inverted-normal), V4 (inverted-thatched): the whole face is inverted →
*neither* matches the upright template → both have a large global δ that is
not feature-specific → the local Thatcher perturbation barely changes the
already-large global mismatch → d(V3,V4) is small. **ISI > 1 by construction.**

**Composite (aligned vs misaligned face halves)**: aligned face halves match
the template's spatial layout → small δ → d(V1,V2) reflects identity. Misaligned
halves mismatch the template's expected spatial layout → large δ → embeddings
shift away from identity. d(V3,V4) is dominated by the alignment-anomaly
signal, not identity. CSI sensitivity emerges.

**Part-whole (eye-swap in whole-face context vs eyes on gray)**: V1=whole
matches template; V2=whole-with-eye-swap creates a local feature-swap
anomaly within a face context → δ amplified. V3=part-on-gray has no face
context, no template applies → no anomaly signal, just featural distance.
**The template makes whole-context swaps stand out** — which is what v5's
FFA already does to PWI 2.20; FTPC formalises and intensifies that
mechanism in a non-gate way.

**Random-bbox control**: face-region bboxes far from canonical face features
do not differentially mismatch the template (the template prior at non-face
spatial locations is weak/uniform), so the anomaly signal is not selectively
triggered → ISI_rbox stays near 1. The control passes by mechanism.

## 4. Why this answers the user's constraints

| Constraint | How FTPC meets it |
|---|---|
| No gates | The framework has zero multiplicative gating. The only modulation is *additive* (anomaly → AFP), with α initialised at 0. |
| Illusion-transmission mechanism | Explicit: the illusion signal IS the template-mismatch field `δ`; it propagates through the anomaly stream and (optionally) additively back into ventral. |
| Strong on normal tasks | The ventral identity path is identical to the v5-minimal model that trained identity to loss ~1.5. Anomaly stream is parallel; with α=0 at init, the identity head is untouched. Anomaly cannot hurt identity convergence. |
| Holistic processing | The template encodes the WHOLE upright face's expected feature configuration. Mismatch is a holistic comparison, not piecewise. |
| Brain-science grounded | Predictive coding (Rao & Ballard 1999, Friston 2010); face-specific top-down templates (Sigala & Logothetis 2002; Bar 2003); fSTS-style grotesqueness signal (Psalta et al. 2014; Taubert et al. 2015). |
| Not patches | Removes Orientation Gate, removes recurrent-self-attention FFA, removes generic Predify PC; replaces with a single coherent face-template-PC framework. |

## 5. Concrete architecture spec

```python
class HOLONetV2(nn.Module):
    def __init__(self, cfg):
        # Ventral identity stream (same as v5 minimal)
        self.lgn   = LGN_Parvo_only(cfg)
        self.v1, self.v2, self.v4, self.it = ...   # CORnet-S blocks
        self.afp   = AFPBlock(in_ch=cfg.it_channels, out_dim=cfg.afp_dim)

        # Canonical face template (learnable)
        self.template = nn.Parameter(torch.zeros(cfg.afp_dim, 7, 7))
        nn.init.normal_(self.template, std=0.02)

        # Anomaly stream
        self.anomaly_conv = nn.Sequential(  # δ-map → richer anomaly features
            nn.Conv2d(cfg.afp_dim, cfg.afp_dim, 3, padding=1),
            nn.GroupNorm(8, cfg.afp_dim), nn.ReLU(),
            nn.Conv2d(cfg.afp_dim, cfg.afp_dim, 3, padding=1),
            nn.GroupNorm(8, cfg.afp_dim),
        )
        self.anomaly_head = nn.Sequential(    # scalar grotesqueness
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(cfg.afp_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Back-modulation: additive only, α starts at 0
        self.alpha_modulate = nn.Parameter(torch.zeros(1))

        # ATL identity head (AdaFace; same as v5)
        self.atl = ATLHead(in_dim=cfg.afp_dim, out_dim=cfg.embed_dim)

    def forward(self, x):
        # Ventral
        parvo = self.lgn(x)
        v1, v2, v4, it = self.v1(parvo), ..., self.it(v4)
        afp_spatial = self.afp(it)                          # (B, D, 7, 7)

        # Template mismatch
        delta = afp_spatial - self.template[None]           # (B, D, 7, 7)
        anomaly_features = self.anomaly_conv(delta)         # (B, D, 7, 7)
        grotesque = self.anomaly_head(anomaly_features)     # (B, 1)

        # Additive (non-gating) back-modulation
        afp_modulated = afp_spatial + self.alpha_modulate * anomaly_features
        afp_pooled = F.adaptive_avg_pool2d(afp_modulated, 1).flatten(1)

        # Identity
        embed = self.atl(afp_pooled)

        return {
            "afp_spatial": afp_spatial,
            "anomaly_map": delta,
            "anomaly_features": anomaly_features,
            "grotesque": grotesque,
            "atl": embed,
            # plus v1, v2, v4, mfp for per-layer eval
        }
```

## 6. Training

- **Identity loss** (AdaFace on `atl`) — same as v5; primary signal.
- **Template prior**: keep it as a learnable parameter, regularised by an
  EMA-shrinkage toward `EMA(afp_spatial across canonical training samples)`
  with weight 0.01 — so the template can be analytically computed from
  canonical training data, but also fine-tuned by the anomaly loss. This
  avoids the chicken-and-egg problem of "template needs AFP, AFP needs
  template" by anchoring the template to a slow EMA early on.
- **Anomaly head loss**: synthetic anomaly labels via on-the-fly perturbation
  of training faces (random eye-swap from another identity in the batch,
  partial occlusion, low-frequency content shuffling at a non-face-feature
  region). Label = 1 if perturbed, 0 if canonical. BCE on the scalar
  `grotesque` head. Weight 0.1. **Critically: NO Thatcher-style perturbations
  in the synthetic anomaly set; all anomaly types are non-Thatcher.**
- **α_modulate**: free parameter; gradient-trained. We expect α to remain
  small for canonical batches (where anomaly_features are noise) and grow
  as the anomaly head becomes accurate.
- All v1 aux losses (PC reconstruction, orientation, gist, face-detect)
  REMOVED — they were the wrong primitives. Replaced by the single
  anomaly-head loss + the template EMA-regulariser.

## 7. How FTPC v2 maps to the §6 falsification

The pre-registered §6 criteria stay the same (ISI ≥ 3, CSI ≥ 1.5, PWI ≤ 0.5
contingent on metric direction, random-bbox ≤ 1.5), but the readout layer is
now the **anomaly_features** layer (the FTPC equivalent of fSTS), with the
ATL layer as the secondary readout. We'll report both; the anomaly stream is
the principled location of the illusion signal.

## 8. Open design choices (would value your input before building)

These are real design decisions where I have a default but could be wrong:

1. **Template parametrisation** — learnable parameter (default) vs frozen EMA.
   Default: learnable + EMA-regulariser (best of both).
2. **Back-modulation α initialised at 0** (default) vs at 0.1 (default-on).
   Default: 0. Conservative — identity is guaranteed untouched at init.
3. **Anomaly perturbations for the synthetic label set** — I propose:
   (a) eye-region swap with another identity, (b) random feature-region
   occlusion with a gray patch, (c) random feature-region with low-pass
   blur. Explicitly NOT including: feature inversion (would contaminate
   Thatcher test), part-swap with same identity (would contaminate
   part-whole). **Question for you**: do you have a stronger view on what
   "anomaly" should mean here?
4. **Drop generic Predify PC entirely** (default) vs keep it. The diagnostic
   says generic PC didn't help; face-template PC replaces it. Default: drop.
5. **Add the fSTS-analogue as a separate readout head** (default) vs make
   it ATL's modulation path only. Default: separate readout — that way the
   §6 falsification has a clean per-layer table including an fsts-like layer.

## 9. Implementation plan

| Tick | Action | Wall time |
|---|---|---|
| 72 (this) | Design doc + open-question discussion | done |
| 73 | Code `HOLONetV2` in a NEW file `holo_net/model_v2.py` (don't pollute v1) + sanity tests (forward, backward, shape, ablation API) | ~3 h |
| 74 | Synthetic-anomaly data augmentation in `holo_net/data.py` (a new `--anomaly_pct 0.3` flag) + loss-orchestrator update | ~2 h |
| 75 | Launch v2 training (30 000 steps, seed 20260521, GPU full) | ~3 h training |
| 76 | Mid-training poll + eval at intermediate checkpoint | poll |
| 77 | Final eval via `eval_falsification.py` (extended to read the new `anomaly_features` layer) | ~15 min |
| 78 | Verdict tick. If pass → write the positive headline; if fail → strong evidence the lever is the objective, FLAG-002 option 2 next. |

## 10. Why I'm confident in this redesign

It's not bold for the sake of being bold — it's the *minimum coherent
framework* that addresses the 3 diagnostic gaps simultaneously:
- the Orientation Gate's wrong-locus problem → solved by REMOVING gates and
  using template-level prediction error;
- the missing fSTS analogue → solved by the explicit anomaly stream;
- the missing second-order relational primitive → solved by template-vs-
  bottom-up comparison, which is intrinsically second-order (template encodes
  expected feature-relations at canonical spatial locations).

And it satisfies the user's invariants: no gates, normal-task performance
preserved (ventral path untouched, α starts at 0), holistic-processing
grounded (template = upright-face Gestalt), and proposes the
*illusion-transmission mechanism* explicitly (the anomaly signal δ).
