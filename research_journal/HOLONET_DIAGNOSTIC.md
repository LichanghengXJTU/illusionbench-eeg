# HOLO-Net diagnostic critique — was the bio-mechanism imitated faithfully?

Written tick 71, in response to the user's FLAG-002 directive: before any new
experiment, deeply analyse whether v5 failed because the bio-science mechanism
wasn't well imitated, or for some other reason.

**TL;DR**: Yes, the bio-mechanism implementation was meaningfully flawed.
Three specific gaps are identifiable from the literature: (1) HOLO-Net's
Thatcher mechanism (Orientation Gate on the whole face) is **not** the
mechanism the brain uses (literature: local-feature orientation tuning, not
a global up/down gate); (2) HOLO-Net is missing an fSTS analogue — the
brain region whose response *specifically* tracks Thatcher grotesqueness;
(3) the FFA holistic-binding module is implemented as gist-conditioned
self-attention, which does not naturally encode the second-order spatial
relations Thatcher tests. Each gap is fixable with a concrete architecture
revision. A v2.0 sketch is at the end.

---

## 1. The user's question, framed

The user's framing (FLAG-002 option 4):

> "If we cannot achieve the story with a brain-science-inspired framework,
> we should carefully analyse whether the brain-science mechanism wasn't
> imitated well, or whether there is some other reason."

Two candidate explanations for v5's refutation:

- **(A) Implementation gap** — HOLO-Net's specific architectural choices do
  not faithfully implement the actual neural mechanism. A better-aligned
  bio-architecture could succeed even under face-identity training.
- **(B) Other** — primarily the training objective hypothesis (per Idea-001,
  E004): face-identity training simply does not emerge configural illusions
  regardless of architecture. Then no architecture revision under identity
  training would work.

This document weighs A vs B with grounding from the neuroscience literature.

---

## 2. What the literature actually says about the Thatcher mechanism

Three relevant primary findings:

### 2.1 Orientation-sensitive LOCAL feature processing (Psalta, Young, Thompson & Andrews 2014)

The Thatcher effect is *better explained by a disruption to the processing of
purely local facial features* than by second-order spatial relations alone.
Their evidence: the inversion effect is also apparent when only the eye region
or only the mouth region is visible — i.e., the local feature itself is
orientation-tuned in the upright-context. ([Psalta et al. 2014, Acta Psychol;
PMC4191594](https://pmc.ncbi.nlm.nih.gov/articles/PMC4191594/))

This is a **direct challenge to HOLO-Net's design §7 assumption** that the
mechanism is global whole-face configural binding. The actual mechanism is at
the level of *individual face-feature detectors* that are orientation-tuned.

### 2.2 fSTS, not FFA, carries the illusion-specific signal (Psalta et al. 2014 fMRI follow-up)

The face-selective superior temporal sulcus (fSTS) shows a release from fMRI
adaptation when normal ↔ Thatcherized faces are alternated **only when
upright**. The FFA shows release in *both* upright and inverted — i.e., FFA
is sensitive to Thatcherization but does **not** produce the
upright-specific illusion signature. ([Psalta et al. 2014, J Neurosci /
Frontiers commentary](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00289/full); also [PMC4298288](https://pmc.ncbi.nlm.nih.gov/articles/PMC4298288/))

**HOLO-Net does not have any fSTS analogue.** Its architecture stops at FFA
→ ATL, missing the superior-temporal-sulcus stream entirely. If the
literature is right that fSTS is where the Thatcher-specific signal lives,
HOLO-Net is architecturally incapable of producing it.

### 2.3 Monkey face-patch evidence — a discrete face patch correlates with Thatcher (Taubert et al. 2015, J Neurosci)

A monkey face-selective patch shows differential responses correlated with
the Thatcher illusion. ([Taubert et al. 2015,
J Neurosci 35(27):9872](https://www.jneurosci.org/content/35/27/9872))
This is consistent with a *discrete circuit* contribution, not a generic
"holistic binding" property of the whole face system.

### 2.4 N170 ERP — orientation × Thatcher interaction at ~170 ms (Boutsen et al. 2006)

N170 latency is delayed by Thatcherization more for upright than inverted
faces; the modulation is face-specific (does not occur for houses).
([Boutsen et al. 2006, NeuroImage; PMID 16632381](https://pubmed.ncbi.nlm.nih.gov/16632381/))
This is consistent with **early, fast** orientation × local-feature processing
— not a slow recurrent global integration.

---

## 3. HOLO-Net's actual implementation, point-by-point, vs the literature

| Component / mechanism | HOLO-Net v5 implementation | Literature mechanism | Faithful? |
|---|---|---|---|
| The Thatcher mechanism | Orientation Gate (whole-face up/down classifier) suppresses FFA recurrence on inverted faces ("only upright engages configural binding") | Orientation-tuned LOCAL feature detectors (Psalta 2014); fSTS-specific upright tracking | **NO** — wrong locus (global vs local), wrong region (FFA gating vs fSTS-specific signal) |
| Orientation Gate input | GAP'd `afp_spatial` → 2-class classifier on whole-face orientation | Local feature-level orientation processing | NO — global classifier cannot do per-feature orientation tuning |
| FFA module | 3-step recurrent self-attention with magno-gist as global query | FFA shows Thatcher sensitivity in BOTH orientations (no specific upright-only signal there) | Partial — FFA *is* part of face processing, but it is NOT the locus of the upright-specific signal |
| fSTS analogue | **MISSING** | fSTS carries the upright-specific Thatcher tracking | NO — this region is not in the architecture at all |
| Second-order relational coding | Implicit in self-attention K·Q similarities, with K and V from AFP global features | Required (alongside local orientation tuning) | Weak — no explicit relational primitive (Capsule-style routing-by-agreement, GLOM-style islands) |
| LGN-Magno / PFC-Gist top-down | Bar (2003) magno-gist pathway → FFA query | Real top-down face-template predictions exist (Bar 2003), but FFA-specific | OK in spirit; not the Thatcher-critical mechanism per literature |

**The largest concrete gap**: HOLO-Net's Thatcher mechanism is built around
the wrong neural correlate. The design assumed "upright vs inverted whole-face
gate → FFA configural binding" — the literature says "orientation-tuned local
features (Psalta) + fSTS-specific upright tracking". The Orientation Gate
that HOLO-Net implements would correctly down-weight FFA processing for an
inverted whole-face, but it cannot encode the *upright-context-conditioned
local-feature orientation*-sensitivity that is the literature's actual answer.

---

## 4. Weighing implementation gap (A) vs other reasons (B)

Evidence that (A) implementation gap matters:

- Concrete bio-mechanism mismatches identified above (Thatcher mechanism,
  missing fSTS).
- The FFA module is not inert (PWI 2.20 — it does something distinctive);
  the architecture *can* produce paradigm-specific signals; it just produces
  the wrong one for Thatcher.
- HOLO-Net was designed against Tsao/Freiwald face-patch literature
  (MFP/AFP/FFA hierarchy) but not against the Psalta 2014 mechanism that
  specifically addresses Thatcher.

Evidence that (B) training objective alone explains the result:

- Claim 2 (E004): all face-identity-trained models so far (FaceNet ×2,
  ArcFace, AdaFace variants except one, HOLO-Net) give ISI ≈ 1.0 on the
  upright-Thatcher manipulation. That is a strong objective-side regularity.
- Idea-001's positive finding: image-text contrastive training emerges
  Thatcher (CLIP, SigLIP, MetaCLIP — multiple architectures, same objective).

**Weighed reading**: both (A) and (B) are in play, but (A) is the more
*actionable* explanation. (B) is a strong empirical regularity but not a
mechanistic claim that "no architecture under identity training could
succeed"; it merely reports that *no architecture we have tested* under
identity training has succeeded. The architectures we have tested — FaceNet
(Inception, triplet), AdaFace (IR-50/IR-101, angular-margin), HOLO-Net
(CORnet-S-style + attention FFA + gate) — all share a feature: **none of
them implements orientation-tuned local-feature processing or an
fSTS-analogue circuit**. That common gap is consistent with (A).

A clean test: an architecture that *does* implement these mechanisms,
trained on identity. If it still gives ISI ≈ 1, (B) wins. If it succeeds,
(A) wins and HOLO-Net's specific implementation was the bottleneck. **This is
the experiment a HOLO-Net v2.0 should run.**

---

## 5. Proposed HOLO-Net v2.0 — minimal, literature-grounded changes

Three targeted revisions, each addressing one gap above:

### 5.1 Replace the Orientation Gate with per-feature orientation-tuned detectors

- Remove the global Orientation Gate (`OrientationGate` on GAP'd afp_spatial).
- Add a `LocalOrientationModule` that, at the V4/MFP level, applies per-spatial-location
  orientation-sensitive units to face-feature regions (eye, mouth) — implemented
  as either (a) a learned bank of oriented Gabor-like filters at face-feature
  spatial locations, or (b) explicit per-region orientation prediction heads
  whose loss is computed on per-feature inversion labels (we can synthesise
  these labels from the existing whole-face inversion + a fixed MediaPipe
  landmark mask, with no Thatcher-stimulus contamination).
- The local-feature orientation signal then modulates downstream face
  processing in a feature-conditioned way (per-spatial-location gating, not a
  scalar whole-face gate).

### 5.2 Add an fSTS-analogue branch

- Add a small branch off MFP/AFP whose role is **upright-specific feature-
  inversion detection** (an "fSTS-style" grotesqueness detector).
- Architecturally: a head that takes per-feature local orientation outputs
  (from 5.1) and whole-face orientation, and predicts "feature mis-orientation
  given whole-face is upright" — i.e., the literal Thatcher signal.
- Trained with auxiliary self-supervised orientation-permutation labels (no
  Thatcher stimuli).
- The branch's output is part of `out["fsts"]`, evaluated on IllusionBench
  separately. The pre-registered Thatcher criterion would shift from FFA to
  fSTS layer.

### 5.3 Replace FFA self-attention with Capsule-style routing-by-agreement (or GLOM-style islands)

- The current FFA implements "holistic binding" as cross-attention with a
  global gist query — this does not naturally encode second-order spatial
  relations.
- A Capsule-network FFA (Sabour-Frosst-Hinton 2017 routing-by-agreement)
  explicitly represents part-pose and the spatial *agreement* between parts —
  exactly the second-order relations Thatcher tests.
- This is design-doc §8's pre-registered fallback; the literature analysis
  above justifies *promoting* it from "fallback if v1 fails" to "v2 main
  architecture because v1 was missing the relational primitive".

### 5.4 Keep identity objective, keep no-illusion-stimuli training

The training objective is held constant (identity / AdaFace) so the v2 test
is genuinely architecture-vs-architecture. If v2 succeeds, **the bio-mechanism
implementation was the cause**; if it fails, **the training objective is the
cause**. Either outcome is a clean answer to FLAG-002.

---

## 6. Cost estimate

- 5.1 (LocalOrientationModule): ~150 lines of code; 2-3 h.
- 5.2 (fSTS branch): ~100 lines + label generation (deterministic from MediaPipe
  landmarks, no new data); 2-3 h.
- 5.3 (Capsule FFA): ~300-400 lines; 1 day (Capsule layers are non-trivial).
- Retraining: same setup as v5 (~3 h on the H100).
- Re-eval: ~10 min via `eval_falsification.py`.
- Total: **~1 calendar day of work + 3 h training**.

If v2.0 also fails, we have strong evidence for (B) — training objective —
and the FLAG-002 option 2 (re-aim at image-text contrastive) becomes the
next test, with the architecture-side reason genuinely ruled out.

---

## 7. Recommendation

Proceed with HOLO-Net v2.0 per §5. The literature provides a clear, specific
diagnosis of where v1.0's bio-fidelity broke down (Thatcher mechanism = wrong
locus, missing fSTS, no relational primitive). The fix is targeted, not
speculative. And the experiment cleanly distinguishes implementation-gap (A)
from training-objective (B) — answering both the FLAG-002 question and
sharpening Claim 8.
