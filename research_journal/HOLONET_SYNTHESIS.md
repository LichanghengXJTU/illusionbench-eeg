# HOLO-Net synthesis — what the Idea-003 arc established (ticks 40-65)

Consolidation of the HOLO-Net experimental arc (E040-E044) and how it connects
to Idea-001. Written tick 66, after the decisive v5 falsification result.

---

## 1. What HOLO-Net was

A strict bio-fidelity face model: each layer mapped to a brain region
(LGN→V1→V2→V4→OFA→MFP→AFP→Orientation Gate→FFA→ATL, + Predify-style PC feedback
+ magno-PFC gist). 56.75M params. Trained on Glint360K face identities with an
AdaFace identity objective + 5 auxiliary losses. **No illusion stimuli in
training** (falsification purity).

**Pre-registered falsification (design §6)**: at the FFA layer, simultaneously
Thatcher ISI ≥ 3.0, Composite CSI ≥ 1.5, Part-Whole PWI ≤ 0.5, Random-bbox
ISI ≤ 1.5 — a combination no prior in the 25-prior battery achieves. Meeting it
would make HOLO-Net the first "paradigm-consistent" configural-processing model.

## 2. The experimental arc

| Exp | What | Outcome |
|---|---|---|
| E040 | minimal HOLO-Net (bio components OFF) trains | identity 13→1.5 ✓ |
| E041 | minimal ablation floor, 4 paradigms | FFA: ISI 0.98 / CSI 1.19 / PWI 0.78 — fails all (intended floor) |
| E042 | full HOLO-Net training (runs v1-v5) | 3 real bugs found+fixed; v5 trains identity to 0.97, gate functional |
| E043 | component-isolation ablation | superseded — premise (a "failure") was a misjudged normal loss bounce |
| **E044** | **v5 pre-registered falsification** | **ISI 1.00 / CSI 0.99 / PWI 2.20 → ALL FOUR: FAIL** |

Three genuine bugs were found and fixed along the way (AdaFace loss formula;
OrientationGate global-pool flip-blindness; FFA/ATL identity-collapse +
gate-detach). One process error: ticks 55-57 misread the normal ~8000-step
identity-loss bounce as failure and prematurely killed 3 runs — corrected at
tick 58 once the minimal reference trajectory was pulled.

## 3. The decisive result (E044) — v5 falsification, FFA layer (pixel-corrected)

| | v5 | criterion | minimal floor |
|---|---|---|---|
| Thatcher ISI | 1.00 | ≥ 3.0 ✗ | 0.98 |
| Composite CSI | 0.99 | ≥ 1.5 ✗ | 1.19 |
| Part-Whole PWI | 2.20 | ≤ 0.5 ✗ | 0.78 |
| Random-bbox ISI | 1.08 | ≤ 1.5 ✓ | 1.08 |

**HOLO-Net v1.0 is REFUTED** by its own pre-registered falsification.

## 4. Why HOLO-Net v1.0 failed — diagnosis

- **[CONFIRMED] Training objective, not architecture.** HOLO-Net is trained with
  a face-IDENTITY objective. Idea-001 E004 already showed identity-trained
  FaceNet gives ISI ≈ 1.0, while image-text contrastive training emerges ISI
  4-7. v5 — a maximally bio-faithful architecture, trained on identity — gives
  ISI ≈ 1.0, exactly like FaceNet. Architectural bio-fidelity does not
  substitute for the training objective.

- **[CONJECTURE] The design §7 Thatcher mechanism is conceptually flawed.** The
  Orientation Gate distinguishes WHOLE-FACE upright vs inverted. But Thatcher is
  a LOCAL-feature inversion within an upright face — to the gate, both
  upright-normal and upright-thatched are "upright". The gate provides no
  Thatcher-specific signal; the architecture has no mechanism that responds to
  local-feature-inversion-conditioned-on-orientation.

- **[CONFIRMED] The FFA module is not inert.** It drives Part-Whole PWI from
  mfp 0.75 → ffa 2.20 (minimal floor 0.78) — the largest single paradigm
  deviation HOLO-Net produces. The holistic-binding module genuinely
  restructures the part-whole representation; it just does so in the opposite
  direction to the (DINOv2-derived) ≤0.5 criterion.

## 5. What this means — the honest scientific value

The HOLO-Net arc is a **rigorous mechanistic negative result**: building a
maximally brain-faithful face architecture is *not sufficient* to produce
human-aligned configural-illusion sensitivity when trained on a face-identity
objective. This is not a project failure — it is a clean, falsifiable test that
came back negative, and it **converges with Idea-001 from the architecture
side**: configural-illusion emergence is a property of the training objective
(image-text contrastive), not of architecture or face-identity supervision.

## 6. The integrated story (if FLAG-002 option 1 — consolidate)

One coherent paper:

1. **IllusionBench-EEG** (Idea-001): a 3-paradigm holistic-face illusion
   benchmark; across 25 visual priors, a clean training-objective × paradigm
   dissociation — CLIP/image-text emerges Thatcher, DINOv2/SSL emerges
   composite + part-whole, face-identity training emerges nothing; EEG decoders
   inherit a uniform low-pass.
2. **The architecture test** (Idea-003 / HOLO-Net): "is the dissociation about
   architecture or objective?" We built the most bio-faithful face architecture
   we could (layer = brain region) and trained it on face identity. It still
   shows no Thatcher/composite illusion (ISI/CSI ≈ 1.0). → The dissociation is
   **objective-driven, not architecture-driven** — demonstrated constructively.
3. Contribution: the benchmark + the dissociation + a constructive negative
   that pins the cause to the training objective. HOLO-Net is a *supporting
   experiment*, not a standalone positive headline.

## 7. The redirect (if FLAG-002 option 2 — re-aim HOLO-Net at the objective)

The scientifically-motivated test: train the SAME HOLO-Net architecture with an
image-text contrastive objective (the lever Idea-001 identifies) instead of /
alongside AdaFace identity. If configural illusions then emerge, HOLO-Net
becomes a positive headline ("bio-faithful architecture + the right objective →
human-aligned configural processing"). Requires face image-text/caption data or
a CLIP-distillation target — HOLO-Net currently trains on Glint360K identity
labels only. Setup + train ≈ 1 day. Risk: it may also fail (the architecture
may not be the part that matters at all).

## 8. Honest assessment

- Idea-001 is the secure deliverable (score 9.3) — and is strengthened by the
  HOLO-Net result.
- Idea-003 as a *positive headline* (the pre-registered "first paradigm-
  consistent model") did not materialise — v1.0 is refuted (score 5.0).
- The work is not wasted: a rigorous negative result that pins causation to the
  training objective is genuine science and a real paper contribution.
- The honest framing: Idea-001 + HOLO-Net = one strong mechanistic paper. If a
  positive HOLO-Net headline is wanted, option 2 (objective re-aim) is the only
  scientifically-grounded route, and it is a fresh ~1-day experiment with real
  risk of also being negative.
