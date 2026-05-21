# E005 — Image-text training family comparison (Q007 discriminator)

**Hypothesis (Q007)**: Three candidate mechanisms for CLIP's face-Thatcher ISI:
  (a) **scale + diverse data**: any large-scale image model emerges it
  (b) **text-image alignment**: language-conditioning encodes upright-orientation priors
  (c) **CLIP-specific**: artifact of OpenAI/LAION CLIP objective

**Discriminating prediction**: Add other image-text contrastive models (SigLIP
with sigmoid loss, MetaCLIP with CLIP arch + different data). If all show
ISI 4-7 → (a) or (b). If only OpenAI/LAION CLIP shows it → (c).

**Method**: Same FFHQ Thatcher + random-bbox stimuli as E002 / E003. Add to
model zoo: P19 SigLIP-base-384, P20 SigLIP-SO400M-384, P21 MetaCLIP-H/14.
Same ISI_pert metric, same bootstrap.

**Pre-registered prediction** (set 2026-05-22 00:20):
- IF (c) CLIP-specific: SigLIP and MetaCLIP ISI ≈ 1-2
- IF (a) or (b): all three give ISI 3.5-7

**Result**:
```
Face Thatcher              Random-bbox
P19 SigLIP-base-384    3.513 [3.216, 3.844]   1.300 [1.159, 1.453]
P20 SigLIP-SO400M      5.588 [4.990, 6.247]   1.573 [1.416, 1.758]
P21 MetaCLIP-H/14      6.084 [5.291, 7.062]   1.757 [1.587, 1.955]

(reference: P06 CLIP-bigG14 = 6.763 face, 1.972 random-bbox)
```

**Interpretation** ([CONFIRMED] / [CONJECTURE]):

- [CONFIRMED] **Q007 (c) REFUTED**: SigLIP (different sigmoid loss) and
  MetaCLIP (same OpenAI CLIP arch but different data curation) both show
  ISI 3.5-6.1 on face Thatcher — consistent with the CLIP family. The
  Thatcher ISI signature is NOT a CLIP-specific artifact.

- [CONFIRMED] **Image-text contrastive training family broadly emerges
  face-Thatcher ISI** — across 8 models (5 CLIP + 2 SigLIP + 1 MetaCLIP)
  the face Thatcher ISI ranges 3.5-6.8 with confidence intervals.

- [CONFIRMED] All three new priors show comparable face-feature-specificity
  ratios: random-bbox ISI drops 50-75% relative to face Thatcher, same as
  the original CLIP family.

- [CONFIRMED] **Scale matters within image-text family**: SigLIP-base (3.5,
  smaller model) < SigLIP-SO400M (5.6, large) < MetaCLIP-H/14 (6.1, full
  data) ≈ CLIP-bigG (6.8). The scaling trend is consistent with the CLIP
  family's own scaling (b32 < L/14 < bigG/14 etc.).

- [CONJECTURE — refining Q007]: Now have a clearer picture of the training
  hierarchy:
    - No language + no scale: ISI ≈ 1 (pixel, untrained ViT, VAE, MAE)
    - No language + scale (DINOv2): ISI up to 3.3 (graduated emergence)
    - Language + scale (CLIP / SigLIP / MetaCLIP): ISI 3.5-6.8
    - Face-identity training: ISI ≈ 1 (active suppression)
  This is consistent with H_(a+b combined): SCALE + LANGUAGE both
  contribute to face-Thatcher emergence. DINOv2 base→giant covers
  scale-without-language; CLIP family + SigLIP + MetaCLIP cover
  language-with-scale; FaceNet shows what happens when scale is paired
  with pose-invariance objective.

- [CONJECTURE — IMPORTANT FOR IDEA-001]: The fact that ATM, AVDE, ENIGMA,
  HVF, ViEEG all anchor to the image-text-contrastive family means
  current EEG decoders inherit a representation with ISI ≈ 5-7 on face
  Thatcher. **The unanswered question (Q005)** is whether the EEG signal
  + EEG encoder + projection-to-CLIP pipeline preserves this signal or
  collapses it.

**Replicability**:
- Seed: 20260521
- New models: P19 (`google/siglip-base-patch16-384`), P20 (`google/siglip-so400m-patch14-384`), P21 (`facebook/metaclip-h14-fullcc2.5b`)
- NPZ paths: `outputs/embeddings/thatcher_ffhq/P19-P21.npz` + same in `thatcher_randombbox/`
- Manifest unchanged (same E002 / E003 stimuli)
- Output: `outputs/tables/thatcher_isi_ffhq.csv` (now 17 rows), `_randombbox.csv`

**Linked Q###**: **Q007 partially answered — family-general, not CLIP-specific.** Spawned refined Q008 (separate scale-from-language contribution). Q004 still next for tick 4.
