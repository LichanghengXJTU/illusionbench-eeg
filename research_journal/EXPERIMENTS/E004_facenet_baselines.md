# E004 — Face-recognition-trained baselines (Q003 discriminator)

**Hypothesis (Q003)**: If the CLIP-class face-Thatcher ISI signal arises from
explicit face-recognition learning, then face-recognition-trained backbones
(FaceNet trained on VGGFace2 / CASIA-Webface, 3.3M-9.1M face identity pairs)
should show ISI ≥ CLIP-bigG (which is NOT explicitly face-trained).

**Discriminating prediction**: FaceNet ISI(face Thatcher) ≥ 5.0 → face-emergent.
FaceNet ISI ≤ CLIP-bigG → not solely from face training.

**Method**: Add facenet-pytorch `InceptionResnetV1` (Inception-Resnet-V1
architecture) with both pretrained weights (VGGFace2 and CASIA-Webface). Same
ISI_pert metric on the same 200 FFHQ identity Thatcher battery as E002.
Also run on the random-bbox stimuli from E003 for full prior-class comparison.

**Stimuli**: Same as E002 + E003 (200 FFHQ identities × 4 conditions).

**Models added**: P17_facenet_vggface2, P18_facenet_casiawebface (both 512-dim
L2-normalized identity embeddings, [-1,1] 160×160 input).

**Pre-registered prediction** (set 2026-05-22 00:00):
- IF Q003 hypothesis correct: FaceNet ISI(face Thatcher) ≈ 5-8
- IF Q003 hypothesis wrong: FaceNet ISI(face Thatcher) ≈ 1-3

**Result**:
```
                        Face Thatcher          Random-bbox
P17_facenet_vggface2     1.117 [1.042, 1.198]   0.382 [0.314, 0.466]
P18_facenet_casiawebface 1.026 [0.974, 1.080]   0.458 [0.395, 0.529]

(for reference)
P06_clip_bigG14          6.763 [6.276, 7.279]   1.972 [1.685, 2.265]
P09_dinov2_giant         3.338 [2.947, 3.784]   0.851 [0.768, 0.935]
N03_pixel                1.000 [1.000, 1.000]   1.000 [1.000, 1.000]
```

**Interpretation** ([CONFIRMED] / [CONJECTURE]):

- [CONFIRMED] **Q003 hypothesis REFUTED**. Both FaceNet variants show
  ISI ≈ 1.0-1.1 on face Thatcher, ~6× weaker than CLIP-bigG. Despite being
  explicitly trained on millions of face identity pairs, FaceNet does NOT
  show a face-orientation-dependent Thatcher signature.

- [CONFIRMED] FaceNet's d_up=0.38 and d_inv=0.34 on face Thatcher are BOTH
  large (10× larger than CLIP's d_up). FaceNet detects the local feature
  perturbation strongly in both orientations equally. The Thatcher illusion
  in models is NOT about whether the model notices the feature inversion —
  it's about whether the model treats it differently when the face is
  upright vs inverted.

- [CONFIRMED] On random-bbox, FaceNet ISI = 0.38-0.46 (substantially < 1):
  inverted random-bbox perturbations move the embedding MORE than upright.
  Together with d_up=0.011 (small response to non-feature perturbations),
  this is consistent with FaceNet attending strongly to face features
  regardless of orientation, and being weakly responsive to non-face regions.

- [CONJECTURE — IMPORTANT REFRAME] The Thatcher ISI signature is NOT a
  face-recognition feature. It is an **orientation-emergent face-configural
  property** that develops in scale-trained image-text contrastive models
  (CLIP) and weaker self-supervised image models (DINOv2) — but is **actively
  absent in face-identification-trained models**. The training objective
  matters: face-identification REQUIRES orientation/pose invariance for
  identity matching, so FaceNet learns to suppress orientation-dependent
  features. CLIP's text-image alignment does NOT require this invariance, so
  upright-face-statistics are preserved and amplified into orientation-dependent
  feature emphasis.

- [CONJECTURE — Q007 spawned] What specifically about CLIP-class training
  drives the orientation-emergence? Candidates: (a) scale + diverse data;
  (b) text-image alignment encoding upright-default expectations through
  language; (c) symmetric contrastive loss not penalizing orientation
  variance. SigLIP / EVA-CLIP / MetaCLIP comparison would discriminate
  (a) from (b)+(c).

**Replicability**:
- Seed: 20260521
- Models: P17/P18 via `facenet-pytorch==2.6.0`
- NPZ paths: `outputs/embeddings/thatcher_ffhq/P17_facenet_vggface2.npz`,
  `P18_facenet_casiawebface.npz`, and the analog in `thatcher_randombbox/`
- Manifest: same as E002 / E003
- Output: `outputs/tables/thatcher_isi_ffhq.csv`, `_randombbox.csv` (rerun with P17/P18 included)

**Linked Q###**: **Q003 ANSWERED (refuted)**, Q007 (new), Q004 (next).
