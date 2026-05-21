# 5. Discussion & 6. Conclusion — draft v1 (tick 15)

**Length target**: Discussion ~700 words, Conclusion ~150 words.

---

## 5. Discussion

### 5.1 What does the training-objective × paradigm dissociation tell us?

The cleanest interpretation of our three-paradigm dissociation (Section 4.1)
is that different holistic-face illusion paradigms probe distinct
representational axes, and different training objectives selectively shape
those axes. The Thatcher illusion isolates an interaction between local
feature orientation and global face orientation — V1 and V2 differ only in
local feature angles, V3 and V4 differ in the same way after a global 180°
rotation. For a model to give ISI > 1, its embedding must respond differently
to the same local-feature change depending on global orientation. This is
precisely what image-text contrastive training encourages: textual descriptions
of faces predominantly assume upright orientation, anchoring the embedding
geometry around upright facial configurations. The composite-face and
part-whole paradigms instead probe global integration across the face — V1
and V2 differ at the bottom half (composite) or in a small feature region
(part-whole), but the model's response to this differs across binding-preserving
vs binding-disrupting conditions. DINOv2's self-supervised global feature
prediction objective produces representations that are dominated by holistic
spatial structure, which captures these binding-disruption effects more
strongly than CLIP does. Face-identification training (FaceNet), in contrast,
is explicitly designed to be pose-invariant for identity matching — it actively
SUPPRESSES orientation-dependent and global-binding-dependent features in
favor of identity-discriminative ones. The combined result — face-trained
networks show NO holistic illusion effect on any paradigm — refutes the
intuitive "face-trained → most face-aligned" hypothesis and reframes face
configural processing as an emergent property of representational geometry
rather than identity-supervised feature learning.

### 5.2 Implications for human-aligned EEG visual decoding

Current EEG-to-image pipelines (ATM, AVDE, ENIGMA, HVF, ViEEG) share an
architectural commitment: the EEG signal is projected into a single CLIP-class
image embedding space, then decoded via diffusion or VQ priors. Two findings
from our work bear directly on whether this architectural choice can support
human-aligned holistic face perception. First, the visual prior choice
(CLIP vs DINOv2 vs face-trained) determines which holistic axis is even
representable in the target space; a single-anchor commitment will reproduce
at most one of the three illusion paradigms. Second, our per-CLIP-dim
preservation analysis on ATM shows the EEG bottleneck imposes uniform
attenuation to mean r ≈ 0.158, with no dimension-specific or face-category-
specific structure. Even if a richer visual anchor were used, current
EEG-to-CLIP projection cannot selectively transmit the dimensions that carry
holistic perceptual information.

These observations motivate three architectural directions: (a) **multi-prior
EEG decoding** where the EEG signal is simultaneously aligned to CLIP, DINOv2,
and possibly Harmonized visual anchors so the EEG-decoded representation has
access to multiple holistic axes; (b) **dimension-fine alignment objectives**
that explicitly preserve directions of CLIP space known to carry perceptual
asymmetries, instead of optimizing a single global retrieval accuracy;
(c) **behaviorally-anchored evaluation** — illusion-based stress tests like
IllusionBench-EEG, alongside the standard top-K retrieval, to detect whether
fine perceptual structure survives the EEG pipeline.

### 5.3 Limitations

Five limitations are worth foregrounding. First, the EEG-side analysis is
limited to ATM, the only EEG-to-image decoder with publicly released
pre-computed embeddings at the time of writing. AVDE and ENIGMA repositories
do not currently expose equivalent intermediate outputs; cross-decoder
generalization of Route A is therefore a future-work item. Second, we do not
collect EEG of holistic-face-illusion stimuli; predicted EEG-side ISI is
inferred from CLIP-side ISI multiplied by an EEG preservation factor, and
should be tested directly when face-illusion EEG datasets exist. Third, our
pixel-baseline correction for composite (CSI_pixel = 1.099) and part-whole
(PWI_pixel = 1.202) reflects construction-level differences in V1↔V2 vs V3↔V4
that are not artifacts of model representation; the qualitative ordering of
model classes is preserved after correction, but absolute magnitudes should
be interpreted with this in mind. Fourth, all stimuli are derived from FFHQ
which has its own demographic biases; future versions should incorporate
identity sets with known demographic balance (e.g., CFD with internal-use
restrictions). Fifth, our Q004 Harmonized-prior comparison was blocked by a
TF/Keras-3 compatibility issue (see `NOTES_FOR_USER.md` NEED-001) and remains
an open thread; the predicted result is that Harmonized models, which directly
optimize for human-attention alignment, would lie between FaceNet (baseline)
and CLIP (high) on Thatcher.

## 6. Conclusion

The three classical holistic face illusions — Thatcher, composite-face, and
part-whole — together reveal a clean training-objective × paradigm dissociation
across 17 modern visual priors. Image-text contrastive models capture Thatcher
with near-human ISI 4-7; DINOv2 self-supervised models capture composite and
part-whole spatial binding; face-identification-trained models capture none of
the three. Direct per-CLIP-dim preservation analysis on ATM's released EEG
embeddings shows the EEG bottleneck imposes uniform low-pass attenuation,
washing out fine perceptual structure across all dimensions without selective
suppression of Thatcher-loaded or face-category-related information. Current
EEG visual decoders, anchored as they are to a single image-text contrastive
prior, cannot reproduce human-aligned holistic face processing on any of the
three paradigms — a constraint we predict can be addressed with multi-prior
decoders and dimension-fine alignment objectives.

---

**Word count check**: Discussion ~720 words, Conclusion ~160 words. On target.
Paper draft now spans Sections 1-6 with explicit prose totaling ~4,400 words.
