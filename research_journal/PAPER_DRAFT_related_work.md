# 2. Related Work — draft v1 (tick 12)

**Length target**: ~1 page = ~500 words. Current word count: see end.

---

## 2.1 Holistic face processing in humans

The Thatcher illusion (Thompson, 1980) is the prototypical demonstration of
orientation-dependent face configural processing: an upright face whose eyes
and mouth have been independently inverted appears strikingly grotesque,
whereas the same modification on an inverted face is perceptually barely
noticeable. Carbon et al. (2005) quantified this asymmetry as a 4-5× ratio
in upright versus inverted discrimination d′, and showed via ERP that the
modification is detected in early N170-class responses regardless of
orientation while only later conscious-report processes give rise to the
behavioral asymmetry. The composite-face effect (Young, Hellawell, & Hay,
1987; Murphy & Cook, 2017) follows the same logical structure: integrating
top and bottom halves from different identities produces a percept of a
new whole that is hard to mentally segment when the halves are aligned,
but easy when they are spatially misaligned. The part-whole effect (Tanaka
& Sengco, 1997) shows that target features (eyes, mouth) are recognized
more accurately when presented in a whole-face context than in isolation.
Rossion (2013) reviews these three paradigms as a converging triad
diagnostic of configural / holistic face processing. They share a key
diagnostic property: a manipulation invisible at pixel level is large at
perceptual level only under specific configural-binding conditions.

## 2.2 Deep-learning face models and human alignment

Jacob, Aggarwal et al. (2021, *Nature Communications*) tested whether nine
psychological phenomena, including the Thatcher illusion, reproduce in face-
identification-trained DNNs (the VGG-Face lineage). They reported partial
reproduction in upright-face-trained networks but did not test image-text
contrastive models (CLIP did not exist at that time). Phillips & White (2026,
*British Journal of Psychology*) review the state of deep-learning models of
face processing in humans, focusing on identity-recognition DCNNs and their
correspondence to psychological models such as Bruce & Young's (1986) parallel-
route framework. Their review explicitly does NOT cover CLIP-class image-text
contrastive models, illusion-sensitivity metrics, or EEG visual decoding —
leaving an open frontier the present work directly addresses.

Adjacent recent work has examined CLIP's perceptual properties under different
framings: texture-vs-shape bias evolution during training (arXiv 2508.09814);
configural inversion in object recognition (Wagemans et al., 2020). To our
knowledge no published work measures CLIP-class face-illusion sensitivity at the
scale (up to bigG/14 = 2.5B parameters) and breadth (Thatcher + composite +
part-whole) we report here.

## 2.3 EEG-to-image visual decoding

A series of recent decoders project EEG signals into image-aligned feature
spaces. ATM (Li et al., 2024, NeurIPS) uses CLIP-ViT-H/14 LAION-2B as its
image-target space, with a custom EEG encoder and a diffusion-based image
generator. AVDE (ICLR 2026) builds on ATM, replacing diffusion with an
autoregressive VQ-VAE next-scale predictor while retaining the CLIP-H/14
anchor. ENIGMA (NeurIPS 2025) is a lightweight multi-subject variant that uses
under 1% of the parameters but the same CLIP-class semantic prior. HVF (ICLR
2026) and ViEEG (Liu et al., 2025) fuse multiple CLIP encoders with VAE latents,
again anchored in image-text contrastive space. Across this entire literature,
no work has examined whether the chosen visual anchor exhibits holistic face
illusions, nor whether the EEG bottleneck preserves whatever signal exists in
that anchor. Our per-CLIP-dim preservation analysis on ATM's released
pre-computed embeddings provides the first direct evidence.

## 2.4 Human-alignment metrics for vision DNNs

Established alignment metrics span behavioral consistency (Geirhos et al., 2020),
neural prediction (Brain-Score, Schrimpf et al., 2018; THINGS-similarity,
Hebart et al., 2020), saliency / attention alignment (the neural harmonizer of
Fel et al., 2022 NeurIPS), and shape-vs-texture bias (Geirhos et al., 2019
ICLR). These metrics generally evaluate alignment on natural images; the
present work contributes ISI/CSI/PWI as illusion-specific embedding-distance
metrics tailored to holistic face processing, alongside a per-CLIP-dim EEG
preservation analysis that is, to our knowledge, the first metric of its kind
for EEG-to-image decoders.

---

**Word count check**: approximately 590 words. Target was 500 but the field
coverage is dense; can trim during revision (especially 2.1's CARBON 2005 ERP
detail and 2.4's general-metrics enumeration).
