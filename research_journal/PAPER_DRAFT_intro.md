# Introduction — draft v1 (tick 11)

**Length target**: ~1.5 pages = ~700 words. Current word count: see end of file.

---

Holistic processing has been understood for nearly half a century as a load-bearing
constraint on human face perception. The classical Thatcher illusion (Thompson, 1980),
the composite-face effect (Young, Hellawell, & Hay, 1987), and the part-whole effect
(Tanaka & Sengco, 1997) all show that humans do not perceive faces as bags of local
features but as holistic configurations integrated across the face. These three
paradigms are widely used to probe configural processing because they have a
shared diagnostic structure: a manipulation that is small or invisible at the
pixel level is large or distinctive at the perceptual level only under
configural-binding conditions (upright, aligned, whole-face context). When that
binding is disrupted (by inversion, misalignment, or feature-isolation), the
perceptual difference collapses. Decades of psychophysics and ERP work (Rossion,
2013; Carbon et al., 2005; Murphy & Cook, 2017) have established these as core
benchmarks for any candidate model of human face perception.

Modern deep neural networks have produced remarkable results on natural-image
benchmarks, and a growing literature now uses such DNNs as candidate models of
biological vision (Schrimpf et al., 2018; Conwell et al., 2024). Within the
last two years a new application has emerged: large-scale image-text contrastive
models — most prominently CLIP (Radford et al., 2021) and its successors at
scale — have become the universal visual anchor for EEG-to-image decoding
(ATM, Li et al. 2024 NeurIPS; AVDE, ICLR 2026; ENIGMA, NeurIPS 2025; HVF, ICLR 2026;
ViEEG, Liu et al. 2025). These pipelines learn to project EEG signal into the
CLIP-embedding space of natural images and use diffusion or VQ priors to
reconstruct the seen image. The field has rapidly converged on this single
architectural class without systematically asking the question that motivates
the present work: **do the visual priors used by these EEG decoders inherit
human-like holistic face processing, and if so, does the EEG signal preserve
that signature through the bottleneck of EEG-to-CLIP projection?**

The closest prior work approaches each half of this question only partially.
Jacob, Aggarwal, et al. (2021) tested face-identification-trained DNNs (the
VGG-Face lineage) on a battery of nine psychological phenomena including the
Thatcher illusion. They found partial reproduction in upright-face-trained
networks. Phillips & White's 2026 review in the *British Journal of Psychology*
surveys the state of deep-learning face-processing models, focusing on identity
recognition; it does not cover CLIP-class image-text contrastive models or EEG
decoders. No work to our knowledge measures whether the dominant image-text
contrastive priors of 2024-2026 exhibit any of the three classical holistic
face illusions, nor whether the EEG bottleneck preserves the resulting
representation when these priors are used as the decoding target.

**We present IllusionBench-EEG**, a three-paradigm benchmark constructed at
FFHQ-1024 native resolution that systematically probes holistic face processing
in 17 modern visual priors (CLIP variants, SigLIP, MetaCLIP, DINOv2 family,
MAE, SDXL VAE, FaceNet face-identity backbones, and untrained controls),
combined with a per-CLIP-dimension EEG-preservation analysis on ATM's
pre-computed embeddings for all 10 THINGS-EEG2 subjects. The benchmark
implements canonical Thatcher (Thompson 1980), composite-face (Murphy & Cook
2017 setup), and part-whole (Tanaka & Sengco 1997 setup) stimulus designs
with MediaPipe landmark-based geometry. Each paradigm produces an
embedding-space sensitivity index (ISI / CSI / PWI) with bootstrap confidence
intervals; pixel-baseline sanity is verified at exactly 1.000 for Thatcher.

Our central findings are threefold. First, a clean **training-objective × paradigm
dissociation** emerges: image-text contrastive priors (CLIP / SigLIP / MetaCLIP)
dominate the Thatcher illusion with ISI 4-7 — within or above human-reference
range — while DINOv2 self-supervised priors dominate composite-face and part-
whole context binding. Second, **face-identification-trained networks (FaceNet
VGGFace2 and CASIA-Webface variants) show no holistic illusion effect on any
paradigm**, despite being explicitly trained on millions of face identity pairs.
The pose-invariance objective intrinsic to identity recognition actively
suppresses the orientation-dependent configural features that other training
paradigms emerge. Third, **direct analysis of ATM's pre-computed EEG embeddings
on all 10 THINGS-EEG2 subjects** reveals that the EEG bottleneck imposes a
uniform per-dimension attenuation (mean Pearson r ≈ 0.158 across 1024 CLIP-H/14
dimensions) with no dimension-specific structure: Thatcher-loaded dimensions are
neither selectively preserved nor selectively destroyed, and face-related
categories receive no privileged preservation. Combined, these findings imply
that current EEG visual decoders cannot reproduce human-aligned holistic
processing on any of the three paradigms, regardless of which paradigm's signal
is on their image side.

We make five specific contributions: (1) **IllusionBench-EEG**, an open
three-paradigm stimulus battery (Thatcher / composite / part-whole) generated
deterministically from FFHQ-1024 with pixel-baseline-verified metrics; (2)
**Training-objective × paradigm dissociation** — the first systematic mapping
showing different model classes capture different aspects of face-configural
processing; (3) **Empirical refutation of "face-trained DNNs are most face-aligned"**:
across all three paradigms, FaceNet performs at baseline while large-scale
image-text contrastive and self-supervised global models exhibit substantial
illusion sensitivity; (4) **A per-CLIP-dimension EEG preservation analysis** on
ATM showing the EEG bottleneck is a non-specific low-pass that cannot
selectively transmit holistic perceptual information; (5) **An architectural
prediction** for human-aligned EEG visual decoding: dimension-fine perceptual
structure preservation (not single CLIP-cluster anchoring) is required.

The paper is organized as follows. Section 2 reviews related work. Section 3
describes the IllusionBench-EEG stimuli and metrics. Section 4 presents the
image-side three-paradigm dissociation results. Section 5 presents the EEG-side
per-CLIP-dim preservation analysis. Section 6 discusses architectural
implications and limitations.

---

**Word count check**: approximately 770 words. Slightly over target 700; can
trim during revision. Key claims appear in correct logical order: hook → bridge →
gap → approach → findings → contributions → roadmap. Ready for review against
the rest of the outline.
