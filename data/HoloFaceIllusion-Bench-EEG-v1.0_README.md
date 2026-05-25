---
license: cc-by-nc-sa-4.0
task_categories:
- image-classification
- image-to-image
language:
- en
tags:
- face
- vision
- psychophysics
- thatcher-illusion
- holistic-face-processing
- eeg
- benchmark
- bio-inspired
size_categories:
- 10K<n<100K
pretty_name: HoloFaceIllusion-Bench-EEG v1.0 (Thatcher)
---

# HoloFaceIllusion-Bench-EEG v1.0 (Thatcher)

A large-scale benchmark of **holistic-face illusion stimuli** built for testing
human-vs-DNN alignment on configural face processing and for training /
fine-tuning EEG-to-image decoders.

This v1.0 release contains the **Thatcher illusion** paradigm only.
Composite-face and Part-Whole paradigms will be released as v1.1 and v1.2.

## Quick stats

| | |
|---|---|
| **Paradigm** | Thatcher illusion (Thompson 1980, eye-only variant) |
| **Identities** | 26,317 |
| **Conditions per identity** | 4 (V1 upright_normal, V2 upright_thatched, V3 inverted_normal, V4 inverted_thatched) |
| **Total images** | 26,317 × 4 = 105,268 PNG (1024 × 1024) |
| **Tar shards** | 70 (webdataset format, ~1.9 GB each) |
| **Total size** | 134 GB |
| **Source** | FFHQ ([Karras et al. 2019](https://github.com/NVlabs/ffhq-dataset)) via `gaunernst/ffhq-1024-wds` |
| **Filtering pass rate** | 37.6% of 70k FFHQ images (see "Construction" below) |
| **Construction** | Pure classical CV — dlib HOG+SVM detection + 68-pt landmark + OpenCV Poisson `seamlessClone`; **no neural networks, no generative AI** |

## Quick start

```python
import webdataset as wds

url = "https://huggingface.co/datasets/<your-username>/HoloFaceIllusion-Bench-EEG/resolve/main/{00000..69000}.tar"
ds = (wds.WebDataset(url)
        .decode("rgb"))

for sample in ds:
    # sample is a dict; keys are file basenames
    v1 = sample["v1_upright_normal.png"]           # H×W×3 uint8 RGB
    v2 = sample["v2_upright_thatched.png"]
    v3 = sample["v3_inverted_normal.png"]
    v4 = sample["v4_inverted_thatched.png"]
    landmarks = sample["landmarks.json"]
    # ...
```

## Why this dataset exists

Holistic face processing — the brain's tendency to process faces as integrated
wholes rather than collections of features — is one of the most robust
behavioural signatures of human face perception. The **Thatcher illusion**
(Thompson 1980) is its canonical demonstration: an upright face with its
eyes rotated 180° looks bizarre at first glance; an inverted face with the
same local manipulation looks almost normal.

Despite the importance of the Thatcher effect, existing stimulus sets for
testing models against it are tiny:

| Source | N identities | Released? |
|---|---|---|
| Jacob et al. 2021 *Nat Comm* | 20 | Yes (OSF) |
| Dobs et al. 2023 *PNAS* | 40-100 per experiment | Yes (OSF) |
| Most psychophysics studies | 20-200 | typically not |
| **HoloFaceIllusion-Bench-EEG v1.0** | **26,317** | **Yes (this release)** |

This is **~1300× larger** than the largest published Thatcher stimulus
set. Each identity comes with a controlled 4-condition quartet — a
structure essential for representational similarity analysis (RSA),
within-subject EEG decoding, and any analysis where pairwise stimulus
comparison matters.

## Intended uses

1. **Benchmark / probing** for vision models: measure illusion sensitivity
   index (ISI) across model checkpoints; compare against human range
   (Carbon 2005: ISI ≈ 4-5 on similar paradigm).
2. **Fine-tuning** bio-inspired face models on holistic processing.
3. **EEG decoder evaluation**: paired with EEG recordings (e.g.,
   THINGS-EEG2), test whether the decoder preserves the Thatcher
   signature through the EEG bottleneck.
4. **Representational similarity analysis** — paired V1/V2 stimuli are
   ideal for comparing model and brain RDMs.

## Intentionally NOT for

- Training general-purpose face recognition from scratch (use VGGFace2 /
  MS1MV2 / WebFace260M for that — they have 100×+ more identities)
- Commercial applications (CC BY-NC-SA 4.0 — non-commercial only)
- Demographic / fairness-critical applications without further audit
  (FFHQ has known Western-Caucasian bias; we inherit it)

## Construction (pure classical CV, no AI/ML)

For each face in FFHQ-1024:

1. **Detect** with dlib HOG + SVM detector (Dalal 2005 + Cortes 1995, pre-DL classical CV)
2. **Locate** 68 facial landmarks via dlib Kazemi & Sullivan 2014 ensemble regression trees (classical ML, pre-DL)
3. **Filter** for benchmark suitability:
   - inter-ocular distance ≥ 120 px
   - eye-line tilt ≤ 15°
   - yaw score ≤ 0.18 (~ < 15° head yaw)
   - face bbox area ≥ 20% of frame (face is dominant element)
   - no glasses (Canny edge density ≤ 0.07 in eye region)
   - mouth not loudly open (inner-lip gap / mouth width ≤ 0.18)
4. **Build polygons** (Thompson 1980 eye-only variant): two eye-only convex hulls (NOT including brows) + one mouth polygon (outer lip dilated 7% IOD)
5. **For each region**:
   - Extract centroid-symmetric bbox patch
   - Rotate 180° around polygon centroid
   - `cv2.seamlessClone(NORMAL_CLONE)` (Pérez et al. 2003 Poisson editing) with mask = original ∪ rotated polygon, eliminating boundary seams
6. **Conditions**:
   - V1 = original
   - V2 = thatcherized
   - V3 = rot180(V1)  — for inverted-control comparison
   - V4 = rot180(V2)  — Thompson's "looks normal" condition

Pixel-baseline ISI = 1.000 by construction (V3 = rot180(V1), V4 = rot180(V2)).

The full reproducible pipeline is on the [companion repo](https://github.com/LichanghengXJTU/illusionbench-eeg) under `stimuli/classical_cv/`.

## Limitations

1. **FFHQ demographic bias** (Western, mostly Caucasian over-representation)
   carries through; we did NOT demographic-balance for this v1.
2. **Eye-only variant**: brows do NOT rotate (per Thompson 1980 canonical
   recipe). The brow-rotating variant produced "obviously inverted brows"
   that defeat the illusion's "looks normal at first glance" property.
3. **No human behavioural ratings per stimulus** — only Carbon 2005 group-
   average human ISI (≈ 4-5) as reference. Per-stimulus human validation
   is a follow-up.
4. **Single canonical paradigm**: Thatcher-only in v1.0. Composite + Part-
   Whole forthcoming.
5. **Mouth open / smile cases**: a smiling V2 has the smile inverted to a
   frown; this is sometimes more visually striking than the Thompson original
   (which used neutral expressions). We deliberately allowed smiles ("在野
   表情") for ecological validity but they may show stronger effect sizes
   than expected.

## License

**CC BY-NC-SA 4.0** — inherited from the FFHQ source license
([Karras et al. 2019](https://github.com/NVlabs/ffhq-dataset#license-and-citation)).
Academic / non-commercial use; share-alike if you redistribute modified
versions. Cite both this dataset and the original FFHQ release in any
derived work.

## Citation

```bibtex
@dataset{holofaceillusionbench_eeg_2026,
  title  = {HoloFaceIllusion-Bench-EEG v1.0 (Thatcher)},
  author = {En Hui Li (HKUST)},
  year   = {2026},
  url    = {https://huggingface.co/datasets/<username>/HoloFaceIllusion-Bench-EEG},
  note   = {CC BY-NC-SA 4.0; derivative of FFHQ (Karras et al. 2019)},
}
```

Plus required FFHQ citation:
```bibtex
@inproceedings{karras2019stylebased,
  title={A Style-Based Generator Architecture for Generative Adversarial Networks},
  author={Karras, Tero and Laine, Samuli and Aila, Timo},
  booktitle={CVPR},
  year={2019},
}
```

Plus the Thatcher illusion canonical reference:
```bibtex
@article{thompson1980thatcher,
  title={Margaret {T}hatcher: a new illusion},
  author={Thompson, Peter},
  journal={Perception},
  volume={9},
  number={4},
  pages={483-484},
  year={1980},
}
```

## Changelog

- **v1.0** (2026-05-25): Thatcher illusion only, 26,317 identities × 4 conditions.

## Contact

[GitHub Issues](https://github.com/LichanghengXJTU/illusionbench-eeg/issues) — feature requests, bug reports, or paradigm extension proposals.
