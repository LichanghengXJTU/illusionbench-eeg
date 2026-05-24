# GPT-Image-1 pilot — medium quality

- **model**: gpt-image-1
- **quality / size**: medium / 1024x1024
- **N pilot**: 10 (10 demographic cells)
- **Total cost**: $0.3780
- **Total elapsed**: 168.7s (mean per image: 16.9s)
- **MediaPipe pass rate**: 0/10 (0%)
- **Mean IOD** (passing only): nan px
- **Mean tilt** (passing only): nan deg

## Per-image

| id | demo | elapsed (s) | cost | MP | IOD (px) | tilt (°) | notes |
|---:|---|---:|---:|:-:|---:|---:|---|
| 0 | young adult East Asian female | 0.0 | $0.0000 | FAIL | nan | nan | reused cached generation |
| 1 | middle-aged Black male | 18.8 | $0.0420 | FAIL | nan | nan | MediaPipe detection failed (no face) |
| 2 | young adult White male | 16.8 | $0.0420 | FAIL | nan | nan | MediaPipe detection failed (no face) |
| 3 | middle-aged South Asian female | 16.4 | $0.0420 | FAIL | nan | nan | MediaPipe detection failed (no face) |
| 4 | older adult Latinx male | 18.0 | $0.0420 | FAIL | nan | nan | MediaPipe detection failed (no face) |
| 5 | young adult White female | 17.9 | $0.0420 | FAIL | nan | nan | MediaPipe detection failed (no face) |
| 6 | middle-aged East Asian male | 26.7 | $0.0420 | FAIL | nan | nan | MediaPipe detection failed (no face) |
| 7 | older adult Black female | 21.3 | $0.0420 | FAIL | nan | nan | MediaPipe detection failed (no face) |
| 8 | young adult Middle Eastern male | 15.0 | $0.0420 | FAIL | nan | nan | MediaPipe detection failed (no face) |
| 9 | middle-aged Latinx female | 17.7 | $0.0420 | FAIL | nan | nan | MediaPipe detection failed (no face) |

## Budget projection

- N=100 → est. **$3.78**
- N=200 → est. **$7.56**
- N=300 → est. **$11.34**
- N=500 → est. **$18.90**
- N=1000 → est. **$37.80**
- N=3000 → est. **$113.40**

## Replicability

Prompt template + demographic grid are in `stimuli/gpt_image_pilot.py`. Per-identity JSON contains full prompt + API response metadata.