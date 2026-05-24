# §6 Verdict — Legacy vs Tanaka-aligned thresholds

## Threshold regimes


### LEGACY

- **ISI** >= 3.0  — *below Carbon 2005 human ISI 4-5; defensible*
- **CSI** >= 1.5  — *arbitrary; no published human anchor*
- **PWI** <= 0.5  — *OPPOSITE to Tanaka 1997 direction; rewards anti-holistic models*
- **ISIrbox** <= 1.5  — *control: ISI on non-feature regions; OK*

### TANAKA_ALIGNED

- **ISI** >= 3.0  — *below Carbon 2005 human ISI 4-5; defensible*
- **CSI** >= 1.5  — *Young 1987 direction (aligned > misaligned discrimination); no human ratio benchmark — kept matched to legacy for direct comparison*
- **PWI** >= 1.0  — *Tanaka 1997 direction: whole-context > part-isolated discrimination implies holistic binding; PWI > 1 = matches human direction*
- **ISIrbox** <= 1.5  — *control: same as legacy*

## Per-model verdict

| model_id | ISI | CSI | PWI | ISIrbox | Legacy n/4 | Legacy 4/4 | Tanaka n/4 | Tanaka 4/4 | Δ |
|---|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|
| N02_untrained_vit | 1.006 | 1.029 | 0.646 | 0.994 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| N03_pixel | 1.000 | 1.000 | 1.000 | 1.000 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| P02_clip_b32 | 5.176 | 1.123 | 0.714 | 2.116 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| P03_clip_l14 | 4.046 | 1.596 | 0.740 | 1.332 | 3/4 | ✗ | 3/4 | ✗ | 0 |
| P04_clip_h14 | 5.502 | 1.146 | 0.569 | 1.441 | 2/4 | ✗ | 2/4 | ✗ | 0 |
| P05_clip_g14 | 5.535 | 1.141 | 0.641 | 1.613 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| P06_clip_bigG14 | 6.763 | 1.206 | 0.586 | 1.972 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| P07_dinov2_base | 1.369 | 1.224 | 0.255 | 1.072 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| P08_dinov2_large | 2.369 | 1.408 | 0.329 | 0.888 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| P09_dinov2_giant | 3.338 | 1.525 | 0.293 | 0.851 | 4/4 | ✓ | 3/4 | ✗ | -1 |
| P10_mae_huge | 1.305 | 1.207 | 0.579 | 0.993 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| P11_sdxl_vae | 0.997 | 1.058 | 0.363 | 0.995 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| P17_facenet_vggface2 | 1.117 | 1.011 | 0.591 | 0.382 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| P18_facenet_casiawebface | 1.026 | 1.050 | 0.793 | 0.458 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| P19_siglip_base_384 | 3.513 | 1.282 | 0.561 | 1.300 | 2/4 | ✗ | 2/4 | ✗ | 0 |
| P20_siglip_so400m | 5.588 | 1.426 | 0.787 | 1.573 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| P21_metaclip_h14 | 6.084 | 1.564 | 0.601 | 1.757 | 2/4 | ✗ | 2/4 | ✗ | 0 |
| P22_cornet_s | 0.991 | 1.048 | 0.178 | 0.938 | 2/4 | ✗ | 1/4 | ✗ | -1 |

**Summary**: 1/18 models PASS under LEGACY; 0/18 models PASS under TANAKA_ALIGNED.


## Disagreements between regimes

| model_id | index | Legacy | Tanaka |
|---|---|:-:|:-:|
| P07_dinov2_base | PWI | PASS | FAIL |
| P08_dinov2_large | PWI | PASS | FAIL |
| P09_dinov2_giant | PWI | PASS | FAIL |
| P11_sdxl_vae | PWI | PASS | FAIL |
| P22_cornet_s | PWI | PASS | FAIL |