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
| HOLONET_v2_dinov2_afp | 0.908 | 1.282 | 0.397 | 0.662 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| HOLONET_v2_dinov2_atl | 0.908 | 1.282 | 0.397 | 0.662 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| HOLONET_v2_dinov2_ffa | 0.901 | 1.300 | 0.446 | 0.654 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| HOLONET_v2_dinov2_mfp | 0.908 | 1.282 | 0.397 | 0.662 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| HOLONET_v2_dinov2_v1 | 0.908 | 1.282 | 0.397 | 0.662 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| HOLONET_v2_dinov2_v2 | 0.908 | 1.282 | 0.397 | 0.662 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| HOLONET_v2_dinov2_v4 | 0.908 | 1.282 | 0.397 | 0.662 | 2/4 | ✗ | 1/4 | ✗ | -1 |

**Summary**: 0/7 models PASS under LEGACY; 0/7 models PASS under TANAKA_ALIGNED.


## Disagreements between regimes

| model_id | index | Legacy | Tanaka |
|---|---|:-:|:-:|
| HOLONET_v2_dinov2_afp | PWI | PASS | FAIL |
| HOLONET_v2_dinov2_atl | PWI | PASS | FAIL |
| HOLONET_v2_dinov2_ffa | PWI | PASS | FAIL |
| HOLONET_v2_dinov2_mfp | PWI | PASS | FAIL |
| HOLONET_v2_dinov2_v1 | PWI | PASS | FAIL |
| HOLONET_v2_dinov2_v2 | PWI | PASS | FAIL |
| HOLONET_v2_dinov2_v4 | PWI | PASS | FAIL |