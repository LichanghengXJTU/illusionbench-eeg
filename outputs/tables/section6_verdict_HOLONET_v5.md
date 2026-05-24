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
| HOLONET_v5_afp | 1.018 | 0.967 | 2.068 | 1.073 | 1/4 | ✗ | 2/4 | ✗ | +1 |
| HOLONET_v5_atl | 0.996 | 1.103 | 0.813 | 1.100 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| HOLONET_v5_ffa | 1.002 | 0.991 | 2.203 | 1.076 | 1/4 | ✗ | 2/4 | ✗ | +1 |
| HOLONET_v5_mfp | 0.993 | 1.085 | 0.752 | 1.115 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| HOLONET_v5_v1 | 1.020 | 1.147 | 1.261 | 1.017 | 1/4 | ✗ | 2/4 | ✗ | +1 |
| HOLONET_v5_v2 | 1.036 | 1.182 | 0.741 | 0.998 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| HOLONET_v5_v4 | 0.997 | 1.127 | 0.747 | 1.025 | 1/4 | ✗ | 1/4 | ✗ | 0 |

**Summary**: 0/7 models PASS under LEGACY; 0/7 models PASS under TANAKA_ALIGNED.


## Disagreements between regimes

| model_id | index | Legacy | Tanaka |
|---|---|:-:|:-:|
| HOLONET_v5_afp | PWI | FAIL | PASS |
| HOLONET_v5_ffa | PWI | FAIL | PASS |
| HOLONET_v5_v1 | PWI | FAIL | PASS |