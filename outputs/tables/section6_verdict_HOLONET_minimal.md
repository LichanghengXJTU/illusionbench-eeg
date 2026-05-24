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
| HOLONET_minimal_afp | 0.975 | 1.192 | 0.779 | 1.083 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| HOLONET_minimal_atl | 0.975 | 1.195 | 0.770 | 1.081 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| HOLONET_minimal_ffa | 0.975 | 1.192 | 0.779 | 1.083 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| HOLONET_minimal_mfp | 0.965 | 1.073 | 0.725 | 1.155 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| HOLONET_minimal_v1 | 1.019 | 1.141 | 1.261 | 1.033 | 1/4 | ✗ | 2/4 | ✗ | +1 |
| HOLONET_minimal_v2 | 1.037 | 1.174 | 0.777 | 1.018 | 1/4 | ✗ | 1/4 | ✗ | 0 |
| HOLONET_minimal_v4 | 1.018 | 1.145 | 0.675 | 1.105 | 1/4 | ✗ | 1/4 | ✗ | 0 |

**Summary**: 0/7 models PASS under LEGACY; 0/7 models PASS under TANAKA_ALIGNED.


## Disagreements between regimes

| model_id | index | Legacy | Tanaka |
|---|---|:-:|:-:|
| HOLONET_minimal_v1 | PWI | FAIL | PASS |