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
| HOLONET_v3.0_dual_afp | 0.908 | 1.282 | 0.397 | 0.662 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| HOLONET_v3.0_dual_delta_inv | 0.633 | 1.259 | 0.369 | 0.448 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| HOLONET_v3.0_dual_delta_up | 0.904 | 1.295 | 0.456 | 0.658 | 2/4 | ✗ | 1/4 | ✗ | -1 |
| HOLONET_v3.0_dual_ffa | 0.753 | 1.275 | 0.408 | 0.541 | 2/4 | ✗ | 1/4 | ✗ | -1 |

**Summary**: 0/4 models PASS under LEGACY; 0/4 models PASS under TANAKA_ALIGNED.


## Disagreements between regimes

| model_id | index | Legacy | Tanaka |
|---|---|:-:|:-:|
| HOLONET_v3.0_dual_afp | PWI | PASS | FAIL |
| HOLONET_v3.0_dual_delta_inv | PWI | PASS | FAIL |
| HOLONET_v3.0_dual_delta_up | PWI | PASS | FAIL |
| HOLONET_v3.0_dual_ffa | PWI | PASS | FAIL |