# Runs index

One line per experiment run. See each run dir for hypothesis.md / config /
results / summary.

| timestamp | tag | seed | status | one-line hypothesis | key result |
|---|---|---|---|---|---|
| 2026-05-23 ~17:25 HKT | holonet-v2-ftpc | 20260521 | KILLED (collapsed @ step 3200) | FTPC SSL on ImageNet-1K develops δ-layer for IllusionBench. | L_dino stuck at ln(out_dim)=11.09 (uniform collapse); L_pc collapsed to ~0.001; T_ema shrank 50→7. Diagnosed: PC loss had gradient through both AFP and T → trivial co-collapse. |
| 2026-05-23 ~18:00 HKT | holonet-v2-pc-detach | 20260521 | running (100 ep) | Same as above, with AFP detached in PC loss (predictive-coding-correct: T learns to predict AFP, not the reverse). | step 150: L_dino 11.27 / L_pc 1.18 / T_ema 48 stable / template_norm 3.16 (T training healthily). Awaits L_dino descent past warmup. |
