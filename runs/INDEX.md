# Runs index

One line per experiment run. See each run dir for hypothesis.md / config /
results / summary.

| timestamp | tag | seed | status | one-line hypothesis | key result |
|---|---|---|---|---|---|
| 2026-05-23 ~17:25 HKT | holonet-v2-ftpc | 20260521 | KILLED (collapsed @ step 3200) | FTPC SSL on ImageNet-1K develops δ-layer for IllusionBench. | L_dino stuck at ln(out_dim)=11.09 (uniform collapse); L_pc collapsed to ~0.001; T_ema shrank 50→7. Diagnosed: PC loss had gradient through both AFP and T → trivial co-collapse. |
| 2026-05-23 ~18:00 HKT | holonet-v2-pc-detach | 20260521 | KILLED (collapsed @ step 3000) | Run 1 + AFP detached in PC loss (predictive-coding-correct). | L_pc fix worked (no full co-collapse), but L_dino STILL collapsed to 11.09 = uniform. Diagnosis: out_dim 65536 is DINOv2 default for ViT-L (~300M); our CORnet-S backbone (~25M) can't sustain 65536 prototypes. |
| 2026-05-23 ~18:15 HKT | holonet-v2-out4k | 20260521 | running (100 ep) | Run 2 with out_dim reduced 65536 → **4096** to match small backbone capacity. Uniform collapse baseline now ln(4096) = 8.32 (vs 11.09). | step 150: L_dino 8.46 (just above 8.32), L_pc 1.11 (held), T_ema 45 (stable), GPU 24.2G. Awaits descent past warmup. |
