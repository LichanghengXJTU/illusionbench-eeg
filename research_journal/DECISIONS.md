# Decisions log

- 2026-05-21 12:40 — server up: H100 80GB / 2TB RAM / 2TB NVMe / 224 CPU on RunPod.
- 2026-05-21 12:42 — install: uv venv + torch 2.5.1+cu124 (H100 compute cap 9.0 confirmed).
- 2026-05-21 12:43 — model downloads batch 1 (CLIP B/L/H + DINOv2 base/large + SDXL VAE).
- 2026-05-21 12:50 — vis-www.cs.umass.edu unreachable; switched LFW source to figshare URL.
- 2026-05-21 13:00 — first Thatcher generator works at LFW 224; user concern raised about V2/V4 visual subtlety.
- 2026-05-21 13:10 — added eyebrow landmarks to thatcherize() — visual effect strengthened.
- 2026-05-21 13:12 — E001 LFW ISI computed for 12 priors; CLIP cluster 3.5-4.1.
- 2026-05-21 22:00 — user red-team critique: ISI_pert ≠ behavioral d'; metric may be embedding-distance artifact; canonical Thatcher stimulus quality concern.
- 2026-05-21 22:35 — FFHQ 1024 visual QC: 8/8 sampled identities show canonical-looking grotesque V2.
- 2026-05-21 22:48 — E002 FFHQ ISI computed; CLIP-bigG ISI=6.76, sanity baselines remain 1.000.
- 2026-05-21 22:50 — user authorized autonomous research loop with: Max-tier budget (no cap), 70/30 exploit/explore, GitHub remote (PAT pending), rigor-strict (no fabrication, no unverified claims). Working dir moved to ~/Desktop/EEG/illusionbench/.
- 2026-05-21 22:55 — created SKILL `research-loop-eeg-illusion`. Journal scaffolding seeded (Q001-Q006, E001-E002, IDEA_PIPELINE Ideas 1-2, LITERATURE with 7+5 entries).
