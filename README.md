# IllusionBench-EEG

Stress-test benchmark for visual priors used by EEG-to-image decoders, on holistic
face illusion stimuli (Thatcher, composite, part-whole). Investigates whether
face-configural processing emerges in CLIP-class visual priors and whether the
signature survives EEG-conditioning.

**Status**: Phase 1 in progress. Working under autonomous research-loop (see
`research_journal/STATE.md` for live status). Live commits track every research
tick.

## Layout

```
illusionbench/
├── stimuli/        — Thatcher / composite / part-whole stimulus generators
├── models/         — visual prior loaders (CLIP, DINOv2, MAE, VAE, Harmonized, ...)
├── extract/        — sequential embedding extraction
├── analysis/       — ISI / d' / CLER metrics + statistical tests
├── attribution/    — causal masking / ROI attribution (planned)
├── train/          — H4 lite re-alignment (planned)
├── figures/        — publication figures
└── research_journal/
    ├── STATE.md
    ├── OPEN_QUESTIONS.md
    ├── IDEA_PIPELINE.md
    ├── EXPERIMENTS/
    ├── LITERATURE.md
    ├── DECISIONS.md
    └── DAILY_SUMMARY/
```

## Reproducibility

- Seed: `20260521` everywhere
- Compute: H100 80GB
- Stimulus source: FFHQ-1024 (`gaunernst/ffhq-1024-wds`)
- EEG datasets (planned): THINGS-EEG2, AllJoined-1.6M

## License

MIT
