# E002 — FFHQ 1024 Thatcher ISI

**Hypothesis**: Higher stimulus quality (FFHQ 1024 vs LFW funneled 224) will reveal a
cleaner / stronger ISI hierarchy, and sanity baselines will remain at 1.000.
(Motivation: user observation that LFW V1/V2 are visually nearly identical; FFHQ 1024
produces canonical-looking Thatcher illusion per visual QC of 8 samples.)

**Method**: Same Thatcherize algorithm as E001, applied at FFHQ native 1024×1024.
MediaPipe scales padding (eye_pad=20, mouth_pad=25, blur_sigma=~5 at 1024). Save PNGs
at 1024; model processors auto-resize to their native input (typically 224).

**Stimuli**: 200 FFHQ identities (4 conditions each = 800 PNGs), resolution 1024×1024.
QC: MediaPipe confidence ≥ 0.7, IoD ≥ 120 px at 1024 native, eye-line tilt ≤ 15°.
Manifest `data/stimuli_ffhq/thatcher_manifest.csv` on server.

**Models**: Same 12 visual priors as E001.

**Metric**: Same as E001 (ISI_pert, bootstrap 1000× over identities).

**Pre-registered prediction** (set 2026-05-21 22:40):
- Sanity baselines must remain at ISI ≈ 1.000 (otherwise the metric is stimulus-dependent and broken)
- CLIP family ISI should increase relative to LFW (higher-quality stimulus → larger effect)
- DINOv2 family should also increase

**Result**:
```
N02_untrained_vit ISI=1.006 [0.979, 1.043]
N03_pixel         ISI=1.000 [1.000, 1.000]
P02_clip_b32      ISI=5.176 [4.684, 5.693]
P03_clip_l14      ISI=4.046 [3.681, 4.499]
P04_clip_h14      ISI=5.502 [5.112, 5.918]
P05_clip_g14      ISI=5.535 [5.120, 6.017]
P06_clip_bigG14   ISI=6.763 [6.276, 7.279]
P07_dinov2_base   ISI=1.369 [1.247, 1.505]
P08_dinov2_large  ISI=2.369 [2.114, 2.645]
P09_dinov2_giant  ISI=3.338 [2.947, 3.784]
P10_mae_huge      ISI=1.305 [1.224, 1.383]
P11_sdxl_vae      ISI=0.997 [0.990, 1.004]
```

**Interpretation** ([CONJECTURE] marked):
- [CONFIRMED] Sanity baselines still 1.000 → metric remains valid at higher resolution.
- [CONFIRMED] CLIP family rose: LFW 3.5-4.1 → FFHQ 4.0-6.8. DINOv2 base→giant 1.4→3.3.
- [CONFIRMED] Stimulus quality is a real factor: visual QC sheet `qc/ffhq_contact.png`
  shows all 8 sampled FFHQ V2 are visibly grotesque-Thatcher (validated by user
  inspection 2026-05-21 22:35).
- [CONJECTURE] CLIP-bigG ISI=6.76 exceeds Carbon 2005 human range (4-5). This is a 
  RAW measurement; it does NOT mean CLIP is "more human aligned" — Carbon's human ISI
  is from behavioral 2AFC d', not embedding distance, so quantitative comparison is
  not direct. Mark as: model exhibits orientation × Thatcher interaction stronger
  than embedding-baseline (1.0), magnitude exceeds rough human reference.
- [CONJECTURE — KEY UNCERTAINTY] The ISI signature could be: (a) face-configural
  processing emergent in CLIP, (b) general upright-orientation perturbation sensitivity
  (any local perturbation amplified upright), (c) representational geometry property
  (more embedding dimensions used for upright). These are NOT distinguished by
  current data; Q001 is the gating question.
- [NOTE] CLIP-L/14 (P03) is an outlier — barely moved LFW→FFHQ (3.9 → 4.0), while
  all LAION-CLIP variants nearly doubled. OpenAI's CLIP-L/14 may have a different
  training distribution that mutes the orientation effect.

**Replicability**:
- Seed: 20260521
- Stimulus code: `~/Desktop/EEG/illusionbench/stimuli/generate_thatcher_ffhq.py`
- NPZ paths: `/workspace/illusionbench-eeg/outputs/embeddings/thatcher_ffhq/*.npz`
- Manifest: `/workspace/illusionbench-eeg/data/stimuli_ffhq/thatcher_manifest.csv`
- Output: `/workspace/illusionbench-eeg/outputs/tables/thatcher_isi_ffhq.csv`
- FFHQ tar shard: `gaunernst/ffhq-1024-wds` 00000.tar

**Linked Q###**: Q001 (face-specificity), Q002 (verification-task sanity), Q006
