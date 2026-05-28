# `stimuli/classical_cv/` — three illusion-generation pipelines

All three paradigms (Thatcher / Part-Whole / Composite) are built with
**classical CV only** — dlib landmarks + MediaPipe FaceMesh (CPU delegate) +
InsightFace `buffalo_l` gender/age + OpenCV `seamlessClone` + Reinhard LAB
colour transfer. No DNN/generative model is used to synthesise any pixel.

Source faces: **FFHQ-26k** (CC BY-NC-SA 4.0). Output dataset:
[`Enhui-1/HoloFaceIllusion-Bench-EEG`](https://huggingface.co/datasets/Enhui-1/HoloFaceIllusion-Bench-EEG).

## File map

| File | Paradigm | Purpose |
|---|---|---|
| `thatcher_classical.py`              | Thatcher    | single-process reference impl (Thompson 1980) |
| `thatcher_parallel.py`               | Thatcher    | multiprocess driver over the FFHQ tar stream |
| `template_partwhole.py`              | Part-Whole  | single-process reference impl (Tanaka 2004) |
| `template_partwhole_parallel.py`     | Part-Whole  | multiprocess driver |
| `composite_v2_pilot.py`              | Composite   | single-process reference impl (Young 1987 / McKone 2013 / Murphy 2017) |
| `composite_v2_parallel.py`           | Composite   | multiprocess Phase A (attrs) + Phase B (generation) |
| `package_template_partwhole.py`      | Part-Whole  | flat PNG → webdataset tar shards |
| `package_composite_v2.py`            | Composite   | flat PNG → webdataset tar shards |
| `repack_thatcher.py`                 | Thatcher    | one-shot: rewrite v1.0 `NNNNN/X.ext` → v1.1 `NNNNN.X.ext` (webdataset-compliant) |
| `composite_v2_chain.sh`              | Composite   | server orchestration: wait for attrs.pkl → Phase B → C → D |
| `composite_v2_post_b.sh`             | Composite   | server orchestration: wait for Phase B DONE → Phase C → D |
| `upload_composite_v2_stall_retry.sh` | Composite   | server upload wrapper with stall-kill + retry |
| `upload_partwhole_v1.4.py`           | Part-Whole  | direct HF upload (v1.4 release) |
| `upload_hf.sh`                       | shared      | thin `hf upload` wrapper (sources `/workspace/.secrets/hf.env`) |
| `color_transfer.py`                  | shared      | Reinhard LAB transfer |
| `face_embedding.py`                  | shared      | dlib 68 landmarks + 128-d embedder |
| `donor_match.py`                     | shared      | base donor-filter dataclass + farthest-point sampler |
| `insight_attrs.py`                   | shared      | InsightFace gender/age wrapper (single-threaded ONNX) |
| `feature_warp.py`                    | shared      | TPS warp helpers |

## Core algorithms

### 1. Thatcher (`thatcher_classical.py`)

For each FFHQ identity:
- dlib 68 landmarks → eye + mouth bbox (eyes = pts 36–47 convex hull, mouth = pts 48–67 convex hull, brows stay put)
- rotate each region 180° **in place** around its centroid
- union-mask blend into the source face (eliminates ghost edges that broke earlier versions)
- output 4 conditions: `V1 upright_normal`, `V2 upright_thatched`, `V3 inverted_normal`, `V4 inverted_thatched`

### 2. Part-Whole (`template_partwhole.py`)

Tanaka 2004 template-based design:
- 408 frontal-face templates × ~15 target identities × 3 features (eye / nose / mouth) → 188,988 cases
- TPS-warp the **donor's** feature region to the **template's** landmark positions
- Poisson NORMAL_CLONE seam, L-channel ring-shift to preserve donor identity
- output 4 conditions: `V1 whole-face target`, `V2 whole-face with foil's feature`, `V3 feature on grey`, `V4 foil's feature on grey`

### 3. Composite (`composite_v2_pilot.py`)

Young 1987 / McKone 2013:
- cut line `y_cut = (lm[33].y + lm[51].y) / 2` (dlib nose-bottom ↔ upper-lip midpoint)
- pick K=3 donors per template by **farthest-point sampling** on dlib embeddings, after hard filters:
  - same gender (InsightFace), age Δ ≤ 15, skin-LAB Δ ≤ (10, 8, 8)
  - dlib embedding L2 ∈ [0.55, 0.85], cheek-width at cut ± 10%, cut-to-chin ± 10%, |yaw| ≤ 15°
- similarity transform (2-point eye anchor) — **no TPS frontalization** (made artifacts worse)
- per-row cheek-x alignment for bottom half
- L-channel-only ring colour shift in `cut ± 30 px` (preserves donor chroma identity)
- Poisson NORMAL_CLONE with eroded-4-px bottom-half mask
- MediaPipe FACE_OVAL (36 landmarks, **CPU delegate**) for precise face-boundary mask
- output 4 conditions: `V1 aligned upright`, `V2 misaligned (dx = 0.5 × face_w)`, `V3 aligned inverted`, `V4 misaligned inverted`

## Server pipeline (RunPod, H100 80GB / 224-core)

The composite generator is large (~21k templates × 3 donors × 6 files ≈ 61k cases / 100 GB). It runs in three phases driven by `composite_v2_parallel.py`:

```
Phase A  ─── 96-worker attrs extraction over 26,317 FFHQ identities (~10 min)
         ─── outputs data/composite_attrs.pkl  (~260 MB)

Phase B  ─── 32-worker per-template generation (~80 min @ 3.5 T/s)
         ─── outputs data/composite_v2_pngs/  (~100 GB, 61k JSON + 246k PNG)

Phase C  ─── package_composite_v2.py → 124 webdataset tars (~3 min)

Phase D  ─── hf upload v2.0/ to HF (~3-5 min @ 500 MB/s with HF_XET_HIGH_PERFORMANCE=1)
```

Orchestration scripts:

- `composite_v2_chain.sh` — runs A → B → C → D in one detached process (uses `setsid` + nohup so it survives SSH disconnects)
- `composite_v2_post_b.sh` — same as above but starts from Phase B already done (used when resuming after a crash)
- `upload_composite_v2_stall_retry.sh` — kill+retry wrapper for `hf upload` if no TX-bytes growth for 30 s (note: xet handshake can be silent for several seconds — set `STALL_SEC=120` if your bandwidth is low)

## Hard-earned performance lessons

| Symptom | Root cause | Fix |
|---|---|---|
| 96 workers stuck at 0 throughput | ONNX Runtime default = N-core thread pool per worker → 96 × ~225 = ~14k threads thrashing | monkey-patch `ort.InferenceSession.__init__` to inject `SessionOptions(intra_op_num_threads=1, inter_op_num_threads=1)` — see `insight_attrs.py` |
| `ImportError: cannot import name 'runtime_version' from google.protobuf` | mediapipe's `optional_dependencies.py` catches `ModuleNotFoundError`, not `ImportError`; protobuf 5 + TF 2.21 raise the latter | sed-patch the `except` clause to catch both |
| Mediapipe FaceLandmarker very slow with 96 workers | default delegate = GPU; 96 procs fight for one EGL context on the H100 | pass `BaseOptions(delegate=BaseOptions.Delegate.CPU)` when creating the landmarker |
| Phase B at 0.2 templates/s with 96 workers | IPC pool overhead dominates when per-task work < ~3 s; main process can't dispatch fast enough | drop to 32 workers → 3.5 T/s (17× speedup); 16 workers also works (2.8 T/s) |
| HF data viewer ❌ on Thatcher | tar layout was `NNNNN/X.ext` (per-file sample key); webdataset needs `NNNNN.X.ext` (per-identity sample key, fields = extensions) | one-shot `repack_thatcher.py` rewrites all 70 tars |
| `ls "$DIR"/*.json` returns 0 for 61k files | bash glob exceeds `ARG_MAX` and silently returns nothing | use `find "$DIR" -maxdepth 1 -name "*.json"` |
| `hf upload` silent for 30+ s during xet handshake | xet protocol negotiates upfront before streaming bytes | wrapper monitors **both** stdout growth AND `/sys/class/net/eth0/statistics/tx_bytes`; alive if either grows; lengthen `STALL_SEC` to ≥ 60 if your wrapper kept killing healthy uploads |

## Setup (one-time, server-side)

```bash
# venv (uv-managed)
cd /workspace/illusionbench-eeg
uv venv .venv && . .venv/bin/activate
uv pip install -r requirements.txt  # or: dlib insightface mediapipe huggingface_hub[cli] opencv-python pillow numpy scipy

# pin protobuf to 4.25 (TF & mediapipe both happy)
uv pip install "protobuf>=4.25,<5"

# patch mediapipe to catch ImportError (one line)
sed -i 's/except ModuleNotFoundError:/except (ModuleNotFoundError, ImportError):/' \
    .venv/lib/python3.12/site-packages/mediapipe/tasks/python/core/optional_dependencies.py

# models
mkdir -p models/mediapipe
ln -sf /workspace/models/shape_predictor_68_face_landmarks.dat       models/
ln -sf /workspace/models/dlib_face_recognition_resnet_model_v1.dat   models/
ln -sf /workspace/models/face_landmarker.task                        models/mediapipe/
# InsightFace buffalo_l auto-downloads on first use into ~/.insightface/models/

# HF token (mode 600, never commit)
mkdir -p /workspace/.secrets && chmod 700 /workspace/.secrets
printf 'export HF_TOKEN=hf_xxxxxxxxxxxx\n' > /workspace/.secrets/hf.env
chmod 600 /workspace/.secrets/hf.env
```

## Running the full composite pipeline

```bash
# Single command, all phases, detached from SSH
setsid bash stimuli/classical_cv/composite_v2_chain.sh \
    > logs/chain.log 2>&1 < /dev/null & disown
```

Phase B uses single-threaded BLAS to avoid the thread-thrash:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -u -m stimuli.classical_cv.composite_v2_parallel \
    --phase b --workers 32 --donors_per_template 3 \
    --tar_src data/HoloFaceIllusion-Bench-EEG-v1.1 \
    --attrs_pkl data/composite_attrs.pkl \
    --png_dir data/composite_v2_pngs
```
