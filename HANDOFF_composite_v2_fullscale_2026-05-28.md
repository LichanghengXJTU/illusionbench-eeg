# HANDOFF: Composite v2.0 Full-Scale Generation + Server Upload
**Date**: 2026-05-28
**Prepared by**: Claude (previous session)
**For**: Next Claude agent picking up the composite v2.0 release

---

## 0. TL;DR

User wants the Composite face-illusion subset of `Enhui-1/HoloFaceIllusion-Bench-EEG` at the same scale as Thatcher (26,317 identities × 4 = 105,268 PNG, ~134 GB) and Part-Whole (188,988 samples × 4 = 755,952 PNG, ~134 GB).

A **2,715-case pilot** (4.3 GB, 6 webdataset tars) has been **fully generated** locally on the user's Mac. It was NOT uploaded — the home network has only ~250 KB/s upload bandwidth, making a 4.3 GB upload take 4–8 hours. The user has asked to **abandon the local upload** and instead:

1. Move the entire pipeline to the user's **server** (where PW v1.4 was generated — has fast fiber).
2. On the server: scale up from 2,715 cases → **~15–20k templates × 3 donors ≈ 45–60k cases** (similar scale to Thatcher / PW), since the server has all 26k FFHQ identities locally and gigabit upload to HF.
3. Upload to HF using `HF_XET_HIGH_PERFORMANCE=1` (which gives 6+ MB/s vs default ~250 KB/s — found this evening, see §6).

**The algorithm is locked. The code is written and tested. The only remaining work is execution on the server.**

---

## 1. Project Status (what's already on HF)

Repo: `Enhui-1/HoloFaceIllusion-Bench-EEG` (CC BY-NC-SA 4.0)

Currently uploaded:
- `README.md` — updated with **composite** section + 3-paradigm table + Young 1987 / McKone 2013 / Murphy 2017 citations
- `preview_composite_1.png`, `preview_composite_2.png` — 2 sample composite grids (from pilot)
- `preview_thatcher_1/2.png`, `preview_partwhole_1/2.png` — existing
- `*.tar` (70 Thatcher tars) — existing
- `v1.4/partwhole_*.tar` (469 PW tars) — existing
- `v2.0/manifest.csv` — composite v2.0 manifest (2,715 rows from pilot)
- **`v2.0/composite_*.tar` — NOT uploaded** (this is what you generate + upload)

The README already has the `composite` config declared:
```yaml
- config_name: composite
  data_files:
  - split: train
    path: "v2.0/composite_*.tar"
```

→ Once you upload the tars, `load_dataset("Enhui-1/HoloFaceIllusion-Bench-EEG", name="composite", streaming=True)` will work automatically.

---

## 2. Algorithm (LOCKED — do not change)

All design decisions are documented in `~/.claude/projects/-Users-enhuili-Desktop-EEG/memory/project_composite_design_2026-05-27.md`. The canonical source paper is Young, Hellawell & Hay (1987) "Configurational information in face perception" (Perception 16:747–759), as refined by McKone et al. (2013) and Murphy, Gray & Cook (2017).

### 2.1 Pipeline (per template T, per donor D)

```
1. Load template T's V1 (original FFHQ face) + dlib 68 landmarks + MediaPipe 478 landmarks
2. Filter T at LOAD time:
     - lwr_upr_ratio >= 3.85          (infant filter)
     - |yaw| <= 15°                   (frontal head)
     - mouth_offset / face_w <= 0.03  (mouth on midline)
3. For each T, pick K=3 donors D via pick_donors() applying ALL of:
     - HARD same-gender (InsightFace genderage)
     - |age diff| <= 15
     - skin LAB Δ <= (10, 8, 8)
     - dlib embed L2 ∈ [0.55, 0.85]
     - cheek_width at y_cut: |T - D| / max(T, D) <= 0.10
     - cut_to_chin ratio: |T - D| / max(T, D) <= 0.10
     - |yaw diff| <= 10°
   farthest-point sampling on embeddings so 3 donors are perceptually distinct.
4. For each (T, D):
     a. Similarity transform (2-point eye-anchor) donor → template frame
        (NO TPS-warp, NO frontalization — both made artifacts worse per pilot)
     b. Per-row horizontal warp aligning donor cheek-x to template cheek-x for y ∈ [y_cut, h]
        (cheek_align_warp_bottom function)
     c. L-channel-only ring color shift: match donor's L mean in cut±30px ring to template's
        (preserves donor chroma identity; PW v1.4 lesson — full Reinhard kills donor identity)
     d. Poisson NORMAL_CLONE with bot_mask = entire bottom half (eroded 4px from edges)
        (NORMAL_CLONE not MIXED_CLONE — PW v1.2 lesson)
     e. Apply MediaPipe FACE_OVAL mask (36 landmarks) — precise face boundary
        (NOT dlib jaw + estimated forehead arc — that had "split face" bug)
     f. Generate 4 conditions:
          V1 aligned upright (canonical Young 1987 illusion)
          V2 misaligned upright (bottom shifted by 0.5 × face_w)
          V3 aligned inverted (V1 rot180)
          V4 misaligned inverted (V2 rot180)
     g. Save 4 PNG + 1 JSON to flat dir
5. Pack flat dir into webdataset tars (~500 pairs per shard, ~850MB each)
```

### 2.2 Cut line and misalignment

- **y_cut = (lm[33].y + lm[51].y) / 2** (dlib: 33 = nose bottom, 51 = upper lip center). Canonical Young 1987 / McKone 2013 cut.
- **misalignment dx = 0.5 × face_w** where face_w = lm[16].x − lm[0].x.

### 2.3 Outputs

- `NNNNN__dMMMMM.v1.png` (aligned upright)
- `NNNNN__dMMMMM.v2.png` (misaligned upright)
- `NNNNN__dMMMMM.v3.png` (aligned inverted)
- `NNNNN__dMMMMM.v4.png` (misaligned inverted)
- `NNNNN__dMMMMM.json` (template, donor, y_cut, face_w, mis_dx_px, ssim_v1v2, seam_disc, lab_delta_cut, template_yaw_deg, donor_yaw_deg)
- `NNNNN__dMMMMM.contact.png` (2×2 QC grid — exclude from tar)

### 2.4 8-condition analysis design

Each template has 3 donors → 6 ordered (target, foil) pairs per template. Each pair maps to 8 trial conditions: target/foil × aligned/misaligned × upright/inverted. We store only 4 PNGs per (T, D); the 8-condition trial structure is built in the loader at analysis time. See `project_composite_design_2026-05-27.md` for the full Holistic Index formula.

---

## 3. Code (all in `/Users/enhuili/Desktop/EEG/illusionbench/stimuli/classical_cv/`)

| File | Lines | Purpose |
|---|---|---|
| **`composite_v2_pilot.py`** | ~660 | Self-contained single-process pilot (algorithm reference) |
| **`composite_v2_parallel.py`** | ~430 | Multiprocess Phase A + Phase B for full-scale generation |
| **`streaming_phase_a.py`** | ~140 | Stream-process tars as they download (for Mac-local) |
| **`package_composite_v2.py`** | ~85 | Pack flat PNG dump into webdataset tars |
| **`upload_composite_v2.py`** | ~200 | (DEPRECATED — use `hf upload` direct instead, see §6) |
| `feature_warp.py` | existing | TPS warp utility (composite uses similarity, not TPS, but kept for ref) |
| `color_transfer.py` | existing | Reinhard LAB transfer (reused) |
| `face_embedding.py` | existing | dlib 128-d embedder (reused) |
| `donor_match.py` | existing | IdAttr dataclass + base filters (reused by parallel) |
| `insight_attrs.py` | existing | InsightFace genderage wrapper (reused) |

### 3.1 What composite_v2_parallel.py does

```bash
# Phase A: per-identity attrs (reuses partial-tar-safe _iter_thatcher_tar_v1)
python -m stimuli.classical_cv.composite_v2_parallel \
    --phase a --workers 32 \
    --tar_src /path/to/thatcher_tars_dir \
    --attrs_pkl /path/to/composite_attrs.pkl

# Phase B: per-template composite generation (writes flat dir)
python -m stimuli.classical_cv.composite_v2_parallel \
    --phase b --workers 32 \
    --tar_src /path/to/thatcher_tars_dir \
    --attrs_pkl /path/to/composite_attrs.pkl \
    --png_dir /path/to/composite_v2_pngs \
    --donors_per_template 3

# Phase C: package into webdataset tars
python -m stimuli.classical_cv.package_composite_v2 \
    --src /path/to/composite_v2_pngs \
    --dst /path/to/composite_v2 \
    --pairs_per_shard 500
```

### 3.2 Critical implementation details (bugs that took hours to find)

**Bug 1: PIL on partial tar**. `tf.getmembers()` raises ReadError on partial tars. Fixed by iterating with `tf.next()` in a loop and breaking on ReadError. See `_iter_thatcher_tar_v1` in `composite_v2_parallel.py`.

**Bug 2: tar member offset random access**. `tf.extractfile("name")` does an internal `_load()` which fails on partial tars. Fixed by indexing `(tar_path, offset_data, size)` and using `open(tar).seek(offset).read(size)` directly. See `_build_png_index` and `_load_rgb`.

**Bug 3: `tf.next()` advances even when reading a partial member's data**. The yield-and-continue pattern reads truncated PNG bytes for the LAST partial member. Filter these by verifying `len(data) == m.size` in `_build_png_index`.

**Bug 4: Phase B IdAttr pool**. Building IdAttr objects per template wastes 5 sec per template. Pre-build the rgb=None pool ONCE in `_phase_b_init`. See `_attrs_idattr_pool`.

**Bug 5: TPS frontalization causes lower-face distortion**. Tried in v0.5 and abandoned. Use mouth-centrality + cheek-width filters instead (v0.6+). See pilot history in `~/Desktop/composite_pilot_preview/`.

---

## 4. Server setup (USER MUST PROVIDE)

This handoff doc cannot tell you the server's hostname / SSH path because the user never shared it with me. **Ask the user**:

> "请告诉我服务器的 SSH 地址、用户名、以及 PW v1.4 跑的 attrs.pkl 路径（之前 /workspace/... 那个）"

Once you have:
- `SSH_HOST`, `SSH_USER`
- Workspace path (likely `/workspace/...`)
- PW v1.4 `attrs.pkl` path

Then:
```bash
# 1. SCP code to server
scp -r stimuli/classical_cv/composite_v2_*.py stimuli/classical_cv/package_composite_v2.py \
    stimuli/classical_cv/feature_warp.py stimuli/classical_cv/color_transfer.py \
    stimuli/classical_cv/face_embedding.py stimuli/classical_cv/donor_match.py \
    stimuli/classical_cv/insight_attrs.py models/ \
    $SSH_USER@$SSH_HOST:/workspace/HoloFaceIllusion/

# 2. On server: install MediaPipe + hf_xet (most other deps already installed for PW v1.4)
ssh $SSH_USER@$SSH_HOST 'pip install mediapipe hf_xet onnxruntime'

# 3. Download MediaPipe face_landmarker.task model (~3.6MB)
ssh $SSH_USER@$SSH_HOST 'curl -sL -o /workspace/models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
```

### 4.1 If server already has FFHQ Thatcher tars locally

The original Thatcher tars (e.g. `/workspace/.hf_cache/.../00000.tar` ... `69000.tar`) contain V1 + landmarks for all 26,317 identities. Phase A reads these directly.

### 4.2 If server has PW v1.4 attrs.pkl

PW v1.4's attrs.pkl has most fields but **MISSING**: `mp_lm` (MediaPipe 478), `mouth_offset_ratio`, `cheek_w_at_cut`, `cut_to_chin_ratio`. You have two options:

**Option A** (re-run Phase A from scratch — 30-60 min on 32-core server):
```bash
python -m stimuli.classical_cv.composite_v2_parallel \
    --phase a --workers 32 \
    --tar_src /workspace/HoloFaceIllusion/thatcher_tars \
    --attrs_pkl /workspace/HoloFaceIllusion/composite_attrs.pkl
```

**Option B** (extend existing PW v1.4 attrs with new fields only — faster but more code):
- Load PW v1.4 attrs.pkl
- For each row, add missing fields by re-running ONLY the MediaPipe + mouth_offset + cheek_w_at_cut + cut_to_chin computations
- Save as `composite_attrs.pkl`
- Saves dlib + InsightFace re-runs (the expensive part)

Recommended: **Option A** for simplicity. 32 cores × ~30/s per worker = ~13 sec/identity × 26k / 32 = ~10 min on server. Trivial.

---

## 5. Full-scale execution plan (on server)

```bash
# === Phase A (server, ~30 min) ===
python -m stimuli.classical_cv.composite_v2_parallel \
    --phase a --workers 32 \
    --tar_src /workspace/.../thatcher_tars \
    --attrs_pkl /workspace/HoloFaceIllusion/composite_attrs.pkl

# Expected output:
#   [phase A] DONE: extracted ~26000 / 26317
#   filter: -infant(~2k) -yaw>15(~3k) -mouth-off-axis(~600)
#   kept after filter: ~20,000 attrs
#   saved: /workspace/.../composite_attrs.pkl (~250 MB)

# === Phase B (server, ~30-60 min on 32 cores) ===
mkdir -p /workspace/HoloFaceIllusion/composite_v2_pngs
python -m stimuli.classical_cv.composite_v2_parallel \
    --phase b --workers 32 \
    --tar_src /workspace/.../thatcher_tars \
    --attrs_pkl /workspace/HoloFaceIllusion/composite_attrs.pkl \
    --png_dir /workspace/HoloFaceIllusion/composite_v2_pngs \
    --donors_per_template 3

# Expected output (~70% template pass rate from 20k):
#   total templates: ~20000
#   ok: ~14000  pool_too_small: ~6000
#   total PNGs: ~168000 (14k × 3 × 4) + ~42k JSON + ~42k contact PNG = ~252k files
#   disk usage: ~70 GB raw

# === Phase C (~5 min) ===
python -m stimuli.classical_cv.package_composite_v2 \
    --src /workspace/HoloFaceIllusion/composite_v2_pngs \
    --dst /workspace/HoloFaceIllusion/composite_v2 \
    --pairs_per_shard 500

# Expected output: ~28-30 tar shards × ~850MB each
#   data/composite_v2/v2.0/composite_000.tar ... composite_029.tar
#   v2.0/manifest.csv

# === Phase D (Upload to HF — HP MODE!) ===
export HF_TOKEN=<your_HF_token_here>  # store in /workspace/.secrets/hf.env
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_FIXED_UPLOAD_CONCURRENCY=64
export HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION=32

# Server has gigabit upload → ~30-50 MB/s effective
# Upload 28 tars × 850MB ≈ 24 GB in 8-15 minutes
hf upload Enhui-1/HoloFaceIllusion-Bench-EEG \
    /workspace/HoloFaceIllusion/composite_v2/v2.0 v2.0 \
    --repo-type dataset \
    --commit-message "composite v2.0 full-scale (~14k templates, ~42k cases)"

# Verify
curl -s "https://huggingface.co/api/datasets/Enhui-1/HoloFaceIllusion-Bench-EEG/tree/main/v2.0" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(len([e for e in d if e['type']=='file']), 'files')"

# === Phase E (Spot-check) ===
# Download 1-2 random tars from HF, extract a few sample PNGs, verify visually:
hf download Enhui-1/HoloFaceIllusion-Bench-EEG \
    --include "v2.0/composite_015.tar" --local-dir /tmp/spotcheck/
tar -xf /tmp/spotcheck/v2.0/composite_015.tar -C /tmp/spotcheck/ \
    --wildcards "*__d*.v1.png" "*__d*.v2.png" 2>/dev/null | head -10
# Inspect the extracted PNGs visually
```

---

## 6. Upload tips (HARD-EARNED)

**THE KEY DISCOVERY OF THIS SESSION**:

```bash
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_FIXED_UPLOAD_CONCURRENCY=64
export HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION=32
hf upload <repo> <local_dir> <repo_subdir> --repo-type dataset
```

Without HP mode: **~250 KB/s** even on fiber (default initial_concurrency=1, ramps slowly).
With HP mode: **5–7 MB/s** observed on user's home cable (20-30× faster). On fiber expect 30-50+ MB/s.

The HP mode jumps concurrency from 1 → 16 initial, 64 → 124 max. Requires 64+ GB RAM (server has it).

**DO NOT use the `upload_composite_v2.py` watchdog wrappers I wrote** — they all fail because:
- `hf upload` is genuinely silent for 5-30 min during xet handshake (no stdout/stderr).
- `HfApi.upload_file()` rejects custom file-like wrappers (ValueError, must be BufferedIOBase).
- Watchdogs that kill on stdout silence kill mid-handshake repeatedly, never making progress.

Just run `hf upload` directly with HP env vars and trust it.

If you DO want a stall-detection safety net, use a 30-min stall timer based on `repo_files()` API (which only updates after commit) — see Section 4 of `project_composite_upload_2026-05-28.md` memory.

---

## 7. Token + secrets

User's HF token is stored server-side at `/workspace/.secrets/hf.env` (mode 600).
Source it per command (`source /workspace/.secrets/hf.env`). NEVER commit the
literal token value. Advise user to rotate after the project is published.

---

## 8. Files preserved from this session (for reference)

| Path | Purpose |
|---|---|
| `/Users/enhuili/Desktop/EEG/illusionbench/data/composite_v2/v2.0/*.tar` | Pilot 4.3 GB output (CAN DELETE if regenerating on server) |
| `/Users/enhuili/Desktop/EEG/illusionbench/data/composite_v2_pngs/` | Pilot raw PNG dump (CAN DELETE) |
| `/Users/enhuili/Desktop/EEG/illusionbench/data/composite_attrs.pkl` | Pilot attrs (~1257 IDs) (DELETE on server, regenerate from 26k) |
| `/Users/enhuili/Desktop/EEG/illusionbench/data/composite_ffhq_src/` | Partially downloaded Thatcher tars (CAN DELETE — server has full set) |
| `/Users/enhuili/Desktop/composite_pilot_preview/` | Visual QC pilot samples (v0p3, v0p4, v0p5, v0p6, v0p7) — KEEP for sanity reference |

---

## 9. Memory references

Memories you should load:
- `~/.claude/projects/-Users-enhuili-Desktop-EEG/memory/MEMORY.md` (index)
- `project_composite_design_2026-05-27.md` (algorithm spec, all design rationale)
- `project_composite_upload_2026-05-28.md` (upload lessons from this session)
- `project_partwhole_template_2026-05-26.md` (template-based design used in PW v1.4)
- `project_classical_cv_pivot.md` (NO DNN/generative rule — composite obeys this; MediaPipe + InsightFace are pre-trained, OK)
- `feedback_api_key_handling.md` (how to handle user's HF token)

---

## 10. What to ASK the user before starting

1. **Server SSH details**: hostname, user, key/password method.
2. **Server workspace path**: where is `/workspace/...`? where are Thatcher tars?
3. **Whether to use existing PW v1.4 attrs.pkl** (Option B above) or regenerate from scratch (Option A, recommended).
4. **Final scale target**: confirm "~14k templates × 3 donors" is the target, or push for full 20k+ if pool allows.
5. **Whether to delete pilot output** in `~/Desktop/EEG/illusionbench/data/composite_v2*` (saves disk).

---

## 11. Definition of DONE

The user considers this task complete when:
- [ ] Phase A → Phase B → Phase C all completed on server
- [ ] All composite tars uploaded to HF (~28 tars, ~24 GB)
- [ ] `v2.0/manifest.csv` matches the tars
- [ ] Spot-check: download 1 random tar, extract V1/V2/V3/V4 of one (T,D), verify visually
- [ ] Total cases comparable to Thatcher (26k) or PW (188k) — target ~42k cases
- [ ] User can `load_dataset("Enhui-1/HoloFaceIllusion-Bench-EEG", name="composite", streaming=True)` and iterate samples

When done, report back to user with:
- Final case count
- HF URL: https://huggingface.co/datasets/Enhui-1/HoloFaceIllusion-Bench-EEG
- 2–3 spot-checked sample PNGs

---

## 12. Open questions / known unknowns

- **Server vs Mac performance**: PW v1.4 ran on a 64-core server. Phase B at 32-core 32-worker should be ~15-30 min for 14k templates.
- **HF rate limit at scale**: User said "HF 限速" was the concern. Testing showed it's actually home upload bandwidth, NOT HF limiting. On server with fiber, HP mode should easily hit 30-50 MB/s.
- **Storage on server**: full output is ~70 GB raw + ~24 GB packaged = ~94 GB. Confirm server has space.
- **CelebA facial-hair filter**: user opted out of this in pilot. If reviewer asks for cleaner male/female differentiation later, add CelebA classifier filter (5 binary attrs: Beard, Mustache, Goatee, Sideburns, 5_o_Clock_Shadow). Not blocking.

---

**End of handoff. Good luck.**
