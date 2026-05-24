"""stimuli/gpt_image_pilot.py — pilot generation of 10 GPT-image source faces.

Goals:
  - Validate API access + measure latency + cost.
  - Check whether MediaPipe FaceMesh recognizes GPT-generated faces (essential
    since downstream Thatcher/composite/PW pipelines all key off FaceMesh).
  - Cover 10 demographic cells (gender × ethnicity × age) so we surface any
    generator bias early.

The 10 prompts share a fixed template that controls: studio portrait, neutral
expression, direct gaze, gray background, soft lighting, no accessories,
synthetic non-identifiable face (anti-likeness clause).

Outputs into data/stimuli_gpt_pilot/:
  - identity_{NNNN}.png       (1024x1024 generated face)
  - identity_{NNNN}.json      (full metadata: prompt, model, version, timing,
                                cost, MediaPipe pass/fail, IOD, tilt)
  - pilot_manifest.csv        (summary table)
  - pilot_report.md           (rendered markdown summary)

Reads OPENAI_API_KEY from environment. Use:
  source /workspace/.secrets/openai.env
  python -m stimuli.gpt_image_pilot --output_root data/stimuli_gpt_pilot
"""
from __future__ import annotations
import argparse
import base64
import csv
import io
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

# Image-generation cost table (as of 2026; verify each invocation).
# These are USD list prices for gpt-image-1; cents per image.
QUALITY_COST = {
    "low":    0.011,   # 1024x1024
    "medium": 0.042,
    "high":   0.167,
}

# 10 demographic cells for the pilot, balanced across gender × ethnicity × age.
# Use a deterministic order so re-runs match.
DEMOGRAPHICS = [
    {"id": 0, "gender": "female",     "ethnicity": "East Asian",     "age": "young adult"},
    {"id": 1, "gender": "male",       "ethnicity": "Black",          "age": "middle-aged"},
    {"id": 2, "gender": "male",       "ethnicity": "White",          "age": "young adult"},
    {"id": 3, "gender": "female",     "ethnicity": "South Asian",    "age": "middle-aged"},
    {"id": 4, "gender": "male",       "ethnicity": "Latinx",         "age": "older adult"},
    {"id": 5, "gender": "female",     "ethnicity": "White",          "age": "young adult"},
    {"id": 6, "gender": "male",       "ethnicity": "East Asian",     "age": "middle-aged"},
    {"id": 7, "gender": "female",     "ethnicity": "Black",          "age": "older adult"},
    {"id": 8, "gender": "male",       "ethnicity": "Middle Eastern", "age": "young adult"},
    {"id": 9, "gender": "female",     "ethnicity": "Latinx",         "age": "middle-aged"},
]

PROMPT_TEMPLATE = (
    "Professional studio portrait headshot photograph of a {age} {ethnicity} "
    "{gender} adult. Neutral facial expression, mouth closed, no smile, eyes "
    "open and looking directly at the camera. Plain neutral gray background. "
    "Even soft frontal studio lighting with no harsh shadows. Sharp focus on "
    "the face. The person wears no makeup, no jewelry, no glasses, no hat, "
    "no hair covering, plain neutral clothing visible only at the shoulders. "
    "Standard passport-style framing: face fills approximately the central "
    "60% of the frame, head and neck visible. Photorealistic high-resolution "
    "DSLR photograph quality.\n\n"
    "IMPORTANT: Generate a fully synthetic, non-identifiable face. Do NOT "
    "reproduce the likeness of any real person, public figure, or celebrity."
)


@dataclass
class PilotRecord:
    identity_id: int
    gender: str
    ethnicity: str
    age: str
    prompt: str
    model: str
    quality: str
    size: str
    elapsed_s: float
    cost_usd: float
    image_path: str
    output_revision: str
    mediapipe_pass: bool
    iod_px: float
    tilt_deg: float
    face_landmark_count: int
    notes: str


def call_openai_image(prompt: str, model: str, quality: str, size: str):
    """Returns (PIL.Image, elapsed_s, raw_response_dict).
    Uses the synchronous OpenAI Python SDK >= 2.0 (Images endpoint)."""
    import openai
    client = openai.OpenAI()
    t0 = time.monotonic()
    resp = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
    )
    elapsed = time.monotonic() - t0
    # gpt-image-1 returns base64-encoded image bytes by default
    b64 = resp.data[0].b64_json
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return img, elapsed, resp.model_dump() if hasattr(resp, "model_dump") else {}


def mediapipe_check(img: Image.Image):
    """Returns (pass: bool, iod_px: float, tilt_deg: float, n_landmarks: int).

    Tolerant of MediaPipe environment failures (e.g., protobuf incompat) —
    returns (False, NaN, NaN, 0) and lets the caller record a 'SKIP' note
    rather than aborting the pilot. The downstream Thatcher pipeline will
    run MediaPipe properly in its own venv at QC time."""
    try:
        import mediapipe as mp
        rgb = np.array(img)
        h, w = rgb.shape[:2]
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.9,
        ) as fm:
            result = fm.process(rgb)
        if not result.multi_face_landmarks:
            return False, float("nan"), float("nan"), 0
        lm = result.multi_face_landmarks[0].landmark
        arr = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        le = arr[LEFT_EYE].mean(axis=0)
        re = arr[RIGHT_EYE].mean(axis=0)
        iod = float(np.linalg.norm(le - re))
        dy = re[1] - le[1]; dx = re[0] - le[0]
        tilt = float(np.degrees(np.arctan2(abs(dy), abs(dx))))
        return True, iod, tilt, len(lm)
    except Exception as e:
        print(f"  [mediapipe SKIP: {type(e).__name__}: {str(e)[:80]}]")
        return False, float("nan"), float("nan"), -1  # -1 signals SKIPPED, not failed


def main(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit(
            "OPENAI_API_KEY not set. Run: source /workspace/.secrets/openai.env"
        )

    records: list[PilotRecord] = []
    total_cost = 0.0
    total_time = 0.0
    print(f"[init] model={args.model} quality={args.quality} size={args.size}")
    print(f"[init] cost-per-image estimate = ${QUALITY_COST.get(args.quality, 0.0):.4f}")

    for cell in DEMOGRAPHICS:
        prompt = PROMPT_TEMPLATE.format(**cell)
        identity_id = cell["id"]
        img_path = out_root / f"identity_{identity_id:04d}.png"
        json_path = out_root / f"identity_{identity_id:04d}.json"
        print(f"\n[{identity_id+1:02d}/10] {cell['age']} {cell['ethnicity']} {cell['gender']}")

        # Idempotent: if image already exists, reload it and re-run QC only.
        if img_path.exists() and not args.regenerate:
            print(f"  reusing existing {img_path.name} (no API call)")
            img = Image.open(img_path).convert("RGB")
            elapsed = 0.0
            cost = 0.0
            raw_resp = {}
            mp_pass, iod, tilt, nlm = mediapipe_check(img)
            record = PilotRecord(
                identity_id=identity_id, gender=cell["gender"], ethnicity=cell["ethnicity"],
                age=cell["age"], prompt=prompt, model=args.model, quality=args.quality,
                size=args.size, elapsed_s=elapsed, cost_usd=cost, image_path=str(img_path),
                output_revision="(cached)", mediapipe_pass=mp_pass, iod_px=iod,
                tilt_deg=tilt, face_landmark_count=nlm,
                notes="reused cached generation",
            )
            records.append(record)
            with open(json_path, "w") as f:
                json.dump(asdict(record), f, indent=2, default=str)
            continue

        try:
            img, elapsed, raw_resp = call_openai_image(
                prompt, args.model, args.quality, args.size
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            records.append(PilotRecord(
                identity_id=identity_id, gender=cell["gender"], ethnicity=cell["ethnicity"],
                age=cell["age"], prompt=prompt, model=args.model, quality=args.quality,
                size=args.size, elapsed_s=0.0, cost_usd=0.0, image_path="",
                output_revision="", mediapipe_pass=False, iod_px=float("nan"),
                tilt_deg=float("nan"), face_landmark_count=0, notes=f"API_ERROR: {e}",
            ))
            continue

        img_path = out_root / f"identity_{identity_id:04d}.png"
        img.save(img_path)
        cost = QUALITY_COST.get(args.quality, 0.0)
        total_cost += cost
        total_time += elapsed
        mp_pass, iod, tilt, nlm = mediapipe_check(img)
        notes = "" if mp_pass else "MediaPipe detection failed (no face)"
        print(f"  saved {img_path.name}  elapsed={elapsed:.1f}s  cost=${cost:.4f}  "
              f"MP={'PASS' if mp_pass else 'FAIL'}  iod={iod:.1f}px  tilt={tilt:.1f}deg")

        record = PilotRecord(
            identity_id=identity_id, gender=cell["gender"], ethnicity=cell["ethnicity"],
            age=cell["age"], prompt=prompt, model=args.model, quality=args.quality,
            size=args.size, elapsed_s=elapsed, cost_usd=cost, image_path=str(img_path),
            output_revision=str(raw_resp.get("model", "")) if raw_resp else "",
            mediapipe_pass=mp_pass, iod_px=iod, tilt_deg=tilt,
            face_landmark_count=nlm, notes=notes,
        )
        records.append(record)
        with open(out_root / f"identity_{identity_id:04d}.json", "w") as f:
            json.dump(asdict(record), f, indent=2, default=str)

    # write manifest CSV
    csv_path = out_root / "pilot_manifest.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in records])

    # write markdown report
    n_ok = sum(1 for r in records if r.mediapipe_pass)
    mp_rate = n_ok / len(records) if records else 0
    mean_iod = float(np.nanmean([r.iod_px for r in records if r.mediapipe_pass]))
    mean_tilt = float(np.nanmean([r.tilt_deg for r in records if r.mediapipe_pass]))
    md_lines = [
        f"# GPT-Image-1 pilot — {args.quality} quality\n",
        f"- **model**: {args.model}",
        f"- **quality / size**: {args.quality} / {args.size}",
        f"- **N pilot**: {len(records)} (10 demographic cells)",
        f"- **Total cost**: ${total_cost:.4f}",
        f"- **Total elapsed**: {total_time:.1f}s (mean per image: "
        f"{total_time/max(len(records),1):.1f}s)",
        f"- **MediaPipe pass rate**: {n_ok}/{len(records)} ({mp_rate*100:.0f}%)",
        f"- **Mean IOD** (passing only): {mean_iod:.1f} px",
        f"- **Mean tilt** (passing only): {mean_tilt:.2f} deg\n",
        "## Per-image\n",
        "| id | demo | elapsed (s) | cost | MP | IOD (px) | tilt (°) | notes |",
        "|---:|---|---:|---:|:-:|---:|---:|---|",
    ]
    for r in records:
        demo = f"{r.age} {r.ethnicity} {r.gender}"
        md_lines.append(
            f"| {r.identity_id} | {demo} | {r.elapsed_s:.1f} | "
            f"${r.cost_usd:.4f} | "
            f"{'PASS' if r.mediapipe_pass else 'FAIL'} | "
            f"{r.iod_px:.0f} | {r.tilt_deg:.1f} | {r.notes} |"
        )
    md_lines.append("\n## Budget projection\n")
    if records:
        cost_per_img = total_cost / len(records)
        for n in [100, 200, 300, 500, 1000, 3000]:
            md_lines.append(f"- N={n} → est. **${cost_per_img*n:.2f}**")
    md_lines.append("\n## Replicability\n")
    md_lines.append(
        f"Prompt template + demographic grid are in `stimuli/gpt_image_pilot.py`. "
        f"Per-identity JSON contains full prompt + API response metadata."
    )
    report_path = out_root / "pilot_report.md"
    report_path.write_text("\n".join(md_lines))
    print(f"\n=== PILOT DONE ===")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Total time: {total_time:.1f}s ({total_time/max(len(records),1):.1f}s/image)")
    print(f"  MediaPipe pass: {n_ok}/{len(records)} ({mp_rate*100:.0f}%)")
    print(f"  Manifest:  {csv_path}")
    print(f"  Report:    {report_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default="data/stimuli_gpt_pilot")
    p.add_argument("--model", default="gpt-image-1",
                   help="OpenAI image model. As of 2026: gpt-image-1 is the "
                        "current name (the user-facing 'GPT-image-2' branding "
                        "may map to gpt-image-1 internally).")
    p.add_argument("--quality", default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--size", default="1024x1024",
                   choices=["1024x1024", "1024x1536", "1536x1024"])
    p.add_argument("--regenerate", action="store_true",
                   help="force re-generation even if cached image exists "
                        "(WILL spend API credits — default is to reuse cache)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
