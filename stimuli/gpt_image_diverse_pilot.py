"""stimuli/gpt_image_diverse_pilot.py — 10-source × 4-condition pilot.

New strategy (user direction 2026-05-25):
  - Source faces: DIVERSE (varied scenes, lighting, slight angles, some shadows)
    rather than monolithic passport-style. Matches natural face variation.
  - Illusion variants: GENERATED via OpenAI images.edit() API (NOT local
    OpenCV transforms). User does not trust local transforms to produce
    seamless illusion stimuli.

Per group = 4 images:
  1. V1_source      — diverse natural portrait
  2. V2_thatcher    — same face, eyes + mouth flipped 180° in place
  3. V3_composite   — same top half, different person's bottom half
  4. V4_partwhole   — same face, different person's eye region

10 groups → 40 images total, estimated $6.68 at high quality.

Outputs under data/stimuli_gpt_diverse_pilot/:
  group_{NN}/V1_source.png
  group_{NN}/V2_thatcher.png
  group_{NN}/V3_composite.png
  group_{NN}/V4_partwhole.png
  group_{NN}/meta.json
  pilot_manifest.csv
  pilot_report.md

CAVEAT: this pilot tests FEASIBILITY of API-generated illusion variants.
If the model cannot produce a recognizable Thatcher effect via edit prompt,
that's a critical negative result and we'd revert to mathematical transforms.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import csv
import io
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image


QUALITY_COST = {"low": 0.011, "medium": 0.042, "high": 0.167}

# 10 demographic + scene cells, hand-curated for diversity.
GROUPS = [
    {
        "group_id": 0, "gender": "female", "ethnicity": "East Asian", "age": "young adult",
        "scene": "outdoor in a soft-lit city park during golden hour, blurred trees behind",
        "lighting": "warm directional sunlight from upper-left",
        "shadow": "subtle warm shadow on the right cheek and under the jaw",
        "angle": "head turned approximately 5 degrees right from frontal, eyes meeting the camera",
        "hair": "shoulder-length straight dark hair",
        "complexion": "fair complexion with natural skin texture",
    },
    {
        "group_id": 1, "gender": "male", "ethnicity": "Black", "age": "middle-aged",
        "scene": "indoor at a warmly-lit coffee shop, soft bokeh of lights behind",
        "lighting": "warm ambient indoor light from a window on the right",
        "shadow": "soft natural shadow under the eyes and on the left side of the nose",
        "angle": "head turned approximately 8 degrees left, gaze directly at camera",
        "hair": "short closely-cropped hair with a trimmed beard",
        "complexion": "deep brown complexion with visible skin pores",
    },
    {
        "group_id": 2, "gender": "male", "ethnicity": "White", "age": "young adult",
        "scene": "office with daylight pouring through a large window, blurred desk behind",
        "lighting": "cool natural daylight from the front-left, soft and even",
        "shadow": "very faint shadow on the right neck and lower jaw",
        "angle": "near-frontal, head tilted slightly down with eyes lifted to camera",
        "hair": "medium-length wavy light-brown hair, slightly tousled",
        "complexion": "fair complexion with slight freckles and natural skin texture",
    },
    {
        "group_id": 3, "gender": "female", "ethnicity": "South Asian", "age": "middle-aged",
        "scene": "studio with a deep teal painted backdrop, simple background",
        "lighting": "soft frontal key light with a slight rim from the right",
        "shadow": "gentle sculpting shadow along the right jaw, no harsh edges",
        "angle": "frontal, gaze directly at camera",
        "hair": "long straight dark hair pulled back loosely",
        "complexion": "warm medium-brown complexion with natural skin tone",
    },
    {
        "group_id": 4, "gender": "male", "ethnicity": "Latinx", "age": "older adult",
        "scene": "casual outdoor setting with a stone wall and greenery behind, blurred",
        "lighting": "overcast soft natural light, very even and flattering",
        "shadow": "minimal soft shadow, gentle facial contouring",
        "angle": "head turned approximately 5 degrees left, calm expression",
        "hair": "short greying hair with a neatly trimmed grey moustache",
        "complexion": "olive complexion with deep skin texture and age lines",
    },
    {
        "group_id": 5, "gender": "female", "ethnicity": "White", "age": "young adult",
        "scene": "indoor living room with a window in the background, soft cozy atmosphere",
        "lighting": "soft natural side light from the left through a sheer curtain",
        "shadow": "gentle soft shadow on the right cheek with no harsh edges",
        "angle": "near-frontal, head slightly turned 7 degrees right, eyes on camera",
        "hair": "long wavy chestnut-brown hair falling over one shoulder",
        "complexion": "fair complexion with natural undertones and clear skin",
    },
    {
        "group_id": 6, "gender": "male", "ethnicity": "East Asian", "age": "middle-aged",
        "scene": "studio with a neutral grey backdrop, professional clean setting",
        "lighting": "balanced three-point studio lighting, soft and even",
        "shadow": "very subtle shadow under the chin, no harsh contrast",
        "angle": "frontal, calm focused gaze at camera",
        "hair": "short business-cut dark hair with slight greying at temples",
        "complexion": "warm medium-tan complexion with natural skin texture",
    },
    {
        "group_id": 7, "gender": "female", "ethnicity": "Black", "age": "older adult",
        "scene": "warm-lit indoor setting with a wooden bookshelf in the background, blurred",
        "lighting": "warm tungsten ambient lighting from upper-right",
        "shadow": "soft warm shadow on the left side of the face, gentle contouring",
        "angle": "near-frontal, slight tilt of the head left, dignified expression",
        "hair": "short natural grey-and-black curly hair",
        "complexion": "rich dark brown complexion with visible texture and dignity",
    },
    {
        "group_id": 8, "gender": "male", "ethnicity": "Middle Eastern", "age": "young adult",
        "scene": "outdoor in a sunlit courtyard with stone walls, blurred natural environment",
        "lighting": "bright midday sun softened by overhead shade, even soft light",
        "shadow": "subtle shadow under the brow ridge from soft overhead light",
        "angle": "head turned approximately 6 degrees right, direct gaze to camera",
        "hair": "medium-length dark curly hair with a short well-groomed beard",
        "complexion": "olive complexion with natural warmth and skin texture",
    },
    {
        "group_id": 9, "gender": "female", "ethnicity": "Latinx", "age": "middle-aged",
        "scene": "indoor studio with a warm beige backdrop and natural texture",
        "lighting": "soft warm key light from the front and gentle fill from the left",
        "shadow": "delicate shadow on the right side of the nose and jaw, very soft",
        "angle": "frontal, peaceful direct gaze to camera",
        "hair": "shoulder-length dark wavy hair parted in the middle",
        "complexion": "medium-tan complexion with natural skin texture and warm undertones",
    },
]


SOURCE_PROMPT_TEMPLATE = """Photorealistic high-resolution DSLR portrait photograph of a {age} {ethnicity} {gender} adult with {hair}, {complexion}. {angle}. The face is well-visible, occupying approximately 60% of the frame, showing head and shoulders only. Neutral facial expression with mouth closed and eyes open.

Setting: {scene}.
Lighting: {lighting}, creating {shadow}.

IMPORTANT: Generate a fully synthetic, non-identifiable face. Do NOT reproduce the likeness of any real person, public figure, or celebrity. The face must look natural, not airbrushed."""


ILLUSION_EDIT_PROMPTS = {
    "V2_thatcher": (
        "Edit this exact photograph to apply the Thatcher illusion transform: "
        "rotate BOTH eye regions (including eyebrows and lashes) 180 degrees in place, "
        "AND rotate the mouth region 180 degrees in place. The eyes appear upside-down "
        "(eyebrows below the eyes, lower lashes above). The mouth appears upside-down "
        "(the chin-side lip ends up on top, the nose-side lip ends up on the bottom). "
        "Keep the rest of the face IDENTICAL: same forehead, nose bridge, nose tip, "
        "cheeks, jaw, chin shape, hair, skin tone, lighting, background, and overall "
        "head orientation. Photorealistic, seamless integration of the rotated regions, "
        "no visible cut lines or rectangular boundaries."
    ),
    "V3_composite": (
        "Edit this exact photograph: keep the TOP HALF of the face (forehead, both eyes, "
        "both eyebrows, nose bridge down to the nose tip) IDENTICAL to the input — same "
        "person above the nose. REPLACE the BOTTOM HALF of the face (lower nose, mouth, "
        "chin, jaw line) with the corresponding lower-face features of a DIFFERENT but "
        "plausible person of the SAME approximate age, gender, ethnicity, and lighting. "
        "Blend seamlessly across the nose tip line — no visible seam. Match the skin "
        "tone, lighting direction, and natural texture between halves. Keep the hair, "
        "background, and shoulders unchanged. Photorealistic, no visible cut line."
    ),
    "V4_partwhole": (
        "Edit this exact photograph: keep EVERYTHING IDENTICAL — same forehead, nose, "
        "mouth, chin, jaw, hair, skin tone, lighting, background — EXCEPT replace just "
        "the eye region (both eyes and both eyebrows) with the eyes and eyebrows of a "
        "DIFFERENT but plausible person of the same approximate age, gender, ethnicity. "
        "The new eyes should have different shape, color, and expression. Blend "
        "seamlessly into the surrounding skin — no visible patch boundary. Match the "
        "skin tone, lighting direction, and natural texture. Photorealistic."
    ),
}


@dataclass
class PilotRecord:
    group_id: int
    variant: str       # V1_source / V2_thatcher / V3_composite / V4_partwhole
    image_path: str
    prompt: str
    elapsed_s: float
    cost_usd: float
    success: bool
    error: str


async def call_generate(client, prompt: str, model: str, quality: str, size: str):
    """generate() = create from prompt, no input image."""
    resp = await client.images.generate(
        model=model, prompt=prompt, size=size, quality=quality, n=1,
    )
    b64 = resp.data[0].b64_json
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


async def call_edit(client, image_bytes: bytes, prompt: str, model: str,
                     quality: str, size: str):
    """edit() = transform an existing image. gpt-image-1 supports this."""
    # The openai SDK's images.edit accepts a file-like object as `image`.
    file_like = io.BytesIO(image_bytes)
    file_like.name = "source.png"   # required by the SDK to infer mime
    resp = await client.images.edit(
        model=model, image=file_like, prompt=prompt,
        size=size, quality=quality, n=1,
    )
    b64 = resp.data[0].b64_json
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


async def run_group(client, cell: dict, args, root: Path,
                     cost_state: dict, sem: asyncio.Semaphore) -> list[PilotRecord]:
    gid = cell["group_id"]
    g_dir = root / f"group_{gid:02d}"
    g_dir.mkdir(parents=True, exist_ok=True)
    records: list[PilotRecord] = []

    # 1) source (generate)
    src_prompt = SOURCE_PROMPT_TEMPLATE.format(**cell)
    src_path = g_dir / "V1_source.png"
    async with sem:
        if src_path.exists() and not args.regenerate:
            print(f"[{gid:02d}] V1 cached")
            src_img = Image.open(src_path).convert("RGB")
            records.append(PilotRecord(
                group_id=gid, variant="V1_source", image_path=str(src_path),
                prompt=src_prompt, elapsed_s=0.0, cost_usd=0.0,
                success=True, error="(cached)",
            ))
        else:
            t0 = time.monotonic()
            try:
                src_img = await call_generate(
                    client, src_prompt, args.model, args.quality, args.size,
                )
                src_img.save(src_path)
                elapsed = time.monotonic() - t0
                cost = QUALITY_COST[args.quality]
                cost_state["spent"] += cost
                print(f"[{gid:02d}] V1 source {elapsed:.1f}s "
                      f"spent=${cost_state['spent']:.2f}")
                records.append(PilotRecord(
                    group_id=gid, variant="V1_source", image_path=str(src_path),
                    prompt=src_prompt, elapsed_s=elapsed, cost_usd=cost,
                    success=True, error="",
                ))
            except Exception as e:
                print(f"[{gid:02d}] V1 ERROR: {e}")
                records.append(PilotRecord(
                    group_id=gid, variant="V1_source", image_path="",
                    prompt=src_prompt, elapsed_s=0.0, cost_usd=0.0,
                    success=False, error=str(e)[:200],
                ))
                return records  # cannot do edits without source

    # Read source bytes for edit calls
    with open(src_path, "rb") as f:
        src_bytes = f.read()

    # 2-4) edits
    for variant in ["V2_thatcher", "V3_composite", "V4_partwhole"]:
        out_path = g_dir / f"{variant}.png"
        edit_prompt = ILLUSION_EDIT_PROMPTS[variant]
        async with sem:
            if out_path.exists() and not args.regenerate:
                print(f"[{gid:02d}] {variant} cached")
                records.append(PilotRecord(
                    group_id=gid, variant=variant, image_path=str(out_path),
                    prompt=edit_prompt, elapsed_s=0.0, cost_usd=0.0,
                    success=True, error="(cached)",
                ))
                continue
            t0 = time.monotonic()
            try:
                edited = await call_edit(
                    client, src_bytes, edit_prompt,
                    args.model, args.quality, args.size,
                )
                edited.save(out_path)
                elapsed = time.monotonic() - t0
                cost = QUALITY_COST[args.quality]
                cost_state["spent"] += cost
                print(f"[{gid:02d}] {variant} {elapsed:.1f}s "
                      f"spent=${cost_state['spent']:.2f}")
                records.append(PilotRecord(
                    group_id=gid, variant=variant, image_path=str(out_path),
                    prompt=edit_prompt, elapsed_s=elapsed, cost_usd=cost,
                    success=True, error="",
                ))
            except Exception as e:
                print(f"[{gid:02d}] {variant} ERROR: {type(e).__name__}: {str(e)[:150]}")
                records.append(PilotRecord(
                    group_id=gid, variant=variant, image_path="",
                    prompt=edit_prompt, elapsed_s=0.0, cost_usd=0.0,
                    success=False, error=f"{type(e).__name__}: {str(e)[:200]}",
                ))

    # write per-group meta
    with open(g_dir / "meta.json", "w") as f:
        json.dump({
            "group_id": gid, "demographics": cell,
            "records": [asdict(r) for r in records],
        }, f, indent=2)

    return records


async def amain(args):
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("OPENAI_API_KEY not set. source /workspace/.secrets/openai.env")

    import openai
    client = openai.AsyncOpenAI()
    cost_state = {"spent": 0.0}
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.monotonic()
    print(f"[init] {len(GROUPS)} groups × 4 images = {len(GROUPS)*4} total")
    print(f"[init] quality={args.quality} size={args.size} concurrency={args.concurrency}")
    print(f"[init] cost-per-image=${QUALITY_COST[args.quality]:.4f} "
          f"projected total=${QUALITY_COST[args.quality]*len(GROUPS)*4:.2f}")
    print()

    tasks = [run_group(client, c, args, root, cost_state, sem) for c in GROUPS]
    all_records = []
    for coro in asyncio.as_completed(tasks):
        recs = await coro
        all_records.extend(recs)

    elapsed = time.monotonic() - t0
    n_ok = sum(1 for r in all_records if r.success and r.error != "(cached)")
    n_cached = sum(1 for r in all_records if r.error == "(cached)")
    n_fail = sum(1 for r in all_records if not r.success)

    # manifest
    csv_path = root / "pilot_manifest.csv"
    all_records.sort(key=lambda r: (r.group_id, r.variant))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_records[0]).keys()))
        w.writeheader()
        w.writerows([asdict(r) for r in all_records])

    # report
    lines = [
        f"# Diverse-source × API-illusion pilot — {args.quality} quality\n",
        f"- model: {args.model}",
        f"- N groups: {len(GROUPS)} × 4 = {len(GROUPS)*4} target images",
        f"- Generated this run: {n_ok}",
        f"- Cached: {n_cached}",
        f"- Failed: {n_fail}",
        f"- Total cost: ${cost_state['spent']:.4f}",
        f"- Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)\n",
        "## Per-group status\n",
        "| group | demographics | V1 | V2_thatcher | V3_composite | V4_partwhole | total $ |",
        "|---:|---|:-:|:-:|:-:|:-:|---:|",
    ]
    by_group: dict[int, dict[str, PilotRecord]] = {}
    for r in all_records:
        by_group.setdefault(r.group_id, {})[r.variant] = r
    for gid in sorted(by_group):
        recs = by_group[gid]
        cell = next(c for c in GROUPS if c["group_id"] == gid)
        demo = f"{cell['age']} {cell['ethnicity']} {cell['gender']}"
        st = lambda v: "✓" if recs.get(v) and recs[v].success else "✗"
        total = sum(recs.get(v, PilotRecord(0,"","","",0,0,False,"")).cost_usd
                    for v in ["V1_source","V2_thatcher","V3_composite","V4_partwhole"])
        lines.append(f"| {gid} | {demo} | {st('V1_source')} | {st('V2_thatcher')} "
                     f"| {st('V3_composite')} | {st('V4_partwhole')} | ${total:.4f} |")
    lines.append("\n## Failure details (if any)\n")
    for r in all_records:
        if not r.success:
            lines.append(f"- group {r.group_id} {r.variant}: `{r.error}`")
    lines.append("\n## Budget projection (assuming similar pass rate)\n")
    cost_per_group = cost_state['spent'] / max(len(GROUPS), 1) if cost_state['spent'] > 0 else 0
    if cost_per_group > 0:
        for n in [50, 100, 200, 300, 500]:
            lines.append(f"- {n} groups → ${cost_per_group * n:.2f}")
    (root / "pilot_report.md").write_text("\n".join(lines))

    print(f"\n=== DONE ===")
    print(f"  generated: {n_ok}  cached: {n_cached}  failed: {n_fail}")
    print(f"  total cost: ${cost_state['spent']:.4f}")
    print(f"  elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  manifest: {csv_path}")
    print(f"  report:   {root}/pilot_report.md")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default="data/stimuli_gpt_diverse_pilot")
    p.add_argument("--model", default="gpt-image-1")
    p.add_argument("--quality", default="high", choices=list(QUALITY_COST.keys()))
    p.add_argument("--size", default="1024x1024",
                   choices=["1024x1024", "1024x1536", "1536x1024"])
    p.add_argument("--concurrency", type=int, default=3,
                   help="lower default than gen-only pilot since edit() may "
                        "have stricter limits")
    p.add_argument("--regenerate", action="store_true",
                   help="WARNING: re-spends credits even for cached images")
    return p.parse_args()


def main(args):
    asyncio.run(amain(args))


if __name__ == "__main__":
    main(parse_args())
