"""stimuli/gpt_image_full.py — full-scale GPT-image-1 source generation.

Generates 504 source identities = 36 demographic cells × 14 identities/cell.
High-quality 1024×1024. Async parallelism (5 concurrent), hard cost cap,
exponential backoff on rate-limit errors. Idempotent: rerun safe, skips
already-generated identity files.

Outputs under data/stimuli_gpt/:
  identity_{NNNN}.png + .json (per-identity metadata)
  manifest.csv (full record)

Reads OPENAI_API_KEY from env. Use:
  source /workspace/.secrets/openai.env
  python -m stimuli.gpt_image_full --quality high --max_cost_usd 100
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
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image


QUALITY_COST = {"low": 0.011, "medium": 0.042, "high": 0.167}

GENDERS = ["female", "male"]
ETHNICITIES = ["East Asian", "Black", "White", "South Asian", "Latinx", "Middle Eastern"]
AGES = ["young adult", "middle-aged", "older adult"]
# 2 × 6 × 3 = 36 cells
N_PER_CELL_DEFAULT = 14   # → 504 IDs total

# Subtle prompt variation (intra-cell): hair color/length/style hints so the
# 14 IDs within a cell are not near-twins. Picked at random per identity via
# the seeded RNG (same seed → same prompts always, for replicability).
HAIR_HINTS = [
    "short straight hair", "short wavy hair", "short curly hair",
    "medium-length straight hair", "medium-length wavy hair", "medium-length curly hair",
    "long straight hair", "long wavy hair", "long curly hair",
    "tightly curled hair", "shoulder-length hair", "very short cropped hair",
    "shaved head with stubble", "thinning hair",
]
COMPLEXION_HINTS = [
    "fair complexion", "medium complexion", "olive complexion",
    "tanned complexion", "deep brown complexion", "rich dark complexion",
]
FACE_SHAPE_HINTS = [
    "oval face shape", "round face shape", "square face shape",
    "heart-shaped face", "long face shape", "diamond-shaped face",
]


PROMPT_TEMPLATE = (
    "Professional studio portrait headshot photograph of a {age} {ethnicity} "
    "{gender} adult with {hair} and {complexion}, {face_shape}. "
    "Neutral facial expression, mouth closed, no smile, eyes open and looking "
    "directly at the camera. Plain neutral gray background. Even soft frontal "
    "studio lighting with no harsh shadows. Sharp focus on the face. The person "
    "wears no makeup, no jewelry, no glasses, no hat, no hair covering, plain "
    "neutral dark clothing visible only at the shoulders. Standard passport-"
    "style framing: face fills approximately the central 60% of the frame, "
    "head and neck visible. Photorealistic high-resolution DSLR photograph "
    "quality.\n\n"
    "IMPORTANT: Generate a fully synthetic, non-identifiable face. Do NOT "
    "reproduce the likeness of any real person, public figure, or celebrity."
)


@dataclass
class IdentitySpec:
    identity_id: int
    gender: str
    ethnicity: str
    age: str
    hair: str
    complexion: str
    face_shape: str
    prompt: str


@dataclass
class GenRecord:
    identity_id: int
    gender: str
    ethnicity: str
    age: str
    cell_id: str
    prompt: str
    model: str
    quality: str
    size: str
    elapsed_s: float
    cost_usd: float
    image_path: str
    success: bool
    error: str


def build_identities(n_per_cell: int, seed: int) -> list[IdentitySpec]:
    rng = random.Random(seed)
    ids = []
    next_id = 0
    for gender in GENDERS:
        for ethnicity in ETHNICITIES:
            for age in AGES:
                for _ in range(n_per_cell):
                    hair = rng.choice(HAIR_HINTS)
                    complexion = rng.choice(COMPLEXION_HINTS)
                    face_shape = rng.choice(FACE_SHAPE_HINTS)
                    prompt = PROMPT_TEMPLATE.format(
                        age=age, ethnicity=ethnicity, gender=gender,
                        hair=hair, complexion=complexion, face_shape=face_shape,
                    )
                    ids.append(IdentitySpec(
                        identity_id=next_id, gender=gender, ethnicity=ethnicity,
                        age=age, hair=hair, complexion=complexion,
                        face_shape=face_shape, prompt=prompt,
                    ))
                    next_id += 1
    rng.shuffle(ids)  # interleave cells to spread rate-limit pressure
    # but keep identity_ids in their original (sequential) order even after shuffle
    return ids


async def gen_one(client, spec: IdentitySpec, args, out_root: Path,
                  cost_state: dict, sem: asyncio.Semaphore) -> GenRecord:
    img_path = out_root / f"identity_{spec.identity_id:04d}.png"
    json_path = out_root / f"identity_{spec.identity_id:04d}.json"
    cell_id = f"{spec.gender}_{spec.ethnicity}_{spec.age}".replace(" ", "")

    if img_path.exists() and not args.regenerate:
        rec = GenRecord(
            identity_id=spec.identity_id, gender=spec.gender,
            ethnicity=spec.ethnicity, age=spec.age, cell_id=cell_id,
            prompt=spec.prompt, model=args.model, quality=args.quality,
            size=args.size, elapsed_s=0.0, cost_usd=0.0,
            image_path=str(img_path), success=True, error="(cached)",
        )
        return rec

    # Hard cost cap: refuse new API calls if projected over cap
    cost_per = QUALITY_COST.get(args.quality, 0.0)
    if cost_state["spent"] + cost_per > args.max_cost_usd:
        return GenRecord(
            identity_id=spec.identity_id, gender=spec.gender,
            ethnicity=spec.ethnicity, age=spec.age, cell_id=cell_id,
            prompt=spec.prompt, model=args.model, quality=args.quality,
            size=args.size, elapsed_s=0.0, cost_usd=0.0,
            image_path="", success=False,
            error=f"COST_CAP: ${cost_state['spent']:.2f}+{cost_per:.4f} > ${args.max_cost_usd}",
        )

    async with sem:
        for attempt in range(5):
            t0 = time.monotonic()
            try:
                resp = await client.images.generate(
                    model=args.model, prompt=spec.prompt,
                    size=args.size, quality=args.quality, n=1,
                )
                elapsed = time.monotonic() - t0
                b64 = resp.data[0].b64_json
                img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                img.save(img_path)
                cost_state["spent"] += cost_per
                cost_state["count"] += 1
                rec = GenRecord(
                    identity_id=spec.identity_id, gender=spec.gender,
                    ethnicity=spec.ethnicity, age=spec.age, cell_id=cell_id,
                    prompt=spec.prompt, model=args.model, quality=args.quality,
                    size=args.size, elapsed_s=elapsed, cost_usd=cost_per,
                    image_path=str(img_path), success=True, error="",
                )
                with open(json_path, "w") as f:
                    json.dump(asdict(rec), f, indent=2, default=str)
                print(f"  [{spec.identity_id:04d}] {cell_id:<40s} "
                      f"{elapsed:5.1f}s  spent=${cost_state['spent']:.2f}/"
                      f"${args.max_cost_usd}", flush=True)
                return rec
            except Exception as e:
                msg = str(e)
                wait = 2 ** attempt + random.random()
                # Specific handling for content-policy refusals
                if "content_policy" in msg.lower() or "safety" in msg.lower():
                    print(f"  [{spec.identity_id:04d}] CONTENT_POLICY: {msg[:120]}",
                          flush=True)
                    return GenRecord(
                        identity_id=spec.identity_id, gender=spec.gender,
                        ethnicity=spec.ethnicity, age=spec.age, cell_id=cell_id,
                        prompt=spec.prompt, model=args.model, quality=args.quality,
                        size=args.size, elapsed_s=0.0, cost_usd=0.0,
                        image_path="", success=False,
                        error=f"CONTENT_POLICY: {msg[:200]}",
                    )
                print(f"  [{spec.identity_id:04d}] attempt {attempt+1} err: "
                      f"{type(e).__name__}: {msg[:80]} (wait {wait:.1f}s)",
                      flush=True)
                await asyncio.sleep(wait)
        return GenRecord(
            identity_id=spec.identity_id, gender=spec.gender,
            ethnicity=spec.ethnicity, age=spec.age, cell_id=cell_id,
            prompt=spec.prompt, model=args.model, quality=args.quality,
            size=args.size, elapsed_s=0.0, cost_usd=0.0, image_path="",
            success=False, error="MAX_RETRIES_EXCEEDED",
        )


async def amain(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit(
            "OPENAI_API_KEY not set. Run: source /workspace/.secrets/openai.env"
        )

    import openai
    client = openai.AsyncOpenAI()
    specs = build_identities(args.n_per_cell, args.seed)
    total_n = len(specs)
    cost_per = QUALITY_COST.get(args.quality, 0.0)
    cost_state = {"spent": 0.0, "count": 0}

    # Count already-cached so we can report accurately
    cached = sum(1 for s in specs if (out_root / f"identity_{s.identity_id:04d}.png").exists())
    print(f"[init] model={args.model} quality={args.quality} size={args.size}")
    print(f"[init] total identities = {total_n} ({cached} already cached)")
    print(f"[init] cost-per-image = ${cost_per:.4f}; "
          f"max worst-case = ${cost_per*total_n:.2f}; cost cap = ${args.max_cost_usd}")
    print(f"[init] concurrency = {args.concurrency}; seed = {args.seed}\n")

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.monotonic()
    tasks = [gen_one(client, s, args, out_root, cost_state, sem) for s in specs]
    records: list[GenRecord] = await asyncio.gather(*tasks)
    elapsed_total = time.monotonic() - t0

    # write manifest
    csv_path = out_root / "manifest.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        w.writeheader()
        w.writerows([asdict(r) for r in records])

    n_ok = sum(1 for r in records if r.success and r.error != "(cached)")
    n_cached = sum(1 for r in records if r.error == "(cached)")
    n_err = sum(1 for r in records if not r.success)
    print(f"\n=== DONE ===")
    print(f"  generated this run: {n_ok}  |  cached: {n_cached}  |  failed: {n_err}")
    print(f"  spent: ${cost_state['spent']:.2f}  ({cost_state['count']} new images)")
    print(f"  elapsed: {elapsed_total:.1f}s  ({elapsed_total/60:.1f} min)")
    print(f"  manifest: {csv_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", default="data/stimuli_gpt")
    p.add_argument("--model", default="gpt-image-1")
    p.add_argument("--quality", default="high", choices=list(QUALITY_COST.keys()))
    p.add_argument("--size", default="1024x1024",
                   choices=["1024x1024", "1024x1536", "1536x1024"])
    p.add_argument("--n_per_cell", type=int, default=N_PER_CELL_DEFAULT,
                   help=f"identities per demographic cell (default {N_PER_CELL_DEFAULT} = 504 total)")
    p.add_argument("--seed", type=int, default=20260524)
    p.add_argument("--concurrency", type=int, default=5,
                   help="max in-flight API calls (default 5; OpenAI tier-1 RPM "
                        "is ~5 for images)")
    p.add_argument("--max_cost_usd", type=float, default=100.0,
                   help="hard cost cap; no new API calls past this")
    p.add_argument("--regenerate", action="store_true",
                   help="WARNING: re-spends credits even on cached images")
    return p.parse_args()


def main(args):
    asyncio.run(amain(args))


if __name__ == "__main__":
    main(parse_args())
