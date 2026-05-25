"""stimuli/named_prompt_test.py — does naming "Thatcher illusion" help?

Tests 4 prompt variants on ONE source image (group_07 V1) to see whether
gpt-image-2 understands the concept by name. Output: 4 PNGs side-by-side.
Cost: 4 × $0.167 = $0.67.
"""
from __future__ import annotations
import asyncio, base64, io, os, time
from pathlib import Path
from PIL import Image


PROMPTS = {
    "named_minimal": "Apply the Thatcher illusion to this face.",
    "named_brief": (
        "Apply the classic Thatcher illusion to this photograph: invert the "
        "eyes and the mouth so they appear upside-down, while the rest of the "
        "face remains normal."
    ),
    "named_with_pedigree": (
        "This image needs the Thatcher illusion transformation, first described "
        "by Peter Thompson in 1980. The Thatcher illusion rotates just the eye "
        "region and just the mouth region by 180 degrees in their original "
        "positions on the face, while leaving the rest of the face (forehead, "
        "nose, cheeks, jaw, hair, background) completely unchanged. The "
        "resulting face appears subtly grotesque when viewed upright."
    ),
    "named_visual_target": (
        "Transform this face to apply the Thatcher illusion. The result should "
        "show: (1) the eyebrows positioned BELOW the eyes, (2) the lower "
        "eyelashes positioned ABOVE the pupils, (3) the lower lip on top and "
        "the upper lip on the bottom. Everything else stays the same: same "
        "person, same hair, same background, same lighting, same skin."
    ),
}


async def amain():
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("source /workspace/.secrets/openai.env first")
    import openai
    client = openai.AsyncOpenAI()

    src_path = Path("data/stimuli_gpt_diverse_pilot/group_07/V1_source.png")
    if not src_path.exists():
        raise SystemExit(f"source not found: {src_path.resolve()}")
    with open(src_path, "rb") as f:
        src_bytes = f.read()

    out_dir = Path("data/stimuli_gpt_named_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    async def one(tag: str, prompt: str):
        out = out_dir / f"thatcher_{tag}.png"
        if out.exists():
            print(f"[{tag}] cached")
            return
        t0 = time.monotonic()
        try:
            file_like = io.BytesIO(src_bytes)
            file_like.name = "source.png"
            resp = await client.images.edit(
                model="gpt-image-2", image=file_like, prompt=prompt,
                size="1024x1024", quality="high", n=1,
            )
            elapsed = time.monotonic() - t0
            img = Image.open(io.BytesIO(base64.b64decode(resp.data[0].b64_json))).convert("RGB")
            img.save(out)
            print(f"[{tag}] {elapsed:.1f}s -> {out}")
        except Exception as e:
            print(f"[{tag}] ERROR: {type(e).__name__}: {e}")

    # serial to avoid rate-limit collision with the running pilot
    for tag, prompt in PROMPTS.items():
        await one(tag, prompt)


if __name__ == "__main__":
    asyncio.run(amain())
