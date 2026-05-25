"""stimuli/gpt_image_masked_thatcher.py — mask-guided Thatcher edit test.

Takes ONE source face (V1) → extracts eye + mouth polygon mask via MediaPipe
FaceMesh → calls OpenAI gpt-image-2 images.edit() with that mask + a NAMED
"Thatcher illusion" prompt.

OpenAI mask convention: PNG with alpha channel where
  alpha = 0   → editable region (replaced)
  alpha = 255 → preserved region (kept exactly)

We make the eye + brow + mouth regions transparent → model re-renders them.
The named-prompt experiment (named_minimal.png) showed the model RECOGNIZES
Thatcher by name but produces duplicate / extra features. Adding a mask
should constrain it to only the intended regions, preventing duplication.

Outputs:
  data/stimuli_gpt_masked_test/mask.png         (the alpha mask itself, for QC)
  data/stimuli_gpt_masked_test/mask_overlay.png (mask overlaid on source, for QC)
  data/stimuli_gpt_masked_test/result.png       (the edit result)
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def build_mask_heuristic(rgb: np.ndarray, feather_px: int) -> np.ndarray:
    """Heuristic eye+mouth mask for centered face portraits at ~1024×1024.

    Avoids MediaPipe entirely (env compat issue on the server's venv).
    The 10-group pilot's V1 sources are all GPT-generated centered portraits
    with consistent framing, so heuristic ellipses at standard face proportions
    work robustly. Used here just to test whether mask + named prompt unlocks
    clean Thatcher; once we confirm the technique works, switch to proper
    MediaPipe landmarks in a separate venv (or load v1 via opencv haar
    cascades).
    """
    h, w = rgb.shape[:2]
    bin_mask = np.zeros((h, w), dtype=np.uint8)

    # Eye ellipses: centered horizontally at ~36% and ~64% width,
    # vertically at ~40% height; size ~16% width × 8% height.
    le_cx = int(0.36 * w); le_cy = int(0.40 * h)
    re_cx = int(0.64 * w); re_cy = int(0.40 * h)
    eye_a = int(0.10 * w); eye_b = int(0.06 * h)   # semi-axes
    cv2.ellipse(bin_mask, (le_cx, le_cy), (eye_a, eye_b),
                angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
    cv2.ellipse(bin_mask, (re_cx, re_cy), (eye_a, eye_b),
                angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

    # Mouth ellipse: centered at ~50% width, ~70% height; ~13% × 5%.
    mo_cx = int(0.50 * w); mo_cy = int(0.70 * h)
    mo_a = int(0.10 * w); mo_b = int(0.045 * h)
    cv2.ellipse(bin_mask, (mo_cx, mo_cy), (mo_a, mo_b),
                angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)

    if feather_px > 0:
        bin_mask = cv2.GaussianBlur(bin_mask, (0, 0),
                                     sigmaX=feather_px, sigmaY=feather_px)

    # OpenAI: alpha 0 = edit, alpha 255 = preserve.
    alpha = 255 - bin_mask
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = alpha
    return rgba


async def amain(args):
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("source /workspace/.secrets/openai.env first")
    import openai

    src_path = Path(args.source)
    if not src_path.exists():
        raise SystemExit(f"source not found: {src_path.resolve()}")
    src_img = Image.open(src_path).convert("RGB")
    src_rgb = np.array(src_img)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mask] building heuristic mask for {src_path}")
    rgba = build_mask_heuristic(src_rgb, feather_px=args.feather_px)
    mask_path = out_dir / "mask.png"
    Image.fromarray(rgba).save(mask_path)

    # Sanity overlay: red tint on the editable region (alpha < 128) for QC
    overlay = src_rgb.copy()
    edit_region = rgba[..., 3] < 128
    overlay[edit_region] = (
        (overlay[edit_region].astype(np.float32) * 0.4 +
         np.array([255, 0, 0], dtype=np.float32) * 0.6).astype(np.uint8)
    )
    Image.fromarray(overlay).save(out_dir / "mask_overlay.png")
    print(f"[mask] wrote {mask_path} and mask_overlay.png "
          f"(edit area = {edit_region.mean()*100:.1f}% of frame)")

    # Call the edit API.
    client = openai.AsyncOpenAI()
    src_buf = io.BytesIO()
    src_img.save(src_buf, format="PNG")
    src_buf.seek(0); src_buf.name = "source.png"
    mask_buf = io.BytesIO()
    Image.fromarray(rgba).save(mask_buf, format="PNG")
    mask_buf.seek(0); mask_buf.name = "mask.png"

    print(f"[edit] calling gpt-image-2 with mask + prompt='{args.prompt}'")
    t0 = time.monotonic()
    try:
        resp = await client.images.edit(
            model=args.model,
            image=src_buf,
            mask=mask_buf,
            prompt=args.prompt,
            size="1024x1024",
            quality=args.quality,
            n=1,
        )
        elapsed = time.monotonic() - t0
        img = Image.open(io.BytesIO(base64.b64decode(resp.data[0].b64_json))).convert("RGB")
        out_path = out_dir / "result.png"
        img.save(out_path)
        print(f"[edit] OK {elapsed:.1f}s -> {out_path}")
    except Exception as e:
        print(f"[edit] ERROR {type(e).__name__}: {e}")
        raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",
                   default="data/stimuli_gpt_diverse_pilot/group_07/V1_source.png")
    p.add_argument("--output_dir", default="data/stimuli_gpt_masked_test")
    p.add_argument("--prompt",
                   default="Apply the Thatcher illusion: invert the eyes and the mouth.")
    p.add_argument("--model", default="gpt-image-2")
    p.add_argument("--quality", default="high")
    p.add_argument("--dilate_px", type=int, default=8)
    p.add_argument("--feather_px", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
