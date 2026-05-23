"""holo_net/train_v2_dinov2.py — train ONLY the face template T on top of a
frozen DINOv2 backbone (HOLO-Net v2.1 option-D path, after runs 1-3 of from-
scratch SSL all collapsed).

What this does:
  - Loads frozen DINOv2 ViT-S/14 (no SSL training of backbone).
  - Streams ImageNet-1K via data_v2 (DINO multi-crop globals only suffice for
    template estimation; we use a simpler single-view loader here).
  - For each batch: run MediaPipe face detection, forward DINOv2 frozen,
    update `face_template.template_ema` on face-detected samples.
  - Save the model state (template_ema is the only thing that matters).

Why this is fast:
  - No backbone gradient (frozen).
  - No SSL teacher-student / no DINO loss.
  - Only the EMA buffer updates; the learnable T parameter is kept identical
    to template_ema (we just copy it at the end).
  - Few hundred batches suffice (EMA effective window ~1/(1-decay) ≈ 1000
    face samples; 500 batches × ~43 face samples/batch ≈ 21k face samples).
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from holo_net.model_v2_dinov2 import HOLONetV2Dinov2, HOLONetV2DinoConfig
from holo_net.data_v2 import detect_face, _worker_init, IN_MEAN, IN_STD


# Simple single-view (no multi-crop) transform for template estimation
def make_single_view_transform(size: int = 224):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC,
                 antialias=True),
        T.CenterCrop(size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IN_MEAN, std=IN_STD),
    ])


class ImageNetForTemplate(Dataset):
    """Map-style HF-backed dataset: returns single 224x224 view + face_mask."""

    def __init__(self, cache_dir: str = "/workspace/.hf_cache",
                 size: int = 224):
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_DATASETS_CACHE", f"{cache_dir}/datasets")
        import datasets
        self.ds = datasets.load_dataset("evanarlian/imagenet_1k_resized_256",
                                          split="train")
        self.tf = make_single_view_transform(size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        view = self.tf(img)
        face = detect_face(img)
        return {"image": view, "face_mask": torch.tensor(face, dtype=torch.bool)}


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "face_mask": torch.stack([b["face_mask"] for b in batch]),
    }


def train_template(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    ckpt_path = out_dir / "checkpoint.pt"

    # Model — frozen DINOv2 backbone + learnable T (we keep T learnable for the
    # forward but we won't gradient-train it; we just copy template_ema → template
    # at the end so eval reads either as the canonical face template).
    mcfg = HOLONetV2DinoConfig()
    model = HOLONetV2Dinov2(mcfg).to(device)
    model.eval()  # backbone frozen, no batchnorm running stats to update
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] dinov2_vits14 (frozen) + face_template (0.{n_trainable:06d}M trainable)")

    # DataLoader
    ds = ImageNetForTemplate(cache_dir=args.cache_dir, size=mcfg.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=True, drop_last=True, pin_memory=True,
                    persistent_workers=(args.num_workers > 0),
                    worker_init_fn=_worker_init, collate_fn=collate)
    print(f"[data] {len(ds):,} samples; batch {args.batch_size}; {args.max_steps} steps target")

    # Loop — pure EMA update on T (no gradient)
    print(f"\n[template-train] starting; log → {log_path}\n")
    t0 = time.time()
    n_faces_total = 0
    for step, batch in enumerate(dl):
        if step >= args.max_steps:
            break
        x = batch["image"].to(device, non_blocking=True)
        face_mask = batch["face_mask"].to(device, non_blocking=True)

        with torch.no_grad():
            out = model(x)
            afp_spatial = out["afp_spatial"]
            face_afp = afp_spatial[face_mask]
            model.face_template.update_ema(face_afp)
            n_faces_total += int(face_mask.sum().item())

        if step % 20 == 0:
            elapsed = time.time() - t0
            rec = {"step": step,
                   "n_faces_in_batch": int(face_mask.sum().item()),
                   "n_faces_total": n_faces_total,
                   "template_ema_norm":
                       float(model.face_template.template_ema.norm().item()),
                   "elapsed_s": elapsed,
                   "step_time_s": elapsed / max(1, step)}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {step:4d}  faces_batch {rec['n_faces_in_batch']:3d}  "
                  f"faces_total {n_faces_total:5d}  T_ema {rec['template_ema_norm']:7.2f}  "
                  f"elapsed {elapsed/60:.1f}min")

    # Copy template_ema → template (so eval reads from .template)
    with torch.no_grad():
        model.face_template.template.copy_(model.face_template.template_ema)

    # Save
    torch.save({"model": model.state_dict(),
                "config": vars(mcfg),
                "n_faces_total": n_faces_total,
                "args": vars(args)}, ckpt_path)
    print(f"\n[template-train] DONE — {n_faces_total} face samples seen → T_ema norm "
          f"{model.face_template.template_ema.norm().item():.2f}")
    print(f"checkpoint at {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/workspace/runs/2026-05-23_holonet-v2-dinov2_seed20260521")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=500,
                   help="EMA effective window = 1/(1-decay) ≈ 1000 face samples; "
                        "500 batches × ~43 face samples ≈ 21k face samples = ample")
    p.add_argument("--cache_dir", default="/workspace/.hf_cache")
    p.add_argument("--seed", type=int, default=20260521)
    return p.parse_args()


if __name__ == "__main__":
    train_template(parse_args())
