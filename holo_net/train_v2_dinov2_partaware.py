"""holo_net/train_v2_dinov2_partaware.py — train ONLY the K=4 part-aware
face templates on top of a frozen DINOv2 backbone (HOLO-Net v2.2, Gen 1
variant 1 of the iteration ladder).

Mirrors train_v2_dinov2.py except update_ema_for_faces splits AFP into
the 4 part regions per PART_REGIONS.
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

from holo_net.model_v2_dinov2_partaware import (
    HOLONetV2Dinov2PartAware, HOLONetV2DinoPartAwareConfig)
from holo_net.data_v2 import detect_face, _worker_init, IN_MEAN, IN_STD


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

    mcfg = HOLONetV2DinoPartAwareConfig()
    model = HOLONetV2Dinov2PartAware(mcfg).to(device)
    model.eval()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] dinov2_vits14 frozen + {len(mcfg.parts)} part templates  "
          f"trainable={n_trainable/1e6:.4f}M")

    ds = ImageNetForTemplate(cache_dir=args.cache_dir, size=mcfg.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=True, drop_last=True, pin_memory=True,
                    persistent_workers=(args.num_workers > 0),
                    worker_init_fn=_worker_init, collate_fn=collate)
    print(f"[data] {len(ds):,} samples; batch {args.batch_size}; "
          f"{args.max_steps} steps target")

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
            info = model.update_ema_for_faces(out["afp_spatial"], face_mask)
            n_faces_total += info.get("n_face_total", 0)

        if step % 20 == 0:
            elapsed = time.time() - t0
            t_ema_norms = {
                k: float(model.part_templates[k].template_ema.norm().item())
                for k in mcfg.parts
            }
            rec = {"step": step,
                   "n_faces_in_batch": info.get("n_face_total", 0),
                   "n_faces_total": n_faces_total,
                   "T_ema_norms": t_ema_norms,
                   "elapsed_s": elapsed,
                   "step_time_s": elapsed / max(1, step)}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            t_ema_str = " ".join(f"{k}={v:.1f}" for k, v in t_ema_norms.items())
            print(f"step {step:4d}  faces {info.get('n_face_total',0):3d}  "
                  f"total {n_faces_total:5d}  T_ema [{t_ema_str}]  "
                  f"elapsed {elapsed/60:.1f}min")

    # Copy template_ema → template (for eval)
    with torch.no_grad():
        for k in mcfg.parts:
            model.part_templates[k].template.copy_(
                model.part_templates[k].template_ema)

    torch.save({"model": model.state_dict(),
                "config": vars(mcfg),
                "n_faces_total": n_faces_total,
                "args": vars(args)}, ckpt_path)
    final_norms = {k: float(model.part_templates[k].template_ema.norm().item())
                   for k in mcfg.parts}
    print(f"\n[template-train] DONE — {n_faces_total} face samples  "
          f"T_ema final norms: {final_norms}")
    print(f"checkpoint at {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",
                   default="/workspace/runs/2026-05-23_holonet-v2.2-partaware_seed20260521")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--cache_dir", default="/workspace/.hf_cache")
    p.add_argument("--seed", type=int, default=20260521)
    return p.parse_args()


if __name__ == "__main__":
    train_template(parse_args())
