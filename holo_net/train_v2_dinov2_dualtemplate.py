"""holo_net/train_v2_dinov2_dualtemplate.py — train T_upright + T_inverted
EMA on face samples and their vflip respectively. Reuses v3's data pipeline."""
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

from holo_net.model_v2_dinov2_dualtemplate import (
    HOLONetV2Dinov2Dual, HOLONetV2DinoDualConfig)
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


def train_dual(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    ckpt_path = out_dir / "checkpoint.pt"

    mcfg = HOLONetV2DinoDualConfig()
    model = HOLONetV2Dinov2Dual(mcfg).to(device)
    model.eval()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] dinov2 frozen + T_up + T_inv  trainable={n_trainable/1e6:.4f}M")

    ds = ImageNetForTemplate(cache_dir=args.cache_dir, size=mcfg.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=True, drop_last=True, pin_memory=True,
                    persistent_workers=(args.num_workers > 0),
                    worker_init_fn=_worker_init, collate_fn=collate)
    print(f"[data] {len(ds):,} samples; batch {args.batch_size}; {args.max_steps} steps")

    print(f"\n[dual-template-train] log → {log_path}\n")
    t0 = time.time()
    n_faces_total = 0
    for step, batch in enumerate(dl):
        if step >= args.max_steps:
            break
        x = batch["image"].to(device, non_blocking=True)
        face_mask = batch["face_mask"].to(device, non_blocking=True)
        x_vflip = torch.flip(x, dims=[-2])
        with torch.no_grad():
            afp_spatial = model._extract_patch_tokens(x)
            afp_vflip_spatial = model._extract_patch_tokens(x_vflip)
            info = model.update_ema_dual(afp_spatial, afp_vflip_spatial, face_mask)
            n_faces_total += info["n_face"]

        if step % 20 == 0:
            elapsed = time.time() - t0
            t_up_norm = float(model.face_template_up.template_ema.norm().item())
            t_inv_norm = float(model.face_template_inv.template_ema.norm().item())
            rec = {"step": step,
                   "n_faces_in_batch": info["n_face"],
                   "n_faces_total": n_faces_total,
                   "T_up_norm": t_up_norm,
                   "T_inv_norm": t_inv_norm,
                   "elapsed_s": elapsed,
                   "step_time_s": elapsed / max(1, step)}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {step:4d}  faces {info['n_face']:3d}  total {n_faces_total:5d}  "
                  f"T_up {t_up_norm:7.2f}  T_inv {t_inv_norm:7.2f}  elapsed {elapsed/60:.1f}min")

    with torch.no_grad():
        model.face_template_up.template.copy_(model.face_template_up.template_ema)
        model.face_template_inv.template.copy_(model.face_template_inv.template_ema)

    torch.save({"model": model.state_dict(),
                "config": vars(mcfg),
                "n_faces_total": n_faces_total,
                "args": vars(args)}, ckpt_path)
    print(f"\n[dual-template-train] DONE — {n_faces_total} face samples  "
          f"T_up norm={model.face_template_up.template_ema.norm().item():.2f}  "
          f"T_inv norm={model.face_template_inv.template_ema.norm().item():.2f}")
    print(f"ckpt at {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",
                   default="/workspace/runs/2026-05-23_holonet-v3.0-dualtemplate_seed20260521")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--cache_dir", default="/workspace/.hf_cache")
    p.add_argument("--seed", type=int, default=20260521)
    return p.parse_args()


if __name__ == "__main__":
    train_dual(parse_args())
