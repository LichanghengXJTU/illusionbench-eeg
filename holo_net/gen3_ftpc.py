"""holo_net/gen3_ftpc.py — Gen 3 backbone + FTPC global face template head.

Combines the v3 FTPC mechanism (face template learned via EMA on ImageNet face
samples) with the Gen 3 from-scratch backbone (which has orient-aux training,
NO vflip aug). The hope: Gen 3's PWI improvement (E054: 0.397→0.145) +
FTPC's δ readout could push some §6 criteria over threshold.

Usage:
  # Train template:
  python -m holo_net.gen3_ftpc train \
      --gen3_ckpt /workspace/runs/2026-05-24_gen3_v4.0_seed20260521/checkpoint.pt \
      --output_dir /workspace/runs/2026-05-24_gen3_v4.0_ftpc_seed20260521 \
      --max_steps 500

  # Eval:
  python -m holo_net.gen3_ftpc eval \
      --ftpc_ckpt /workspace/runs/2026-05-24_gen3_v4.0_ftpc_seed20260521/checkpoint.pt \
      --tag HOLONET_v4.0_gen3_ftpc
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from holo_net.gen3_model import Gen3Backbone, Gen3ModelConfig


# ---- Constants ----

IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)

PARADIGMS = {
    "thatcher":   ("data/stimuli_ffhq/thatcher_manifest.csv",            1.000, "ISI"),
    "composite":  ("data/stimuli_ffhq_composite/thatcher_manifest.csv",  1.099, "CSI"),
    "partwhole":  ("data/stimuli_ffhq_partwhole/thatcher_manifest.csv",  1.202, "PWI"),
    "randombbox": ("data/stimuli_ffhq_randombbox/thatcher_manifest.csv", 1.000, "ISIrbox"),
}
CRITERIA = {"thatcher": (">=", 3.0), "composite": (">=", 1.5),
            "partwhole": ("<=", 0.5), "randombbox": ("<=", 1.5)}


# ---- Model: Gen3 backbone + FTPC face template ----

class FaceTemplate(nn.Module):
    """Same as v3's: learnable + EMA buffer."""
    def __init__(self, dim: int = 384, spatial: int = 16,
                 init_std: float = 0.02, ema_decay: float = 0.999):
        super().__init__()
        self.dim, self.spatial, self.ema_decay = dim, spatial, ema_decay
        self.template = nn.Parameter(torch.zeros(dim, spatial, spatial))
        nn.init.normal_(self.template, std=init_std)
        self.register_buffer("template_ema", torch.zeros(dim, spatial, spatial))
        self.register_buffer("ema_initialised", torch.tensor(False))

    @torch.no_grad()
    def update_ema(self, face_afp_batch: torch.Tensor) -> None:
        if face_afp_batch.numel() == 0:
            return
        batch_mean = face_afp_batch.mean(dim=0)
        if not bool(self.ema_initialised):
            self.template_ema.copy_(batch_mean)
            self.ema_initialised.fill_(True)
        else:
            self.template_ema.mul_(self.ema_decay).add_(
                batch_mean, alpha=1.0 - self.ema_decay)


class Gen3FTPC(nn.Module):
    """Frozen Gen3 backbone + learnable face template."""

    def __init__(self, gen3_cfg: Gen3ModelConfig | None = None):
        super().__init__()
        if gen3_cfg is None:
            gen3_cfg = Gen3ModelConfig()
        self.gen3_cfg = gen3_cfg
        self.backbone = Gen3Backbone(gen3_cfg)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.face_template = FaceTemplate(
            dim=gen3_cfg.backbone_dim, spatial=gen3_cfg.img_size // gen3_cfg.patch_size,
        )

    def load_gen3_weights(self, gen3_ckpt_path: str, which: str = "student"):
        ckpt = torch.load(gen3_ckpt_path, map_location="cpu", weights_only=False)
        # The Gen3Backbone in our wrapper IS the same module as the student/teacher
        # in gen3_train; load matching state.
        state = ckpt[which]
        miss, unexp = self.backbone.load_state_dict(state, strict=True)
        print(f"  [gen3 load] {which}: missing={len(miss)} unexpected={len(unexp)}  "
              f"step={ckpt.get('step', '?')}  epoch={ckpt.get('epoch', '?')}")

    @torch.no_grad()
    def _extract_afp_spatial(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x, mode="features")
        return out["patch_spatial"]   # (B, D, H, W)

    def forward(self, x: torch.Tensor) -> dict:
        afp_spatial = self._extract_afp_spatial(x)
        afp_pooled = F.adaptive_avg_pool2d(afp_spatial, 1).flatten(1)
        T = self.face_template.template
        delta = afp_spatial - T[None]
        delta_pooled = F.adaptive_avg_pool2d(delta, 1).flatten(1)
        return {
            "afp_spatial": afp_spatial,
            "afp_pooled": afp_pooled,
            "afp": afp_pooled, "atl": afp_pooled,
            "delta": delta,
            "delta_pooled": delta_pooled,
            "ffa": delta_pooled,
            "v1": afp_pooled, "v2": afp_pooled, "v4": afp_pooled, "mfp": afp_pooled,
        }


# ---- Training: template-only EMA over face samples ----

def make_single_view_transform(size: int = 224):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IN_MEAN, std=IN_STD),
    ])


class ImageNetForFTPC(Dataset):
    def __init__(self, cache_dir: str = "/workspace/.hf_cache",
                 size: int = 224):
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_DATASETS_CACHE", f"{cache_dir}/datasets")
        import datasets
        self.ds = datasets.load_dataset("evanarlian/imagenet_1k_resized_256",
                                          split="train")
        self.tf = make_single_view_transform(size)
        from holo_net.data_v2 import detect_face
        self.detect_face = detect_face

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        view = self.tf(img)
        face = self.detect_face(img)
        return {"image": view,
                "face_mask": torch.tensor(face, dtype=torch.bool)}


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "face_mask": torch.stack([b["face_mask"] for b in batch]),
    }


def train_template(args):
    from holo_net.data_v2 import _worker_init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    ckpt_path = out_dir / "checkpoint.pt"

    gen3_cfg = Gen3ModelConfig()
    model = Gen3FTPC(gen3_cfg).to(device)
    model.load_gen3_weights(args.gen3_ckpt, which=args.which_weights)
    model.eval()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Gen3 (frozen) + FaceTemplate; trainable={n_trainable/1e6:.4f}M")

    ds = ImageNetForFTPC(cache_dir=args.cache_dir, size=gen3_cfg.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=True, drop_last=True, pin_memory=True,
                    persistent_workers=(args.num_workers > 0),
                    worker_init_fn=_worker_init, collate_fn=collate)
    print(f"[data] {len(ds):,} samples; batch {args.batch_size}; {args.max_steps} steps")

    print(f"\n[ftpc-train] log → {log_path}\n")
    t0 = time.time()
    n_faces = 0
    for step, batch in enumerate(dl):
        if step >= args.max_steps:
            break
        x = batch["image"].to(device, non_blocking=True)
        face_mask = batch["face_mask"].to(device, non_blocking=True)
        with torch.no_grad():
            afp_spatial = model._extract_afp_spatial(x)
            face_afp = afp_spatial[face_mask]
            model.face_template.update_ema(face_afp.detach())
            n_faces += int(face_mask.sum().item())

        if step % 20 == 0:
            elapsed = time.time() - t0
            t_norm = float(model.face_template.template_ema.norm().item())
            rec = {"step": step, "n_faces_batch": int(face_mask.sum().item()),
                   "n_faces_total": n_faces, "T_ema_norm": t_norm,
                   "elapsed_s": elapsed}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {step:4d}  faces {int(face_mask.sum().item()):3d}  "
                  f"total {n_faces:5d}  T_ema {t_norm:7.2f}  elap {elapsed/60:.1f}min")

    with torch.no_grad():
        model.face_template.template.copy_(model.face_template.template_ema)

    torch.save({"model": model.state_dict(),
                "gen3_cfg": vars(gen3_cfg),
                "n_faces_total": n_faces,
                "gen3_ckpt": args.gen3_ckpt,
                "which_weights": args.which_weights,
                "args": vars(args)}, ckpt_path)
    print(f"\n[ftpc-train] DONE — {n_faces} face samples  "
          f"T_ema norm {model.face_template.template_ema.norm().item():.2f}")
    print(f"ckpt at {ckpt_path}")


# ---- Eval: IllusionBench §6 ----

def make_eval_transform(size: int = 224):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IN_MEAN, std=IN_STD),
    ])


def run_compute_metrics(emb_dir: Path, manifest: Path, output_csv: Path,
                          repo: Path) -> None:
    cmd = [sys.executable, "-m", "analysis.compute_metrics",
           "--embedding_dir", str(emb_dir),
           "--manifest", str(manifest),
           "--output", str(output_csv)]
    subprocess.run(cmd, check=True, cwd=str(repo),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@torch.no_grad()
def eval_falsification(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = Path(args.repo_root)

    ckpt = torch.load(args.ftpc_ckpt, map_location=device, weights_only=False)
    gen3_cfg = Gen3ModelConfig()
    model = Gen3FTPC(gen3_cfg).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"[load] ftpc ckpt; n_faces_total={ckpt.get('n_faces_total', '?')}  "
          f"T_norm={model.face_template.template.norm().item():.2f}")

    tf = make_eval_transform()
    per_para_per_layer = {}
    for para, (manifest_rel, pixbase, idx_name) in PARADIGMS.items():
        manifest = repo / manifest_rel
        df = pd.read_csv(manifest).sort_values("stim_id").reset_index(drop=True)
        stim_ids = df["stim_id"].to_numpy()
        print(f"\n[{para}] {len(df)} stims")
        imgs = [tf(Image.open(p).convert("RGB"))
                for p in tqdm(df["path"], desc=f"{para}", leave=False)]
        bs = args.batch_size
        feats = {"afp": [], "ffa": []}
        for i in tqdm(range(0, len(imgs), bs), desc="fwd", leave=False):
            x = torch.stack(imgs[i:i+bs]).to(device)
            out = model(x)
            feats["afp"].append(out["afp_pooled"].cpu())
            feats["ffa"].append(out["ffa"].cpu())
        emb_dir = repo / "outputs/embeddings" / f"{para}_{args.tag}"
        emb_dir.mkdir(parents=True, exist_ok=True)
        for layer, fs in feats.items():
            emb = torch.cat(fs, dim=0)
            emb_n = F.normalize(emb, p=2, dim=1).numpy().astype(np.float32)
            model_id = f"{args.tag}_{layer}"
            np.savez(emb_dir / f"{model_id}.npz",
                     stim_ids=stim_ids, embeddings=emb_n,
                     model_id=model_id, output_dim=int(emb_n.shape[1]))
        table = repo / "outputs/tables" / f"{args.tag}_{para}.csv"
        run_compute_metrics(emb_dir, manifest, table, repo)
        df_t = pd.read_csv(table)
        per_layer = {}
        for _, row in df_t.iterrows():
            layer = str(row["model_id"]).rsplit("_", 1)[-1]
            per_layer[layer] = float(row["isi_pert"]) / pixbase
        per_para_per_layer[para] = per_layer

    print(f"\n=== Gen 3 + FTPC §6 (pixel-corrected) ===  tag={args.tag}")
    header = "layer    " + "".join(f"{PARADIGMS[p][2]:>11}" for p in PARADIGMS) + "  pass/4"
    print(header)
    print("-" * len(header))
    for layer in ("afp", "ffa"):
        line = f"{layer:8s} "
        n_pass = 0
        for p in PARADIGMS:
            v = per_para_per_layer[p].get(layer, float("nan"))
            line += f"{v:11.3f}"
            op, thr = CRITERIA[p]
            ok = (v >= thr) if op == ">=" else (v <= thr)
            if ok:
                n_pass += 1
        line += f"   {n_pass}/4"
        print(line)


# ---- CLI ----

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "eval"])
    p.add_argument("--gen3_ckpt", default="")
    p.add_argument("--ftpc_ckpt", default="")
    p.add_argument("--output_dir", default="/workspace/runs/2026-05-24_gen3_v4.0_ftpc_seed20260521")
    p.add_argument("--tag", default="HOLONET_v4.0_gen3_ftpc")
    p.add_argument("--which_weights", default="student", choices=["student", "teacher"])
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--cache_dir", default="/workspace/.hf_cache")
    p.add_argument("--repo_root", default="/workspace/illusionbench-eeg")
    p.add_argument("--seed", type=int, default=20260521)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "train":
        assert args.gen3_ckpt, "--gen3_ckpt required for train"
        train_template(args)
    elif args.cmd == "eval":
        assert args.ftpc_ckpt, "--ftpc_ckpt required for eval"
        # Reuse parse_args batch_size for eval
        args.batch_size = 64 if args.batch_size > 64 else args.batch_size
        eval_falsification(args)
