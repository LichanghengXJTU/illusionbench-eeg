"""eeg_decoder/extract_dinov2_things.py — extract frozen DINOv2 ViT-S/14
features on the THINGS-EEG2 stimulus images, aligned to the canonical
ATM/EEG index.

Output: tensor of shape (n_stim, 384) saved to
  data/things_features/dinov2_vits14_train.pt   shape (16540, 384)
  data/things_features/dinov2_vits14_test.pt    shape (  200, 384)

The index follows the ATM `image_metadata.npy`:
  train: nested loop over 1654 concepts × 10 images = 16540, alphabetical
  test : 200 concepts × 1 image = 200

These align with `ATM_S_eeg_features_sub-XX_train.pt` reshape:
  (66160, 1024) → (16540, 4 reps, 1024)
and `_test.pt` (200, 1024) directly.

Run from the repo root:
  cd /workspace/illusionbench-eeg
  TORCH_HOME=/workspace/.torch_cache python -m eeg_decoder.extract_dinov2_things \
      --images_root /workspace/illusionbench-eeg/data/things_images/images_set \
      --output_dir  /workspace/illusionbench-eeg/data/things_features \
      --batch_size  128
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)


def make_eval_transform(size: int = 224):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC,
                 antialias=True),
        T.CenterCrop(size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IN_MEAN, std=IN_STD),
    ])


class ThingsImageDataset(Dataset):
    """Read THINGS images in the canonical ATM index order.

    Args:
      images_root: path to `images_set/` (must contain training_images/,
                                            test_images/, image_metadata.npy)
      split: 'train' or 'test'
    """

    def __init__(self, images_root: Path, split: str):
        self.images_root = Path(images_root)
        self.split = split
        meta = np.load(self.images_root / "image_metadata.npy",
                        allow_pickle=True).item()
        if split == "train":
            self.concepts = meta["train_img_concepts"]
            self.files = meta["train_img_files"]
            self.subdir = "training_images"
        elif split == "test":
            self.concepts = meta["test_img_concepts"]
            self.files = meta["test_img_files"]
            self.subdir = "test_images"
        else:
            raise ValueError(f"unknown split: {split}")
        assert len(self.concepts) == len(self.files)
        self.tf = make_eval_transform()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.images_root / self.subdir / self.concepts[idx] / self.files[idx]
        img = Image.open(p).convert("RGB")
        return self.tf(img)


def _load_dinov2(torch_cache: str = "/workspace/.torch_cache") -> torch.nn.Module:
    os.environ.setdefault("TORCH_HOME", torch_cache)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                            trust_repo=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_split(images_root: Path, split: str, output_dir: Path,
                    batch_size: int, num_workers: int,
                    pooling: str = "mean") -> Path:
    """Run frozen DINOv2 on a split; save (n_stim, 384) tensor.

    Pooling options:
      'mean' — mean-pool patch tokens (matches eval_extract_dinov2.afp_pooled
               used in tick-80 IllusionBench eval)
      'cls'  — use the DINOv2 CLS token

    'mean' is recommended for compatibility with the IllusionBench eval at FFA.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    model = _load_dinov2().to(device)
    ds = ThingsImageDataset(images_root, split)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                    shuffle=False, drop_last=False, pin_memory=(device == "cuda"))
    print(f"[data] split={split} n={len(ds)} batch={batch_size}")

    feats = []
    t0 = time.time()
    for batch in tqdm(dl, desc=f"DINOv2 {split}"):
        x = batch.to(device, non_blocking=True)
        out = model.forward_features(x)
        patch = out["x_norm_patchtokens"]    # (B, N, D)
        if pooling == "mean":
            f = patch.mean(dim=1)             # (B, D)
        elif pooling == "cls":
            f = out["x_norm_clstoken"]       # (B, D)
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        feats.append(f.float().cpu())
    elapsed = time.time() - t0

    feats = torch.cat(feats, dim=0)
    assert feats.shape == (len(ds), 384), feats.shape
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"dinov2_vits14_{split}.pt"
    torch.save(feats, out_path)
    print(f"[save] {out_path}  shape={tuple(feats.shape)}  "
          f"({elapsed:.1f}s, {elapsed/len(ds)*1000:.2f}ms/img)")
    return out_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_root",
                   default="/workspace/illusionbench-eeg/data/things_images/images_set")
    p.add_argument("--output_dir",
                   default="/workspace/illusionbench-eeg/data/things_features")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    p.add_argument("--splits", default="test,train",
                   help="comma-separated; 'test' first since it's fast")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        extract_split(Path(args.images_root), split, Path(args.output_dir),
                       args.batch_size, args.num_workers, args.pooling)
    print("\n[done]")
