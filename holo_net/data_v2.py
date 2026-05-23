"""holo_net/data_v2.py — ImageNet-1K DataLoader for HOLO-Net v2.1 (FTPC) SSL.

Reads `evanarlian/imagenet_1k_resized_256` (non-gated, 256-resized, ~70 GB)
from the HF cache (cache redirected to /workspace/.hf_cache; root partition
is too small). For each sample:

  - DINOv2-style multi-crop augmentation: 2 global (224×224, scale 0.32-1.0)
    + N_local local (96×96, scale 0.05-0.32) views of the same source image.
  - MediaPipe face detection on the source image (per-worker lazy init);
    yields a per-sample `face_mask` bool used by:
       (a) the PC self-supervised loss: ‖AFP_spatial − T‖² only on face samples.
       (b) the template EMA update: averaged AFP_spatial over face samples only.

Per-worker MediaPipe is lazy-init'd in `worker_init_fn` to avoid the TF-Lite
fork issue (MediaPipe creates a GL/TF context that doesn't fork cleanly).
"""
from __future__ import annotations
from dataclasses import dataclass
import os

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------
# MediaPipe lazy-init per worker
# --------------------------------------------------------------------------

_MP_FD = None  # set per process / per worker


def _get_mp():
    """Lazy per-process MediaPipe FaceDetection singleton."""
    global _MP_FD
    if _MP_FD is None:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import mediapipe as mp
        _MP_FD = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
    return _MP_FD


def detect_face(img_pil: Image.Image) -> bool:
    """True iff MediaPipe detects ≥1 face."""
    fd = _get_mp()
    arr = np.array(img_pil.convert("RGB"))
    r = fd.process(arr)
    return bool(r.detections)


def _worker_init(worker_id: int):
    """DataLoader worker_init_fn: pre-initialise MediaPipe in the worker
    process so the first batch isn't artificially slow."""
    _get_mp()


# --------------------------------------------------------------------------
# DINOv2-style multi-crop augmentation
# --------------------------------------------------------------------------

# Standard ImageNet normalisation (matches DINOv2)
IN_MEAN = [0.485, 0.456, 0.406]
IN_STD  = [0.229, 0.224, 0.225]


class MultiCropTransform:
    """DINOv2 multi-crop: n_global × global_size + n_local × local_size.

    Augmentations per crop: RandomResizedCrop (different scale ranges for
    global vs local), RandomHorizontalFlip, ColorJitter, RandomGrayscale,
    ToTensor, ImageNet-Normalize.
    """
    def __init__(self, n_global: int = 2, n_local: int = 8,
                 global_size: int = 224, local_size: int = 96,
                 global_scale=(0.32, 1.0), local_scale=(0.05, 0.32)):
        self.n_global, self.n_local = n_global, n_local
        common = [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4,
                          saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IN_MEAN, std=IN_STD),
        ]
        self.global_t = T.Compose([
            T.RandomResizedCrop(global_size, scale=global_scale,
                                interpolation=T.InterpolationMode.BICUBIC),
            *common,
        ])
        self.local_t = T.Compose([
            T.RandomResizedCrop(local_size, scale=local_scale,
                                interpolation=T.InterpolationMode.BICUBIC),
            *common,
        ])

    def __call__(self, img_pil: Image.Image) -> dict:
        return {
            "global": [self.global_t(img_pil) for _ in range(self.n_global)],
            "local":  [self.local_t(img_pil) for _ in range(self.n_local)],
        }


# --------------------------------------------------------------------------
# Dataset + DataLoader
# --------------------------------------------------------------------------

@dataclass
class ImageNetFTPCConfig:
    cache_dir: str = "/workspace/.hf_cache"
    dataset_name: str = "evanarlian/imagenet_1k_resized_256"
    split: str = "train"
    n_global: int = 2
    n_local: int = 8
    global_size: int = 224
    local_size: int = 96
    detect_face: bool = True   # set False to disable MediaPipe (face_mask all False)


class ImageNetFTPCDataset(Dataset):
    """HF map-style dataset that returns multi-crop views + a face_mask bool."""

    def __init__(self, cfg: ImageNetFTPCConfig):
        self.cfg = cfg
        # Redirect HF cache to /workspace (root partition is small).
        os.environ.setdefault("HF_HOME", cfg.cache_dir)
        os.environ.setdefault("HF_DATASETS_CACHE", f"{cfg.cache_dir}/datasets")
        import datasets
        self.ds = datasets.load_dataset(cfg.dataset_name, split=cfg.split)
        self.transform = MultiCropTransform(
            n_global=cfg.n_global, n_local=cfg.n_local,
            global_size=cfg.global_size, local_size=cfg.local_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        views = self.transform(img)
        face = detect_face(img) if self.cfg.detect_face else False
        return {
            "global_views": views["global"],
            "local_views":  views["local"],
            "face_mask":    torch.tensor(face, dtype=torch.bool),
            "label":        torch.tensor(int(ex.get("label", -1)), dtype=torch.long),
        }


def collate_dino(batch):
    """Stack global / local views into per-view tensors of shape (B, 3, H, W).
    Returns:
      global_views: list of n_global tensors, each (B, 3, 224, 224)
      local_views:  list of n_local  tensors, each (B, 3,  96,  96)
      face_mask:    (B,) bool
      labels:       (B,) long
    """
    n_global = len(batch[0]["global_views"])
    n_local  = len(batch[0]["local_views"])
    return {
        "global_views": [torch.stack([b["global_views"][g] for b in batch])
                         for g in range(n_global)],
        "local_views":  [torch.stack([b["local_views"][l] for b in batch])
                         for l in range(n_local)],
        "face_mask":    torch.stack([b["face_mask"] for b in batch]),
        "labels":       torch.stack([b["label"]     for b in batch]),
    }


def make_dataloader(cfg: ImageNetFTPCConfig, batch_size: int = 256,
                    num_workers: int = 16, **kwargs) -> DataLoader:
    ds = ImageNetFTPCDataset(cfg)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, drop_last=True, pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init if cfg.detect_face else None,
        collate_fn=collate_dino, **kwargs,
    )


# --------------------------------------------------------------------------
# CLI sanity check
# --------------------------------------------------------------------------

def _sanity_check():
    """Load one batch, print shapes, verify face_mask makes sense."""
    cfg = ImageNetFTPCConfig(detect_face=True)
    print(f"config: {cfg}")
    dl = make_dataloader(cfg, batch_size=64, num_workers=4)
    print(f"dataset size: {len(dl.dataset):,} samples")
    print("loading 2 batches...")
    it = iter(dl)
    for b_idx in range(2):
        batch = next(it)
        print(f"\n--- batch {b_idx} ---")
        print(f"global_views: {len(batch['global_views'])} tensors, "
              f"shape {tuple(batch['global_views'][0].shape)}")
        print(f"local_views:  {len(batch['local_views'])} tensors, "
              f"shape {tuple(batch['local_views'][0].shape)}")
        n_face = int(batch["face_mask"].sum())
        B = batch["face_mask"].shape[0]
        print(f"face_mask: {n_face}/{B} = {n_face/B:.1%} face-detected")
        print(f"labels[:8]: {batch['labels'][:8].tolist()}")
    print("\n=== data_v2 sanity OK ===")


if __name__ == "__main__":
    _sanity_check()
