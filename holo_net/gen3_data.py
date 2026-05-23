"""holo_net/gen3_data.py — Gen 3 dataloader for from-scratch HOLO-Net
backbone retrain.

Key design choices (vs prior data_v2.py):
  1. NO vertical flip in any augmentation — this is THE lever for orientation
     bias. Vflip would teach the model orientation-invariance, which is
     exactly what we DON'T want.
  2. Multi-crop DINOv2-style: 2 global 224 views + 8 local 96 views.
  3. Orientation auxiliary labels:
     - With prob `aux_invert_prob` (default 0.5), the orientation-aux head
       sees a vflipped version of the image with label 1 (inverted)
     - Otherwise sees the upright version with label 0 (upright)
     - This is a SEPARATE forward (orient-aux view), distinct from the SSL
       views used in DINO loss

Per CLAUDE.md / user directive: training data is natural images (ImageNet-1K
+ optional face addon). NO illusion stimuli, NO FFHQ. The face-mask
detection (MediaPipe per-worker) is preserved as in data_v2 for downstream
template/eval pipeline compatibility.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset


IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)


@dataclass
class Gen3DataConfig:
    cache_dir: str = "/workspace/.hf_cache"
    image_size_global: int = 224
    image_size_local: int = 96
    n_global: int = 2
    n_local: int = 8
    global_scale_min: float = 0.32
    global_scale_max: float = 1.0
    local_scale_min: float = 0.05
    local_scale_max: float = 0.32
    # Orientation aux
    aux_invert_prob: float = 0.5         # P(orient_label = 1)
    # MediaPipe face mask (kept for downstream compat, not used in loss here)
    enable_face_detection: bool = False  # OFF for from-scratch SSL (faster)


# ---- Per-worker MediaPipe lazy init -- preserved for future v compat ----
_face_detector = None


def detect_face(img_pil) -> bool:
    """Best-effort face detection; returns False if MediaPipe unavailable."""
    global _face_detector
    if _face_detector is None:
        return False
    try:
        rgb = np.array(img_pil.convert("RGB"))
        res = _face_detector.process(rgb)
        return bool(res.detections)
    except Exception:
        return False


def _worker_init(worker_id):
    """Optional per-worker MediaPipe init.  Off by default for Gen 3."""
    pass


# ---- Transform builders (NO vflip anywhere) ----

def _norm():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IN_MEAN, std=IN_STD),
    ])


def build_global_view_transform(cfg: Gen3DataConfig):
    """Single global 224 view.

    Augmentation: random resized crop + horizontal flip (NO vflip) +
    color jitter (light) + gaussian blur.
    """
    return T.Compose([
        T.RandomResizedCrop(
            cfg.image_size_global,
            scale=(cfg.global_scale_min, cfg.global_scale_max),
            interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(p=0.5),         # horizontal OK; preserves upright
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        _norm(),
    ])


def build_local_view_transform(cfg: Gen3DataConfig):
    """Single local 96 view."""
    return T.Compose([
        T.RandomResizedCrop(
            cfg.image_size_local,
            scale=(cfg.local_scale_min, cfg.local_scale_max),
            interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.2, 0.1),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
        _norm(),
    ])


def build_orient_aux_transform(cfg: Gen3DataConfig):
    """Orientation aux view: light augmentation (no random crop scale variation
    so the global structure is preserved) + ALWAYS upright, NO horiz flip
    (to avoid confounding orient-aux with hflip)."""
    return T.Compose([
        T.Resize(cfg.image_size_global, interpolation=T.InterpolationMode.BICUBIC,
                 antialias=True),
        T.CenterCrop(cfg.image_size_global),
        T.ColorJitter(0.2, 0.2, 0.1, 0.05),
        _norm(),
    ])


class MultiCropWithOrientAux:
    """Returns dict with:
      'global_views' : list of n_global tensors (3, 224, 224)
      'local_views'  : list of n_local tensors (3, 96, 96)
      'orient_view'  : tensor (3, 224, 224) — the image with potential vflip
      'orient_label' : 0 (upright) or 1 (inverted)
    """
    def __init__(self, cfg: Gen3DataConfig):
        self.cfg = cfg
        self.tf_global = build_global_view_transform(cfg)
        self.tf_local = build_local_view_transform(cfg)
        self.tf_orient = build_orient_aux_transform(cfg)

    def __call__(self, img):
        out = {}
        out["global_views"] = [self.tf_global(img) for _ in range(self.cfg.n_global)]
        out["local_views"] = [self.tf_local(img) for _ in range(self.cfg.n_local)]
        # Orient aux
        orient_view = self.tf_orient(img)
        if torch.rand(1).item() < self.cfg.aux_invert_prob:
            orient_view = torch.flip(orient_view, dims=[-2])
            out["orient_label"] = 1
        else:
            out["orient_label"] = 0
        out["orient_view"] = orient_view
        return out


# ---- Dataset wrapper ----

class ImageNet1KGen3(Dataset):
    """HuggingFace evanarlian/imagenet_1k_resized_256, with MultiCropWithOrientAux."""

    def __init__(self, cfg: Gen3DataConfig):
        self.cfg = cfg
        os.environ.setdefault("HF_HOME", cfg.cache_dir)
        os.environ.setdefault("HF_DATASETS_CACHE", f"{cfg.cache_dir}/datasets")
        import datasets
        self.ds = datasets.load_dataset("evanarlian/imagenet_1k_resized_256",
                                          split="train")
        self.transform = MultiCropWithOrientAux(cfg)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        return self.transform(img)


def collate(batch: list[dict]) -> dict:
    """Stack multi-crop views + orient aux."""
    n_global = len(batch[0]["global_views"])
    n_local = len(batch[0]["local_views"])
    out = {
        "global_views": [torch.stack([b["global_views"][k] for b in batch])
                          for k in range(n_global)],
        "local_views": [torch.stack([b["local_views"][k] for b in batch])
                         for k in range(n_local)],
        "orient_view": torch.stack([b["orient_view"] for b in batch]),
        "orient_label": torch.tensor([b["orient_label"] for b in batch],
                                       dtype=torch.long),
    }
    return out


def _sanity_check():
    cfg = Gen3DataConfig(cache_dir=os.path.expanduser("~/.cache/hf_local"))
    print(f"global views: {cfg.n_global} × {cfg.image_size_global}")
    print(f"local views : {cfg.n_local} × {cfg.image_size_local}")
    print(f"orient invert prob: {cfg.aux_invert_prob}")

    # Test transforms on a dummy PIL image
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (256, 256), color=(120, 80, 40))
    t = MultiCropWithOrientAux(cfg)
    out = t(img)
    print(f"orient_label: {out['orient_label']}")
    print(f"orient_view shape: {tuple(out['orient_view'].shape)}")
    for i, v in enumerate(out["global_views"]):
        print(f"  global[{i}]: {tuple(v.shape)}")
    for i, v in enumerate(out["local_views"]):
        print(f"  local[{i}]:  {tuple(v.shape)}")
    print("=== gen3_data sanity OK ===")


if __name__ == "__main__":
    _sanity_check()
