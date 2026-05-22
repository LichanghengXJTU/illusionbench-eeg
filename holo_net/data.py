"""Dataset loaders for HOLO-Net training.

Glint360K via WebDataset:
  - 17.1M images, 360,232 identities
  - 112×112 already-aligned faces (RetinaFace)
  - 1385 .tar.gz shards from gaunernst/glint360k-wds-gz

Each batch yields a dict with:
  - "image": (B, 3, 224, 224) — upsampled from 112×112 to 224×224
  - "identity": (B,) long — class index 0..360231
  - "orientation": (B,) long — 0=upright, 1=inverted
  - "is_face": (B,) long — always 1 for Glint (used only when mixing
    ImageNet non-face samples in)
  - "image_paired": (B, 3, 224, 224) — same identity, different augmentation
    for view-invariance loss

For the orientation aux label:
  - We RANDOMLY vertical-flip 50% of samples and label them "inverted"
  - **Critical**: we NEVER perform feature-region inversion (no Thatcher
    contamination of training set)

For the view-invariance pair:
  - We sample the SAME image twice with different random augmentations
    (crop, color jitter, horizontal flip) — this approximates same-identity-
    different-pose without needing identity-grouped sampling. A stricter
    version would group by identity but is more expensive.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import io
import os
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image


def _augment_face(img_pil: Image.Image, training_size: int = 224) -> torch.Tensor:
    """Standard face augmentations:
      - Random horizontal flip (50%)
      - Random crop ±10% with resize to 224×224
      - Color jitter
      - Convert to tensor + normalize to [-1, 1]
    NO vertical flip here (that's added separately as orientation aug).
    NO Thatcher-style feature-region perturbations.
    """
    tfm = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResizedCrop(training_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return tfm(img_pil)


def _flip_vertical(img_tensor: torch.Tensor) -> torch.Tensor:
    """Vertical flip = whole-face inversion (NOT thatcher)."""
    return torch.flip(img_tensor, dims=[-2])


def make_glint360k_dataset(
    shard_pattern: str,
    p_inverted: float = 0.5,
    pair_for_view_invariance: bool = True,
) -> IterableDataset:
    """Create a WebDataset over Glint360K with multi-task augmentations.

    Args:
      shard_pattern: glob or brace-pattern over .tar.gz shards
        e.g. "/workspace/glint360k/glint360k-{0000..1384}.tar.gz"
      p_inverted: probability of applying whole-face vertical flip
        (for Orientation Gate training)
      pair_for_view_invariance: whether to yield a second augmentation
        of the same image (for view-invariance loss)
    """
    import webdataset as wds

    def transform(sample):
        # sample is a dict with "jpg" (bytes) and "cls" (bytes containing int)
        img_bytes = sample["jpg"] if isinstance(sample["jpg"], bytes) else sample["jpg"]
        cls = sample["cls"]
        if isinstance(cls, bytes):
            cls = int(cls.decode("utf-8").strip())
        elif isinstance(cls, str):
            cls = int(cls.strip())
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Upsample 112→224 to match HOLO-Net input expectation
        img_pil = img_pil.resize((224, 224), Image.BILINEAR)

        # Primary augmentation
        img1 = _augment_face(img_pil)

        # Orientation label (random vertical flip = whole-face inversion)
        if torch.rand(1).item() < p_inverted:
            img1 = _flip_vertical(img1)
            orient = 1  # inverted
        else:
            orient = 0  # upright

        # View-invariance pair: a second augmentation of the same image
        # (different random crop, jitter; NOT inverted to keep view-invariance
        # focused on pose/light/translation variation, not orientation)
        if pair_for_view_invariance:
            img2 = _augment_face(img_pil)  # different random aug
        else:
            img2 = img1

        return {
            "image": img1,
            "image_paired": img2,
            "identity": torch.tensor(cls, dtype=torch.long),
            "orientation": torch.tensor(orient, dtype=torch.long),
            "is_face": torch.tensor(1, dtype=torch.long),  # always 1 for Glint
        }

    ds = (wds.WebDataset(shard_pattern, shardshuffle=100)
          .shuffle(1000)
          .decode()
          .map(transform))
    return ds


def make_glint360k_dataloader(
    shard_pattern: str,
    batch_size: int = 256,
    num_workers: int = 8,
    p_inverted: float = 0.5,
    pair_for_view_invariance: bool = True,
) -> DataLoader:
    """High-level dataloader for Glint360K with multi-task augmentations."""
    ds = make_glint360k_dataset(shard_pattern, p_inverted, pair_for_view_invariance)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ----------------------------------------------------------------------------
# CLI sanity check
# ----------------------------------------------------------------------------

def _sanity_check():
    """Verify dataloader yields correct-shape batches."""
    # Use small subset for sanity (first 10 shards)
    shard_pattern = "/workspace/glint360k/glint360k-{0000..0009}.tar.gz"
    dl = make_glint360k_dataloader(shard_pattern, batch_size=8, num_workers=2)
    for i, batch in enumerate(dl):
        print(f"Batch {i}:")
        for k, v in batch.items():
            if torch.is_tensor(v):
                print(f"  {k:20s}: shape={tuple(v.shape)} dtype={v.dtype}")
        print(f"  identity sample: {batch['identity'][:8].tolist()}")
        print(f"  orientation sample: {batch['orientation'][:8].tolist()}")
        if i >= 2:
            break
    print("\n=== Dataset sanity check DONE ===")


if __name__ == "__main__":
    _sanity_check()
