"""eeg_decoder/data.py — load ATM EEG features + image features for the
THINGS-EEG2 split.

Data layout (server-side, fetched in tick 81):

  /workspace/illusionbench-eeg/data/atm_emb_eeg/emb_eeg/
    ATM_S_eeg_features_sub-{01..10}_train.pt   # tensor (66160, 1024)
    ATM_S_eeg_features_sub-{01..10}_test.pt    # tensor (200, 1024)

  /workspace/illusionbench-eeg/data/things_features/
    ViT-H-14_features_train.pt   # dict {text_features (1654, 1024), img_features (16540, 1024)}
    ViT-H-14_features_test.pt    # dict {text_features (200, 1024), img_features (200, 1024)}
    dinov2_vits14_train.pt       # tensor (16540, 384)  — created in tick 82
    dinov2_vits14_test.pt        # tensor (200, 384)    — created in tick 82

Conventions:
- ATM train: 16540 stims × 4 reps per subject = 66160 trials, in stim-major
  order (matches ATM repo `eegdatasets_leaveone.py`). Reshape to
  (16540, 4, 1024) for trial-averaging if needed.
- ATM test: pre-averaged to (200, 1024) per subject.
- Image features index aligns with EEG by THINGS concept order.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_data_root() -> Path:
    env = os.environ.get("ILLUSIONBENCH_DATA_ROOT")
    if env:
        return Path(env)
    return Path("/workspace/illusionbench-eeg/data")


def load_atm_eeg(subject: int, split: str = "train",
                 average_reps: bool = False,
                 data_root: Optional[Path] = None) -> torch.Tensor:
    """Load per-subject ATM EEG features.

    Args:
      subject: 1..10
      split: 'train' or 'test'
      average_reps: if True and split=='train', collapse 4 reps → 1 by mean.
        Returns shape (16540, 1024) instead of (66160, 1024).
    """
    root = data_root or get_data_root()
    p = root / f"atm_emb_eeg/emb_eeg/ATM_S_eeg_features_sub-{subject:02d}_{split}.pt"
    x = torch.load(p, map_location="cpu", weights_only=True)
    if split == "train" and average_reps:
        # Reshape (16540 * 4, 1024) → (16540, 4, 1024) → mean → (16540, 1024)
        n_stim = 16540
        n_rep = x.shape[0] // n_stim
        x = x.reshape(n_stim, n_rep, x.shape[-1]).mean(dim=1)
    return x


def load_image_features(feature: str = "ViT-H-14", split: str = "train",
                          data_root: Optional[Path] = None) -> torch.Tensor:
    """Load THINGS image features for a given feature type.

    Args:
      feature: 'ViT-H-14' (CLIP) or 'dinov2_vits14' (DINOv2, after tick 82)
      split: 'train' or 'test'

    Returns:
      img_features tensor of shape (n_stim, dim). For 'ViT-H-14' the source
      file is a dict; we return only `img_features`.
    """
    root = data_root or get_data_root()
    if feature == "ViT-H-14":
        p = root / f"things_features/ViT-H-14_features_{split}.pt"
        d = torch.load(p, map_location="cpu", weights_only=True)
        return d["img_features"]
    elif feature.startswith("dinov2"):
        p = root / f"things_features/{feature}_{split}.pt"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run eeg_decoder.extract_dinov2_things "
                f"(tick 82) to produce DINOv2 features on THINGS images.")
        return torch.load(p, map_location="cpu", weights_only=True)
    raise ValueError(f"unknown feature type: {feature}")


def expand_image_features_to_trials(img_features: torch.Tensor,
                                      n_reps: int = 4) -> torch.Tensor:
    """Repeat each image feature `n_reps` times to align with ATM train EEG
    (which has 4 trials per image, in stim-major order).

    Input: (16540, dim) → Output: (66160, dim)
    """
    return img_features.repeat_interleave(n_reps, dim=0)


def sanity_check() -> dict:
    """Quick check that all expected files exist and shapes match."""
    out = {"missing": [], "present": [], "shapes": {}}
    root = get_data_root()
    # ATM EEG
    for i in range(1, 11):
        for split in ("train", "test"):
            p = root / f"atm_emb_eeg/emb_eeg/ATM_S_eeg_features_sub-{i:02d}_{split}.pt"
            if p.exists():
                t = torch.load(p, map_location="cpu", weights_only=True)
                out["present"].append(str(p))
                out["shapes"][str(p)] = tuple(t.shape)
            else:
                out["missing"].append(str(p))
    # Image features
    for feat in ("ViT-H-14",):
        for split in ("train", "test"):
            p = root / f"things_features/{feat}_features_{split}.pt"
            if p.exists():
                d = torch.load(p, map_location="cpu", weights_only=True)
                out["present"].append(str(p))
                out["shapes"][str(p)] = {k: tuple(v.shape) for k, v in d.items()}
            else:
                out["missing"].append(str(p))
    # DINOv2 (likely missing until tick 82)
    for split in ("train", "test"):
        p = root / f"things_features/dinov2_vits14_{split}.pt"
        if p.exists():
            t = torch.load(p, map_location="cpu", weights_only=True)
            out["present"].append(str(p))
            out["shapes"][str(p)] = tuple(t.shape)
        else:
            out["missing"].append(str(p))
    return out


if __name__ == "__main__":
    import json
    report = sanity_check()
    print(f"Present: {len(report['present'])}")
    print(f"Missing: {len(report['missing'])}")
    if report["missing"]:
        for m in report["missing"]:
            print(f"  MISSING: {m}")
    print("\nShape summary (first 3):")
    for k, v in list(report["shapes"].items())[:3]:
        print(f"  {k} -> {v}")
