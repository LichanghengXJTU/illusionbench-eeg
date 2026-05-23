"""holo_net/gen3_model.py — Gen 3 backbone (from-scratch ViT-S/14) +
DINO projection head + orientation auxiliary head.

Architecture:
  Image → ViT-S/14 (random init) → CLS token (384-d) + patch tokens (256, 384)
       │
       ├─ DINO head: 3-layer MLP 384→2048→2048→8192 + weight-norm last layer
       │   (used for DINOv2-style SSL — teacher-student KL with centering)
       │
       └─ Orient head: 384→256→2 (light MLP)
           (used for orient-aux 0°/180° classification — biases the backbone
            toward orientation-asymmetric features)

The orient-aux head is the LEVER for breaking the orientation-invariance that
froze the v3/v2.x ceiling at ISI ~1. By training it, the backbone develops
features that distinguish upright from inverted images — exactly what
Thatcher detection requires.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Gen3ModelConfig:
    img_size: int = 224
    patch_size: int = 14
    backbone_dim: int = 384       # ViT-S/14 d_embed
    backbone_arch: str = "vit_small_patch14_dinov2"
    dino_hidden_dim: int = 2048
    dino_bottleneck_dim: int = 256
    dino_out_dim: int = 8192      # DINOv2 default 65536 → smaller for single GPU
    dino_n_layers: int = 3
    orient_hidden_dim: int = 256
    orient_n_classes: int = 2     # 0=upright, 1=inverted


class DINOProjectionHead(nn.Module):
    """3-layer MLP + L2-norm + weight-normed last linear (DINO/DINOv2 std)."""

    def __init__(self, in_dim: int, hidden_dim: int, bottleneck_dim: int,
                 out_dim: int, n_layers: int = 3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim,
                                                     bias=False))
        # Init: weight_g freeze at 1 (DINOv2 convention)
        self.last.weight_g.data.fill_(1.0)
        self.last.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)   # L2-norm bottleneck
        x = self.last(x)
        return x


class OrientHead(nn.Module):
    """Light 1-hidden-layer MLP for 0°/180° classification."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Gen3Backbone(nn.Module):
    """Wraps ViT-S/14 + DINO head + orient head.

    Forward modes:
      mode='dino_global'  : x (B, 3, 224, 224) → DINO projection (B, out_dim) AND CLS (B, 384)
      mode='dino_local'   : x (B, 3, 96,  96)  → DINO projection (B, out_dim)
                              [no CLS extraction; local view only used for student]
      mode='orient'       : x (B, 3, 224, 224) → orient logits (B, 2)
      mode='features'     : x (B, 3, 224, 224) → CLS + patch tokens dict (for eval)
    """

    def __init__(self, cfg: Optional[Gen3ModelConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = Gen3ModelConfig()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.backbone_arch, pretrained=False, num_classes=0,
            img_size=cfg.img_size)
        # NOTE: forward_features returns (B, N+1, D) with CLS at idx 0
        self.dino_head = DINOProjectionHead(
            in_dim=cfg.backbone_dim,
            hidden_dim=cfg.dino_hidden_dim,
            bottleneck_dim=cfg.dino_bottleneck_dim,
            out_dim=cfg.dino_out_dim,
            n_layers=cfg.dino_n_layers,
        )
        self.orient_head = OrientHead(
            in_dim=cfg.backbone_dim,
            hidden_dim=cfg.orient_hidden_dim,
            n_classes=cfg.orient_n_classes,
        )

    def _forward_backbone(self, x: torch.Tensor) -> dict:
        """Returns CLS, patch tokens (B, N, D), pooled (B, D)."""
        feat = self.backbone.forward_features(x)   # (B, N+1, D)
        cls_tok = feat[:, 0, :]                     # (B, D)
        patch_tok = feat[:, 1:, :]                  # (B, N, D)
        return {"cls": cls_tok, "patch": patch_tok}

    def forward(self, x: torch.Tensor, mode: str = "dino_global") -> dict:
        out = self._forward_backbone(x)
        if mode in {"dino_global", "dino_local"}:
            out["dino_proj"] = self.dino_head(out["cls"])
            return out
        elif mode == "orient":
            out["orient_logits"] = self.orient_head(out["cls"])
            return out
        elif mode == "features":
            B = x.shape[0]
            D = self.cfg.backbone_dim
            H = W = self.cfg.img_size // self.cfg.patch_size
            assert out["patch"].shape == (B, H * W, D), \
                f"patch shape {out['patch'].shape} != ({B}, {H*W}, {D})"
            out["patch_spatial"] = out["patch"].transpose(1, 2).reshape(B, D, H, W).contiguous()
            out["afp_pooled"] = F.adaptive_avg_pool2d(out["patch_spatial"], 1).flatten(1)
            return out
        raise ValueError(f"unknown mode: {mode}")


def _sanity_check():
    cfg = Gen3ModelConfig()
    model = Gen3Backbone(cfg)
    n_total = sum(p.numel() for p in model.parameters())
    n_back = sum(p.numel() for p in model.backbone.parameters())
    n_dino = sum(p.numel() for p in model.dino_head.parameters())
    n_orient = sum(p.numel() for p in model.orient_head.parameters())
    print(f"params: total {n_total/1e6:.1f}M  "
          f"backbone {n_back/1e6:.1f}M  dino_head {n_dino/1e6:.2f}M  "
          f"orient_head {n_orient/1e6:.2f}M")

    x224 = torch.randn(2, 3, 224, 224)
    x96 = torch.randn(2, 3, 96, 96)

    out_g = model(x224, mode="dino_global")
    print("dino_global forward:",
          {k: tuple(v.shape) for k, v in out_g.items() if torch.is_tensor(v)})

    # ViT at img_size=224 won't accept 96 directly without re-instantiating;
    # local views need a fresh backbone with img_size=96, OR resize to 224.
    # For DINO multi-crop, the conventional approach is: re-run patch embed
    # with the local-view image. timm's ViT will fail at 96 because positional
    # embedding is fixed for 224.
    # → We will resize local-view 96 → 224 in training before feeding (or use
    #   a separate backbone with img_size=96). Note this caveat in trainer.

    out_o = model(x224, mode="orient")
    print("orient forward:",
          {k: tuple(v.shape) for k, v in out_o.items() if torch.is_tensor(v)})

    out_f = model(x224, mode="features")
    print("features forward:",
          {k: tuple(v.shape) for k, v in out_f.items() if torch.is_tensor(v)})

    print("=== gen3_model sanity OK ===")


if __name__ == "__main__":
    _sanity_check()
