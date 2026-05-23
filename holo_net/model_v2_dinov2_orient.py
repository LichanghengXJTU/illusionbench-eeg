"""holo_net/model_v2_dinov2_orient.py — HOLO-Net v2.3 (Gen 1 variant 2)
in the post-redirect iteration ladder.

Diagnosis of v2.2 (E049): part-aware alone severed holistic context →
CSI/PWI regressed. The fix: keep v3's global FTPC template (which works for
CSI/PWI) and ADD an orientation-asymmetric channel via a 2× DINOv2 forward
on x and vflip(x). The two δ residuals concatenated give a 768-d FFA readout
that should encode both holistic deviation (δ_global, like v3) AND
orientation-conditional deviation (δ_orient, NEW).

Analytical prediction for Thatcher:
  V1 upright_normal:    AFP_orig ≈ T → δ_global ≈ 0;     AFP_vflip = inverted_view → ≠ T → δ_orient ≈ medium
  V2 upright_thatched:  AFP_orig slightly off T → δ_global small-med; AFP_vflip very off → δ_orient large
  V3 inverted_normal:   AFP_orig ≠ T → δ_global ≈ med;   AFP_vflip ≈ T → δ_orient ≈ 0
  V4 inverted_thatched: AFP_orig ≠ T → δ_global ≈ med;   AFP_vflip slightly off → δ_orient small-med

  d(V1, V2) in (δ_global, δ_orient) ≈ √(med² + med²) ≈ MEDIUM
  d(V3, V4) in same ≈ √(0² + small-med²) ≈ SMALL
  → ISI = MEDIUM / SMALL > 1 → pro-Thatcher direction.

Reuses v3's checkpoint (template T is the same global FTPC template; was
trained on ImageNet face samples, predominantly upright).
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HOLONetV2DinoOrientConfig:
    dinov2_model: str = "dinov2_vits14"
    image_size: int = 224
    patch_spatial: int = 16
    feature_dim: int = 384
    torch_cache: str = "/workspace/.torch_cache"
    template_init_std: float = 0.02
    template_ema_decay: float = 0.999


def _load_dinov2(cfg: HOLONetV2DinoOrientConfig) -> nn.Module:
    os.environ.setdefault("TORCH_HOME", cfg.torch_cache)
    model = torch.hub.load("facebookresearch/dinov2", cfg.dinov2_model,
                            trust_repo=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class FaceTemplate(nn.Module):
    """Same as model_v2_dinov2.FaceTemplate — kept here for state_dict
    compatibility when loading the v3 checkpoint."""

    def __init__(self, dim: int, spatial: int = 16, init_std: float = 0.02,
                 ema_decay: float = 0.999):
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

    def forward(self) -> torch.Tensor:
        return self.template


class HOLONetV2Dinov2Orient(nn.Module):
    """v2.3 — frozen DINOv2 + global FTPC + orientation-aware vflip readout."""

    def __init__(self, cfg: Optional[HOLONetV2DinoOrientConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = HOLONetV2DinoOrientConfig()
        self.cfg = cfg
        self.backbone = _load_dinov2(cfg)
        # Same parameter name 'face_template' as v3 to load v3 checkpoint
        self.face_template = FaceTemplate(
            dim=cfg.feature_dim, spatial=cfg.patch_spatial,
            init_std=cfg.template_init_std, ema_decay=cfg.template_ema_decay)

    @torch.no_grad()
    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone.forward_features(x)
        patch = out["x_norm_patchtokens"]
        B, N, D = patch.shape
        H = W = int(N ** 0.5)
        return patch.transpose(1, 2).reshape(B, D, H, W).contiguous()

    def forward(self, x: torch.Tensor) -> dict:
        # First forward: original image
        afp_spatial = self._extract_patch_tokens(x)                         # (B, D, H, W)
        afp_pooled = F.adaptive_avg_pool2d(afp_spatial, 1).flatten(1)        # (B, D)

        # Second forward: vflipped image (vflip the input, NOT the feature
        # map; we want DINOv2's actual response to inverted input)
        x_vflip = torch.flip(x, dims=[-2])                                   # vflip H axis
        afp_vflip_spatial = self._extract_patch_tokens(x_vflip)
        afp_vflip_pooled = F.adaptive_avg_pool2d(afp_vflip_spatial, 1).flatten(1)

        T = self.face_template()                                             # (D, H, W)
        T_pooled = F.adaptive_avg_pool2d(T[None], 1).flatten(1).squeeze(0)   # (D,)

        # Two δ channels
        delta_global_pooled = afp_pooled - T_pooled[None]                    # (B, D)
        delta_orient_pooled = afp_vflip_pooled - T_pooled[None]              # (B, D)

        # FFA readout = concat
        delta_concat = torch.cat([delta_global_pooled, delta_orient_pooled],
                                   dim=1)                                       # (B, 2D)

        out = {
            "afp_spatial": afp_spatial,
            "afp_vflip_spatial": afp_vflip_spatial,
            "afp_pooled": afp_pooled,
            "afp_vflip_pooled": afp_vflip_pooled,
            "afp": afp_pooled,
            "atl": afp_pooled,
            "delta_global_pooled": delta_global_pooled,
            "delta_orient_pooled": delta_orient_pooled,
            "delta_concat": delta_concat,
            "ffa": delta_concat,    # 2D = 768
        }
        for k in ("v1", "v2", "v4", "mfp"):
            out[k] = afp_pooled
        return out

    def get_layer_for_eeg_readout(self, name: str,
                                    output_dict: dict) -> torch.Tensor:
        if name in {"v1", "v2", "v4", "mfp"}:
            return output_dict["afp_pooled"]
        if name in {"afp", "afp_pooled"}:
            return output_dict["afp_pooled"]
        if name in {"ffa", "delta", "fsts", "delta_concat"}:
            return output_dict["ffa"]
        if name == "atl":
            return output_dict["afp_pooled"]
        # Special diagnostic layers (not in the default 7)
        if name == "delta_global":
            return output_dict["delta_global_pooled"]
        if name == "delta_orient":
            return output_dict["delta_orient_pooled"]
        raise ValueError(f"unknown layer: {name}")


def _sanity_check():
    cfg = HOLONetV2DinoOrientConfig()
    model = HOLONetV2Dinov2Orient(cfg)
    print(f"backbone: {cfg.dinov2_model}; feature_dim={cfg.feature_dim}")
    B = 4
    x = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        out = model(x)
    print("forward shapes:")
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k:22s}: {tuple(v.shape)}")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"params: total {n_total/1e6:.2f}M  trainable {n_trainable/1e6:.4f}M  "
          f"(reuses v3 face_template)")
    print("=== v2.3 orient sanity OK ===")


if __name__ == "__main__":
    _sanity_check()
