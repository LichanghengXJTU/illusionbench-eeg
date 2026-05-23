"""holo_net/model_v2_dinov2_partaware.py — HOLO-Net v2.2 (Gen 1, variant 1)
in the post-redirect iteration ladder (NOTES_FOR_USER 2026-05-23 ~22:20).

Pivot from v3 (global FTPC, E045): K=4 part-aware face templates at fixed
sub-regions of the 16×16 patch grid. δ readout = concat of per-part pooled
mismatches → 1536-d (4 × 384).

Pre-registered hypothesis (v2.2):
- CSI 1.300 (close to 1.5 threshold) likely lifts toward PASS — per-part
  mismatch captures composite-face misalignment more cleanly than global δ.
- PWI 0.446 (PASS) likely held — already captured by the DINOv2 backbone.
- ISIrbox 0.654 (PASS) likely held.
- **ISI 0.901 (anti-Thatcher) UNLIKELY to flip** in v2.2 alone — fixed-grid
  parts don't introduce upright/inverted asymmetry in the model's
  processing. Verdict: if ISI < 1.0, the next iteration v2.3 adds
  orientation-aware readout (compare AFP to vflip(AFP)).

Architecture:
  image → DINOv2 ViT-S/14 (frozen) → patch tokens (B, 384, 16, 16)
                                                      │
        {T_eyes, T_nose, T_mouth, T_chin} (384, h_k, w_k each, learnable + EMA)
                                                      │
                  per-part: δ_k = AFP[region_k] − T_k → δ_k_pooled ∈ ℝ^384
                                                      │
                                  concat → δ_concat ∈ ℝ^1536  ← FFA readout

Part region map on the 16×16 patch grid (MediaPipe-aligned face crop):
  eyes : rows 5-7   cols 3-12  (3 × 10)
  nose : rows 7-9   cols 6-9   (3 × 4)
  mouth: rows 10-12 cols 4-11  (3 × 8)
  chin : rows 12-15 cols 5-10  (4 × 6)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Part region map (row_start, row_end_exclusive, col_start, col_end_exclusive)
PART_REGIONS: dict[str, tuple[int, int, int, int]] = {
    "eyes":  (5, 8, 3, 13),    # 3 × 10
    "nose":  (7, 10, 6, 10),   # 3 × 4
    "mouth": (10, 13, 4, 12),  # 3 × 8
    "chin":  (12, 16, 5, 11),  # 4 × 6
}


@dataclass
class HOLONetV2DinoPartAwareConfig:
    dinov2_model: str = "dinov2_vits14"
    image_size: int = 224
    patch_spatial: int = 16
    feature_dim: int = 384
    torch_cache: str = "/workspace/.torch_cache"
    template_init_std: float = 0.02
    template_ema_decay: float = 0.999
    parts: tuple = ("eyes", "nose", "mouth", "chin")


def _load_dinov2(cfg: HOLONetV2DinoPartAwareConfig) -> nn.Module:
    os.environ.setdefault("TORCH_HOME", cfg.torch_cache)
    model = torch.hub.load("facebookresearch/dinov2", cfg.dinov2_model,
                            trust_repo=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class PartTemplate(nn.Module):
    """One learnable + EMA template for a face part."""
    def __init__(self, dim: int, h: int, w: int, init_std: float,
                 ema_decay: float):
        super().__init__()
        self.dim, self.h, self.w, self.ema_decay = dim, h, w, ema_decay
        self.template = nn.Parameter(torch.zeros(dim, h, w))
        nn.init.normal_(self.template, std=init_std)
        self.register_buffer("template_ema", torch.zeros(dim, h, w))
        self.register_buffer("ema_initialised", torch.tensor(False))

    @torch.no_grad()
    def update_ema(self, face_part_batch: torch.Tensor) -> None:
        """face_part_batch (n_face, D, h, w) from face-mask=True samples."""
        if face_part_batch.numel() == 0:
            return
        batch_mean = face_part_batch.mean(dim=0)  # (D, h, w)
        if not bool(self.ema_initialised):
            self.template_ema.copy_(batch_mean)
            self.ema_initialised.fill_(True)
        else:
            self.template_ema.mul_(self.ema_decay).add_(
                batch_mean, alpha=1.0 - self.ema_decay)


class HOLONetV2Dinov2PartAware(nn.Module):
    """HOLO-Net v2.2 — frozen DINOv2 + K=4 part-aware FTPC templates."""

    def __init__(self, cfg: Optional[HOLONetV2DinoPartAwareConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = HOLONetV2DinoPartAwareConfig()
        self.cfg = cfg
        self.backbone = _load_dinov2(cfg)
        self.parts = list(cfg.parts)

        self.part_templates = nn.ModuleDict()
        for k in self.parts:
            r0, r1, c0, c1 = PART_REGIONS[k]
            h, w = r1 - r0, c1 - c0
            self.part_templates[k] = PartTemplate(
                dim=cfg.feature_dim, h=h, w=w,
                init_std=cfg.template_init_std,
                ema_decay=cfg.template_ema_decay)

    @torch.no_grad()
    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone.forward_features(x)
        patch = out["x_norm_patchtokens"]  # (B, N, D)
        B, N, D = patch.shape
        H = W = int(N ** 0.5)
        return patch.transpose(1, 2).reshape(B, D, H, W).contiguous()

    def _part_region(self, afp_spatial: torch.Tensor, k: str) -> torch.Tensor:
        r0, r1, c0, c1 = PART_REGIONS[k]
        return afp_spatial[..., r0:r1, c0:c1]   # (B, D, h, w)

    def forward(self, x: torch.Tensor) -> dict:
        afp_spatial = self._extract_patch_tokens(x)
        afp_pooled = F.adaptive_avg_pool2d(afp_spatial, 1).flatten(1)  # (B, D)

        # Per-part δ
        delta_parts: dict[str, torch.Tensor] = {}
        delta_pooled_parts: dict[str, torch.Tensor] = {}
        for k in self.parts:
            T_k = self.part_templates[k].template
            r0, r1, c0, c1 = PART_REGIONS[k]
            # Match spatial shape: only compute if patch grid is the expected
            # 16×16 (i.e., not a local view of differing size).
            if afp_spatial.shape[-2] == self.cfg.patch_spatial:
                region = afp_spatial[..., r0:r1, c0:c1]
                delta = region - T_k[None]                            # (B, D, h, w)
                delta_parts[k] = delta
                delta_pooled_parts[k] = F.adaptive_avg_pool2d(delta, 1).flatten(1)

        # Concat per-part pooled δ
        if delta_pooled_parts:
            delta_concat = torch.cat([delta_pooled_parts[k]
                                       for k in self.parts], dim=1)   # (B, D × K)
        else:
            delta_concat = None

        out = {
            "afp_spatial": afp_spatial, "afp_pooled": afp_pooled,
            "afp": afp_pooled, "atl": afp_pooled,
        }
        if delta_concat is not None:
            out["delta_concat"] = delta_concat
            out["delta_pooled_parts"] = delta_pooled_parts
            out["delta_parts"] = delta_parts
            # FFA = δ_concat (the part-aware FTPC readout)
            out["ffa"] = delta_concat
        else:
            out["ffa"] = afp_pooled

        # v1/v2/v4/mfp aliases: fallback to afp_pooled (no anatomy layers in
        # frozen ViT setup)
        for k in ("v1", "v2", "v4", "mfp"):
            out[k] = afp_pooled
        return out

    def update_ema_for_faces(self, afp_spatial: torch.Tensor,
                              face_mask: torch.Tensor) -> dict[str, int]:
        """Update each part's EMA template from face-mask=True samples.
        Returns n_face_samples per part."""
        info = {}
        n_face = int(face_mask.sum().item())
        info["n_face_total"] = n_face
        if n_face == 0 or afp_spatial.shape[-2] != self.cfg.patch_spatial:
            return info
        face_afp_spatial = afp_spatial[face_mask]  # (n_face, D, H, W)
        for k in self.parts:
            r0, r1, c0, c1 = PART_REGIONS[k]
            face_part = face_afp_spatial[..., r0:r1, c0:c1]
            self.part_templates[k].update_ema(face_part.detach())
            info[f"n_face_for_{k}"] = n_face
        return info

    def get_layer_for_eeg_readout(self, name: str,
                                    output_dict: dict) -> torch.Tensor:
        if name in {"v1", "v2", "v4", "mfp"}:
            return output_dict["afp_pooled"]
        if name in {"afp", "afp_pooled"}:
            return output_dict["afp_pooled"]
        if name in {"ffa", "delta", "fsts", "delta_concat"}:
            return output_dict["ffa"]   # δ_concat 1536-d in v2.2
        if name == "atl":
            return output_dict["afp_pooled"]
        raise ValueError(f"unknown layer: {name}")


def _sanity_check():
    cfg = HOLONetV2DinoPartAwareConfig()
    model = HOLONetV2Dinov2PartAware(cfg)
    print(f"backbone: {cfg.dinov2_model}; feature_dim={cfg.feature_dim}; "
          f"patch_spatial={cfg.patch_spatial}; parts={cfg.parts}")

    # Part region summary
    total_part_pixels = 0
    for k in cfg.parts:
        r0, r1, c0, c1 = PART_REGIONS[k]
        h, w = r1 - r0, c1 - c0
        T_shape = tuple(model.part_templates[k].template.shape)
        print(f"  {k:5s}  region [{r0}:{r1}, {c0}:{c1}]  T {T_shape}  ({h*w} patches)")
        total_part_pixels += h * w
    print(f"  total covered patches: {total_part_pixels} / {cfg.patch_spatial**2}")

    B = 4
    x = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    out = model(x)
    print("forward shapes:")
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k:18s}: {tuple(v.shape)}")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  {k}[{kk}]: {tuple(vv.shape)}")

    # EMA update test
    face_mask = torch.tensor([True, False, True, False])
    info = model.update_ema_for_faces(out["afp_spatial"], face_mask)
    print(f"EMA update info: {info}")
    for k in cfg.parts:
        n = model.part_templates[k].template_ema.norm().item()
        print(f"  {k} T_ema norm: {n:.2f}")

    # Trainable params
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: total {n_total/1e6:.2f}M  trainable {n_trainable/1e6:.4f}M "
          f"(backbone frozen)")
    print("=== v2.2 part-aware FTPC sanity OK ===")


if __name__ == "__main__":
    _sanity_check()
