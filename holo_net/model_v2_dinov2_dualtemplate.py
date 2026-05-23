"""holo_net/model_v2_dinov2_dualtemplate.py — HOLO-Net v3.0 (Gen 2 variant 1)
in the post-redirect iteration ladder.

Gen 1 EXHAUSTED at ISI ceiling 1.082 (E051). Gen 2 hypothesis: dual
asymmetric templates T_upright + T_inverted trained from upright face
samples and their vflipped versions, respectively. The readout uses both
templates so any asymmetry in their learned representations contributes
to the FFA layer.

Forward:
  AFP = DINOv2(x).pooled
  δ_upright  = AFP - T_upright_pooled
  δ_inverted = AFP - T_inverted_pooled
  ffa = concat(δ_upright, δ_inverted)  → 768-d

Honest expectation (filed pre-run): if T_upright and T_inverted are trained on
the same data (just with one vflipped), they will be symmetric mirrors of
each other and the readout will give symmetric responses → ISI ≈ 1.0. If
DINOv2 has even a small intrinsic upright bias (its training distribution
includes some non-upright images but mostly upright), there COULD be a
slight asymmetry. Best case: ISI 1.05-1.15. If ISI ≤ 1.2 (per pre-registered
trigger), escalate to Gen 3 = from-scratch backbone retrain.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HOLONetV2DinoDualConfig:
    dinov2_model: str = "dinov2_vits14"
    image_size: int = 224
    patch_spatial: int = 16
    feature_dim: int = 384
    torch_cache: str = "/workspace/.torch_cache"
    template_init_std: float = 0.02
    template_ema_decay: float = 0.999


def _load_dinov2(cfg: HOLONetV2DinoDualConfig) -> nn.Module:
    os.environ.setdefault("TORCH_HOME", cfg.torch_cache)
    model = torch.hub.load("facebookresearch/dinov2", cfg.dinov2_model,
                            trust_repo=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class FaceTemplate(nn.Module):
    """Single learnable + EMA template (same as v3's FaceTemplate)."""
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


class HOLONetV2Dinov2Dual(nn.Module):
    """v3.0 — frozen DINOv2 + dual asymmetric templates T_upright + T_inverted."""

    def __init__(self, cfg: Optional[HOLONetV2DinoDualConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = HOLONetV2DinoDualConfig()
        self.cfg = cfg
        self.backbone = _load_dinov2(cfg)
        self.face_template_up = FaceTemplate(
            dim=cfg.feature_dim, spatial=cfg.patch_spatial,
            init_std=cfg.template_init_std, ema_decay=cfg.template_ema_decay)
        self.face_template_inv = FaceTemplate(
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
        afp_spatial = self._extract_patch_tokens(x)
        afp_pooled = F.adaptive_avg_pool2d(afp_spatial, 1).flatten(1)

        T_up_pooled = F.adaptive_avg_pool2d(
            self.face_template_up.template[None], 1).flatten(1).squeeze(0)
        T_inv_pooled = F.adaptive_avg_pool2d(
            self.face_template_inv.template[None], 1).flatten(1).squeeze(0)
        delta_up = afp_pooled - T_up_pooled[None]
        delta_inv = afp_pooled - T_inv_pooled[None]
        delta_concat = torch.cat([delta_up, delta_inv], dim=1)   # (B, 2D)

        out = {
            "afp_spatial": afp_spatial,
            "afp_pooled": afp_pooled,
            "afp": afp_pooled, "atl": afp_pooled,
            "delta_up": delta_up,
            "delta_inv": delta_inv,
            "delta_concat": delta_concat,
            "ffa": delta_concat,
        }
        for k in ("v1", "v2", "v4", "mfp"):
            out[k] = afp_pooled
        return out

    def update_ema_dual(self, afp_spatial: torch.Tensor,
                          afp_vflip_spatial: torch.Tensor,
                          face_mask: torch.Tensor) -> dict[str, int]:
        info = {"n_face": int(face_mask.sum().item())}
        if info["n_face"] == 0:
            return info
        face_afp = afp_spatial[face_mask].detach()
        face_afp_vflip = afp_vflip_spatial[face_mask].detach()
        self.face_template_up.update_ema(face_afp)
        self.face_template_inv.update_ema(face_afp_vflip)
        return info

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
        if name == "delta_up":
            return output_dict["delta_up"]
        if name == "delta_inv":
            return output_dict["delta_inv"]
        raise ValueError(f"unknown layer: {name}")
