"""holo_net/model_v2_dinov2.py — HOLO-Net v2.1 FTPC with a frozen, pre-trained
DINOv2 backbone (option D — pivoting after runs 1-3 of from-scratch SSL all
collapsed to uniform output).

Rationale:
- The user's framing: a model trained on natural images that include faces;
  must be strong on normal tasks; FTPC is the modeling novelty.
- DINOv2 (Oquab et al. 2024, Meta) is SOTA SSL on a natural-image dataset
  (LVD-142M, includes faces in natural distribution) — exactly matches the
  "trained on diverse natural visual experience including faces" criterion.
- Using DINOv2 as the frozen backbone keeps the FTPC novelty (face template
  T + δ readout = fSTS-analogue) while removing the from-scratch SSL collapse
  risk that ran us 3× in a row.
- DINOv2 features are already a competitive EEG-decoder target in the
  literature (the same family as ATM/AVDE), so the EEG-decoder Stage 3 path
  is preserved.

Architecture:
  image → DINOv2 ViT-S/14 (frozen) → patch tokens (B, 384, 16, 16)
                                                      │
                            face_template T (384, 16, 16) (learnable + EMA)
                                                      │
                                       δ = patch − T  ← fSTS-analogue
                                                      │
                                pooled δ → δ_pooled (B, 384) ← test-time
                                                                illusion readout

For per-layer EEG readout compatibility with eval_extract.py / eval_falsification:
  afp = afp_pooled (CLS or mean-pooled patch feature)
  ffa = δ_pooled (fSTS-analogue)
  atl = afp_pooled (no AdaFace)
  v1/v2/v4/mfp = unavailable here (frozen ViT, no anatomy layers) — return
  afp_pooled as fallback alias.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HOLONetV2DinoConfig:
    dinov2_model: str = "dinov2_vits14"   # ViT-S/14, 21M params, 384-d
    image_size: int = 224                   # 224 = 16x16 patches
    patch_spatial: int = 16                 # 224 / 14 ≈ 16
    feature_dim: int = 384                  # ViT-S/14 output dim
    torch_cache: str = "/workspace/.torch_cache"

    # Face template
    template_init_std: float = 0.02
    template_ema_decay: float = 0.999
    pc_loss_weight: float = 0.1


def _load_dinov2(cfg: HOLONetV2DinoConfig) -> nn.Module:
    """Load a pre-trained DINOv2 model via torch.hub (caches to /workspace)."""
    os.environ.setdefault("TORCH_HOME", cfg.torch_cache)
    # facebookresearch/dinov2 publishes: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    model = torch.hub.load("facebookresearch/dinov2", cfg.dinov2_model,
                            trust_repo=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class FaceTemplate(nn.Module):
    """Same as model_v2.FaceTemplate — learnable T + EMA buffer over face-AFP."""

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


class HOLONetV2Dinov2(nn.Module):
    """HOLO-Net v2.1 with frozen DINOv2 backbone + FTPC template head."""

    def __init__(self, cfg: Optional[HOLONetV2DinoConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = HOLONetV2DinoConfig()
        self.cfg = cfg

        # Backbone — frozen, no_grad always
        self.backbone = _load_dinov2(cfg)

        # Face template over patch-token spatial map
        self.face_template = FaceTemplate(
            dim=cfg.feature_dim, spatial=cfg.patch_spatial,
            init_std=cfg.template_init_std, ema_decay=cfg.template_ema_decay)

    @torch.no_grad()
    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run frozen DINOv2 to get patch tokens reshaped as (B, D, H, W)."""
        # DINOv2 forward_features returns dict with 'x_norm_patchtokens'
        # of shape (B, N, D) where N = H*W patches.
        out = self.backbone.forward_features(x)
        patch = out["x_norm_patchtokens"]  # (B, N, D)
        B, N, D = patch.shape
        H = W = int(N ** 0.5)
        return patch.transpose(1, 2).reshape(B, D, H, W).contiguous()

    def forward(self, x: torch.Tensor) -> dict:
        # Frozen backbone forward (no_grad inside _extract_patch_tokens)
        afp_spatial = self._extract_patch_tokens(x)               # (B, D, H, W)
        afp_pooled = F.adaptive_avg_pool2d(afp_spatial, 1).flatten(1)  # (B, D)

        # Template mismatch — only when shapes match (always for our setup,
        # but keep the guard for robustness).
        T = self.face_template()
        delta = delta_pooled = None
        if afp_spatial.shape[-2:] == T.shape[-2:]:
            delta = afp_spatial - T[None]                          # (B, D, H, W)
            delta_pooled = F.adaptive_avg_pool2d(delta, 1).flatten(1)  # (B, D)

        out = {
            "afp_spatial": afp_spatial, "afp_pooled": afp_pooled,
            "afp": afp_pooled,            # alias
            "atl": afp_pooled,            # alias (no AdaFace)
        }
        if delta is not None:
            out["delta"] = delta
            out["delta_pooled"] = delta_pooled
            out["ffa"] = delta_pooled     # alias for eval: fSTS-analogue
        else:
            out["ffa"] = afp_pooled
        # v1/v2/v4/mfp aliases: unavailable in pure-DINOv2 setup; fallback
        for k in ("v1", "v2", "v4", "mfp"):
            out[k] = afp_pooled
        return out

    def pc_loss(self, afp_spatial: torch.Tensor,
                face_mask: torch.Tensor) -> torch.Tensor:
        """PC loss with AFP detached (T learns to predict AFP)."""
        if not bool(face_mask.any()):
            return torch.zeros((), device=afp_spatial.device)
        face_afp = afp_spatial[face_mask].detach()
        T = self.face_template()
        return (face_afp - T[None]).pow(2).mean()

    def get_layer_for_eeg_readout(self, name: str,
                                    output_dict: dict) -> torch.Tensor:
        if name in {"v1", "v2", "v4", "mfp"}:
            return output_dict["afp_pooled"]   # fallback (no anatomy layers)
        if name in {"afp", "afp_pooled"}:
            return output_dict["afp_pooled"]
        if name in {"ffa", "delta", "fsts", "delta_pooled"}:
            return output_dict["delta_pooled"]
        if name == "atl":
            return output_dict["afp_pooled"]
        raise ValueError(f"unknown layer: {name}")


def _sanity_check():
    """Forward + backward sanity + EMA update test."""
    cfg = HOLONetV2DinoConfig()
    model = HOLONetV2Dinov2(cfg)
    print(f"backbone: {cfg.dinov2_model}; feature_dim={cfg.feature_dim}; "
          f"patch_spatial={cfg.patch_spatial}")

    B = 4
    x = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    out = model(x)
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k:14s}: {tuple(v.shape)}")

    # PC loss test
    face_mask = torch.tensor([True, False, True, False])
    pc = model.pc_loss(out["afp_spatial"], face_mask)
    print(f"PC loss (2 face samples): {pc.item():.4f}")
    pc.backward()
    print("backward OK — gradient on T only (backbone frozen)")

    # Verify backbone frozen
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable/1e6:.2f}M  / total: {n_total/1e6:.2f}M  "
          f"(backbone frozen)")

    # EMA update
    model.face_template.update_ema(out["afp_spatial"][face_mask].detach())
    print(f"template_ema norm after update: {model.face_template.template_ema.norm().item():.2f}")
    print("=== DINOv2-FTPC sanity OK ===")


if __name__ == "__main__":
    _sanity_check()
