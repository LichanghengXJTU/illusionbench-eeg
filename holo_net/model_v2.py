"""holo_net/model_v2.py — HOLO-Net v2.1 (FTPC: Face-Template Predictive Coding).

Final architecture after the discussion arc ticks 70-74 with the user. The
binding design principles:

1. **NO gates anywhere.** v1's Orientation Gate + recurrent-FFA-binding +
   AdaFace identity head are all removed.
2. **NO synthetic anomaly supervision.** No anomaly-head BCE, no perturbation
   labels. Template + δ emerge from natural-image training alone (pure
   predictive coding, as the brain learns).
3. **Trained with self-supervised learning (DINOv2-style) on a general
   natural-image dataset that includes faces in natural distribution** —
   NOT face-only (the user's biological-faithfulness point: humans are
   exposed to diverse natural visuals, of which faces are a salient subset).
4. **Face template T is a learnable parameter**, EMA-regularised toward the
   mean AFP_spatial of face-containing batch samples (MediaPipe-detected).
5. **Optional self-supervised PC loss** `‖AFP_spatial − T‖²` on face samples
   (weight 0.1), to drive AFP and T to be mutually predictive on canonical
   faces.
6. **The illusion-transmission mechanism**: `δ = AFP_spatial − T`, computed
   purely as a read-out at test time. The pooled δ forms the "fSTS-analogue"
   output where the falsification §6 criteria are evaluated.
7. **Identity-equivalent readout = afp_pooled.** No AdaFace head; SSL features
   serve identity (DINOv2 is the SOTA SSL natural-image identity).

For per-layer evaluation compatibility with `eval_extract.py` and
`eval_falsification.py`, layer aliases are exposed:

  v1, v2, v4, mfp  -- pooled spatial features
  afp              -- afp_pooled (general visual readout)
  ffa              -- delta_pooled (the fSTS-analogue, illusion-signal readout)
  atl              -- afp_pooled (alias; no AdaFace in v2)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse v1's CORnet blocks + AFP block — same brain-region backbone bricks.
from holo_net.model import CORnet_V1, CORnetBlock, AFPBlock


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HOLONetV2Config:
    image_size: int = 224
    v1_channels: int = 64
    v2_channels: int = 128
    v4_channels: int = 256
    it_channels: int = 512
    v2_cycles: int = 2
    v4_cycles: int = 4
    it_cycles: int = 2
    afp_dim: int = 512
    afp_spatial: int = 7   # AFP_spatial is (D, 7, 7) for 224-input

    # Face template
    template_init_std: float = 0.02
    template_ema_decay: float = 0.999
    pc_loss_weight: float = 0.1

    # DINOv2-style SSL projection head (operates on AFP-pooled feature)
    # NOTE (tick 79): 65536 is the DINOv2 default — calibrated for ViT-L (~300M).
    # Our backbone is CORnet-S-style (~25M); 65536 prototypes is too many for
    # that capacity and the student collapses to uniform output (verified in
    # run 1 and run 2 — both hit L_dino = ln(65536) = 11.09 = uniform).
    # 4096 is the lower-capacity default (used in some DINO ViT-S configs).
    ssl_out_dim: int = 4096
    ssl_hidden_dim: int = 2048
    ssl_bottleneck_dim: int = 256


# ---------------------------------------------------------------------------
# DINOv2-style projection head (3-layer MLP + L2 + weight-normed final linear)
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    """Standard DINOv2 projection head: 3-layer MLP → bottleneck → L2-norm →
    weight-normed linear. Per Caron et al. 2021 (DINO, ICCV) and Oquab et al.
    2024 (DINOv2). The last layer has its weight magnitude fixed (weight_g = 1,
    frozen), so outputs are cosine similarities between the bottleneck unit
    vector and each of `out_dim` prototype unit vectors.
    """
    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        # Fix magnitude to 1, freeze (DINO recipe)
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


# ---------------------------------------------------------------------------
# Face template — learnable parameter + EMA-regulariser
# ---------------------------------------------------------------------------

class FaceTemplate(nn.Module):
    """Canonical upright face template T ∈ R^{D × H × W}.

    `template` is a learnable parameter (shaped by ‖AFP − T‖² PC loss and any
    downstream gradient through δ). The `template_ema` buffer is a slow
    running mean of AFP_spatial on face-detected training samples — used by
    the trainer as a regulariser anchor and (optionally) as a fallback if the
    learnable template has not yet been initialised.
    """
    def __init__(self, dim: int, spatial: int = 7, init_std: float = 0.02,
                 ema_decay: float = 0.999):
        super().__init__()
        self.dim, self.spatial, self.ema_decay = dim, spatial, ema_decay
        self.template = nn.Parameter(torch.zeros(dim, spatial, spatial))
        nn.init.normal_(self.template, std=init_std)
        self.register_buffer("template_ema", torch.zeros(dim, spatial, spatial))
        self.register_buffer("ema_initialised", torch.tensor(False))

    @torch.no_grad()
    def update_ema(self, face_afp_batch: torch.Tensor) -> None:
        """Update `template_ema` with the mean of face samples in this batch.
        Caller MUST pass only face-detected samples (face_mask filtering).
        Safe no-op when face_afp_batch is empty.
        """
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


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HOLONetV2(nn.Module):
    """HOLO-Net v2.1 (FTPC). See module docstring."""

    def __init__(self, cfg: Optional[HOLONetV2Config] = None):
        super().__init__()
        if cfg is None:
            cfg = HOLONetV2Config()
        self.cfg = cfg

        # Ventral backbone: image → V1 → V2 → V4 → IT → AFP. No LGN module
        # (Parvo path = identity at input; magno disabled by user directive).
        self.v1 = CORnet_V1(in_ch=3, out_ch=cfg.v1_channels)
        self.v2 = CORnetBlock(cfg.v1_channels, cfg.v2_channels, cfg.v2_cycles)
        self.v4 = CORnetBlock(cfg.v2_channels, cfg.v4_channels, cfg.v4_cycles)
        self.it = CORnetBlock(cfg.v4_channels, cfg.it_channels, cfg.it_cycles)
        self.afp = AFPBlock(in_ch=cfg.it_channels, out_dim=cfg.afp_dim)

        # Face template (FTPC core)
        self.face_template = FaceTemplate(
            dim=cfg.afp_dim, spatial=cfg.afp_spatial,
            init_std=cfg.template_init_std, ema_decay=cfg.template_ema_decay)

        # DINOv2 SSL projection head
        self.ssl_head = DINOHead(
            in_dim=cfg.afp_dim, out_dim=cfg.ssl_out_dim,
            hidden_dim=cfg.ssl_hidden_dim, bottleneck_dim=cfg.ssl_bottleneck_dim)

    def forward(self, x: torch.Tensor, return_ssl: bool = True) -> dict:
        # Ventral
        v1 = self.v1(x)
        v2 = self.v2(v1)
        v4 = self.v4(v2)
        it = self.it(v4)
        afp_spatial = self.afp(it)
        afp_pooled = F.adaptive_avg_pool2d(afp_spatial, 1).flatten(1)

        # Template mismatch — only valid when AFP_spatial matches the template's
        # spatial size (i.e. only on 224×224 global views). For multi-crop local
        # views (96×96 → AFP 3×3), skip δ; the template + PC loss + template-EMA
        # are designed for the global-view scale (the canonical face Gestalt).
        T = self.face_template()
        delta = delta_pooled = None
        if afp_spatial.shape[-2:] == T.shape[-2:]:
            delta = afp_spatial - T[None]                              # (B, D, H, W)
            delta_pooled = F.adaptive_avg_pool2d(delta, 1).flatten(1)  # (B, D)

        out = {
            "v1": v1, "v2": v2, "v4": v4, "mfp": it,
            "afp_spatial": afp_spatial, "afp_pooled": afp_pooled,
            "afp": afp_pooled,            # alias
            "atl": afp_pooled,            # alias for eval: no AdaFace
        }
        if delta is not None:
            out["delta"] = delta
            out["delta_pooled"] = delta_pooled
            out["ffa"] = delta_pooled     # alias for eval: fSTS-analogue
        else:
            out["ffa"] = afp_pooled       # fallback alias when δ undefined (local view)
        if return_ssl:
            out["ssl_out"] = self.ssl_head(afp_pooled)
        return out

    def pc_loss(self, afp_spatial: torch.Tensor,
                face_mask: torch.Tensor) -> torch.Tensor:
        """Self-supervised face-template predictive-coding loss:
            L_pc = mean_{i in faces} ‖stop_grad(AFP_spatial_i) − T‖²
        face_mask: (B,) bool — True for face-detected samples in the batch.
        Returns 0 when no face samples (safe no-op).

        **AFP is DETACHED** here (tick-78 fix). PC theory: the top-down prior
        (T) learns to predict the bottom-up observation (AFP); the observation
        is NOT pulled toward the prior. Without this detach, both T and AFP
        get gradient from ‖AFP−T‖² and trivially collapse toward each other
        (and toward zero). Run 1 (without detach) showed exactly this:
        L_pc → ~0.001, AFP magnitude shrinking, and L_dino consequently
        collapsed to ln(out_dim) (uniform student output)."""
        if not bool(face_mask.any()):
            return torch.zeros((), device=afp_spatial.device)
        face_afp = afp_spatial[face_mask].detach()        # ← stop grad on AFP
        T = self.face_template()
        diff = face_afp - T[None]
        return diff.pow(2).mean()

    def get_layer_for_eeg_readout(self, name: str,
                                    output_dict: dict) -> torch.Tensor:
        """Return a (B, D)-shape feature for the requested brain-region layer.
        Aliases (per the v2 design): ffa → δ-pooled (fSTS-analogue);
        atl → afp_pooled (alias for afp; no AdaFace head in v2)."""
        if name in {"v1", "v2", "v4", "mfp"}:
            spatial = output_dict[name]
            return F.adaptive_avg_pool2d(spatial, 1).flatten(1)
        if name in {"afp", "afp_pooled"}:
            return output_dict["afp_pooled"]
        if name in {"ffa", "delta", "fsts", "delta_pooled"}:
            return output_dict["delta_pooled"]
        if name == "atl":
            return output_dict["afp_pooled"]
        raise ValueError(f"unknown layer: {name}")


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

def _sanity_check():
    """Forward + backward + EMA-update sanity test. No GPU required."""
    cfg = HOLONetV2Config()
    model = HOLONetV2(cfg).train()
    print(f"\nHOLO-Net v2.1 (FTPC) config: {cfg}\n")

    B = 4
    x = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    out = model(x, return_ssl=True)
    print(f"Forward — input {tuple(x.shape)}:")
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k:14s}: {tuple(v.shape)}")

    print("\nget_layer_for_eeg_readout (eval_extract.py compatibility):")
    for name in ["v1", "v2", "v4", "mfp", "afp", "ffa", "atl"]:
        f = model.get_layer_for_eeg_readout(name, out)
        print(f"  {name:6s}: {tuple(f.shape)}")

    # PC loss test (2 of 4 samples are faces)
    face_mask = torch.tensor([True, False, True, False])
    pc = model.pc_loss(out["afp_spatial"], face_mask)
    print(f"\nPC loss (2 face samples in batch): {pc.item():.4f}")

    # Backward — SSL + PC
    loss = out["ssl_out"].sum() + cfg.pc_loss_weight * pc
    loss.backward()
    print("Backward — OK; gradients flow through MLP head + PC path")

    # EMA template update
    print("\nTemplate EMA update test:")
    print(f"  template_ema norm BEFORE: {model.face_template.template_ema.norm().item():.4f}")
    print(f"  ema_initialised BEFORE:   {bool(model.face_template.ema_initialised)}")
    with torch.no_grad():
        out2 = model(x, return_ssl=False)
    model.face_template.update_ema(out2["afp_spatial"][face_mask].detach())
    print(f"  template_ema norm AFTER:  {model.face_template.template_ema.norm().item():.4f}")
    print(f"  ema_initialised AFTER:    {bool(model.face_template.ema_initialised)}")

    # Param count
    n = sum(p.numel() for p in model.parameters())
    n_no_ssl = sum(p.numel() for p in model.parameters()
                    if not any(p is q for q in model.ssl_head.parameters()))
    print(f"\nTotal parameters: {n/1e6:.2f}M  (without SSL head: {n_no_ssl/1e6:.2f}M)")

    print("\n=== HOLO-Net v2.1 sanity check DONE ===")


if __name__ == "__main__":
    _sanity_check()
