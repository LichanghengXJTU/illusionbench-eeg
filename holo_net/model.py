"""HOLO-Net main architecture (PyTorch).

Each brain region is a separate nn.Module so that ablation can be done by
disabling individual components via the HOLONetConfig flags. The forward()
returns a dict of intermediate representations at each "brain region" stage
for multi-layer EEG readout (V1, V2, V4, MFP, AFP, FFA, ATL).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class HOLONetConfig:
    """Component-level ablation flags + hyperparameters."""
    # Image input
    image_size: int = 224

    # Channel widths through V1-V4-IT pipeline (CORnet-S compatible)
    v1_channels: int = 64
    v2_channels: int = 128
    v4_channels: int = 256
    it_channels: int = 512  # MFP output

    # Recurrence cycles (CORnet-S defaults)
    v2_cycles: int = 2
    v4_cycles: int = 4
    it_cycles: int = 2

    # LGN
    use_magno: bool = True       # ablation: disable magno stream
    use_parvo: bool = True       # ablation: disable parvo stream
    magno_blur_sigma: float = 2.5

    # OFA branch
    use_ofa_branch: bool = True

    # AFP
    afp_dim: int = 512

    # Orientation Gate
    use_orientation_gate: bool = True
    orientation_gate_soft: bool = True   # soft (sigmoid) vs hard (binary)

    # FFA module
    use_ffa: bool = True
    ffa_attn_heads: int = 8
    ffa_iterations: int = 3
    ffa_dim: int = 512

    # PFC-Gist
    use_pfc_gist: bool = True
    gist_dim: int = 512

    # Identity head (ATL)
    embed_dim: int = 512

    # Predictive coding feedback (custom, hand-rolled)
    use_pc_feedback: bool = True
    pc_iterations: int = 3       # number of forward+feedback cycles
    pc_loss_weight: float = 0.1  # for training stage

    # Auxiliary loss weights (recorded for reference; actual loss in losses.py)
    w_identity: float = 1.0
    w_predcode: float = 0.1
    w_orientation: float = 0.1
    w_face_detect: float = 0.05
    w_view_invariance: float = 0.2
    w_gist: float = 0.05


# ----------------------------------------------------------------------------
# LGN module — dual input streams
# ----------------------------------------------------------------------------

class LGN(nn.Module):
    """Lateral geniculate nucleus: Magno (low-SF) + Parvo (high-SF) split.

    Reference: Bar (2003) Nature Neurosci.
      Magno: Gaussian-blurred, grayscale-like, fast feedforward
      Parvo: full resolution, color-preserving, slow detail processing
    """
    def __init__(self, cfg: HOLONetConfig):
        super().__init__()
        self.cfg = cfg
        self.magno_sigma = cfg.magno_blur_sigma
        # Magno: single-channel low-SF
        self.magno_proj = nn.Conv2d(3, 8, kernel_size=1)
        # Parvo: 3-channel full-res passthrough (identity-init for clean start)
        self.parvo_proj = nn.Conv2d(3, 3, kernel_size=1)

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable Gaussian blur via convolution with Gaussian kernel."""
        sigma = self.magno_sigma
        ksize = int(6 * sigma + 1) | 1  # odd
        coords = torch.arange(ksize, dtype=x.dtype, device=x.device) - ksize // 2
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        kernel = g1d[:, None] @ g1d[None, :]  # (K, K)
        kernel = kernel.expand(x.shape[1], 1, ksize, ksize)
        pad = ksize // 2
        return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (magno, parvo) tensors.
        magno: (B, 8, H, W), low-SF
        parvo: (B, 3, H, W), high-SF
        """
        magno = parvo = x  # default fallback
        if self.cfg.use_magno:
            blurred = self._gaussian_blur(x)
            magno = self.magno_proj(blurred)
        else:
            magno = torch.zeros_like(x[:, :8] if x.shape[1] >= 8
                                      else x.new_zeros(x.shape[0], 8, x.shape[2], x.shape[3]))
        if self.cfg.use_parvo:
            parvo = self.parvo_proj(x)
        return magno, parvo


# ----------------------------------------------------------------------------
# CORnet-S blocks (V1 / V2 / V4 / IT)
# Reference: Kubilius et al. 2019 NeurIPS
# ----------------------------------------------------------------------------

class CORnet_V1(nn.Module):
    """V1: feedforward only. 7x7 conv stride 2 → maxpool 3x3 stride 2 → 3x3 conv.

    Modified to accept concat(magno_upsampled, parvo) input → 11 input ch.
    """
    def __init__(self, in_ch: int = 11, out_ch: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class CORnetBlock(nn.Module):
    """Recurrent CORnet block for V2 / V4 / IT (CORnet-S style).

    Structure (following Kubilius 2019 official impl):
      conv_input: in_ch → out_ch (1×1) + stride-2 spatial downsample
      Per-cycle bottleneck: out_ch → 4*out_ch (1×1) → 4*out_ch (3×3) → out_ch (1×1)
      Identity skip per cycle.
    """
    def __init__(self, in_ch: int, out_ch: int, n_cycles: int, downsample: bool = True):
        super().__init__()
        self.n_cycles = n_cycles
        scale = 4
        bot_ch = out_ch * scale

        # First-step input projection: in_ch → out_ch with stride-2 downsample
        stride = 2 if downsample else 1
        self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        self.bn_input = nn.BatchNorm2d(out_ch)

        # Bottleneck (used at every cycle, weight-shared)
        self.conv1 = nn.Conv2d(out_ch, bot_ch, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
        # Separate BNs per cycle for stability
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(bot_ch) for _ in range(n_cycles)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(bot_ch) for _ in range(n_cycles)])
        self.bn3 = nn.ModuleList([nn.BatchNorm2d(out_ch) for _ in range(n_cycles)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection (downsample once)
        x = self.relu(self.bn_input(self.conv_input(x)))
        for t in range(self.n_cycles):
            identity = x
            out = self.relu(self.bn1[t](self.conv1(x)))
            out = self.relu(self.bn2[t](self.conv2(out)))
            out = self.bn3[t](self.conv3(out))
            x = self.relu(out + identity)
        return x


# ----------------------------------------------------------------------------
# OFA branch: face / non-face classifier off V4
# ----------------------------------------------------------------------------

class OFABranch(nn.Module):
    """Occipital face area: face vs non-face binary classifier off V4.

    Trained with cross-entropy; gradient flow steers V4 specialization
    toward face features without changing main pathway architecture.
    """
    def __init__(self, v4_dim: int = 256):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(v4_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # face / non-face
        )

    def forward(self, v4_features: torch.Tensor) -> torch.Tensor:
        pooled = self.gap(v4_features)
        return self.mlp(pooled)  # logits (B, 2)


# ----------------------------------------------------------------------------
# AFP block: view-invariant face cell representation
# ----------------------------------------------------------------------------

class AFPBlock(nn.Module):
    """Anterior face patches: view-invariant identity coding.

    Reference: Freiwald & Tsao (2010) Science.
    Implementation: 1×1 → 3×3 conv (1 recurrent cycle) → 1×1 → spatial features
    """
    def __init__(self, in_ch: int = 512, out_dim: int = 512):
        super().__init__()
        bn_ch = 256
        self.conv1 = nn.Conv2d(in_ch, bn_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bn_ch)
        self.conv2 = nn.Conv2d(bn_ch, bn_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_ch)
        self.conv3 = nn.Conv2d(bn_ch, out_dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, mfp_features: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(mfp_features)))
        # 1 recurrent cycle
        x_residual = x
        x = self.relu(self.bn2(self.conv2(x))) + x_residual
        x = self.bn3(self.conv3(x))  # (B, out_dim, H, W)
        return x


# ----------------------------------------------------------------------------
# Orientation Gate: upright/inverted classifier producing soft gating signal
# ----------------------------------------------------------------------------

class OrientationGate(nn.Module):
    """Detects face orientation, produces gating signal that modulates
    FFA holistic-binding activation.

    Reference: Yin (1969) inversion effect; Rossion & Jacques (2008) N170.

    NOTE (tick-54 bug fix): orientation = whole-face vertical flip = a spatial
    permutation. Global average pooling is permutation-invariant, so a GAP'd
    feature vector is provably blind to orientation — verified at random init:
    orientation_logits differed by only 3e-5 between an image and its vertical
    flip, and the orientation CE loss sat dead-flat at ln 2 for 6450 training
    steps (E042 run v1). The conv stack DOES carry the orientation (afp_spatial
    flips with the input); only the pooling discarded it. Fix: pool to a coarse
    grid×grid map (top-vs-bottom layout preserved) instead of 1×1.
    """
    def __init__(self, in_dim: int = 512, soft: bool = True, grid: int = 4):
        super().__init__()
        self.grid = grid
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim * grid * grid, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # upright / inverted logits
        )
        self.soft = soft

    def forward(self, afp_spatial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (orientation_logits, gate_signal).
        orientation_logits: (B, 2) for cross-entropy training loss
        gate_signal: (B, 1, 1, 1) scalar gate ∈ [0, 1] (sigmoid) or {0, 1} (hard)
        """
        pooled = self.pool(afp_spatial)   # (B, in_dim, grid, grid) — keeps spatial layout
        logits = self.classifier(pooled)  # (B, 2)
        if self.soft:
            # Soft gate = sigmoid(upright_logit - inverted_logit)
            gate = torch.sigmoid(logits[:, 0] - logits[:, 1])
        else:
            gate = (logits[:, 0] > logits[:, 1]).float()
        gate = gate.view(-1, 1, 1, 1)
        return logits, gate


# ----------------------------------------------------------------------------
# FFA module: 3-step recurrent self-attention with Magno-Gist global query
# ----------------------------------------------------------------------------

class FFAModule(nn.Module):
    """Fusiform face area: holistic configural binding.

    Implementation:
      - Query: gist_token (B, 1, D) from Magno-PFC pathway
      - Key/Value: AFP_spatial reshaped to tokens (B, H*W, D)
      - 3 iterations of cross-attention with feedback updating K/V
      - Final output = last attention output, used as 512-d holistic embedding
    """
    def __init__(self, dim: int = 512, n_heads: int = 8, n_iters: int = 3):
        super().__init__()
        self.dim = dim
        self.n_iters = n_iters
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, gist_token: torch.Tensor, afp_spatial: torch.Tensor) -> torch.Tensor:
        """
        gist_token: (B, 1, D) magno-derived global query
        afp_spatial: (B, D, H, W) — gated AFP features
        Returns: (B, D) holistic-bound representation
        """
        B, D, H, W = afp_spatial.shape
        K = V = afp_spatial.view(B, D, H * W).transpose(1, 2)  # (B, H*W, D)
        out = gist_token  # (B, 1, D) initial state

        for _ in range(self.n_iters):
            # Cross-attention: Q=current state, K/V from current K/V
            attn_out, _ = self.attn(out, K, V)  # (B, 1, D)
            # Feedback: update K/V by adding attention output
            K = self.norm_k(K + attn_out)
            V = self.norm_v(V + attn_out)
            out = self.norm_out(out + attn_out)
        return out.squeeze(1)  # (B, D)


# ----------------------------------------------------------------------------
# PFC-Gist MLP: magno → gist_token (Bar 2003 fast top-down)
# ----------------------------------------------------------------------------

class PFCGist(nn.Module):
    """Bar (2003) magnocellular top-down gist signal.

    Magno features → global avg pool → MLP → gist_token + face/non-face logits.
    The gist_token feeds into FFA module as global query.
    """
    def __init__(self, in_ch: int = 8, gist_dim: int = 512):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, gist_dim),
        )
        # Auxiliary face/non-face logit from magno pathway
        self.face_head = nn.Linear(in_ch, 2)

    def forward(self, magno: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gist_token (B, 1, gist_dim), face_logits (B, 2))."""
        pooled = self.gap(magno).flatten(1)  # (B, in_ch)
        gist = self.mlp(pooled).unsqueeze(1)  # (B, 1, gist_dim)
        face_logits = self.face_head(pooled)  # (B, 2)
        return gist, face_logits


# ----------------------------------------------------------------------------
# ATL Identity Head: AdaFace-style angular-margin embedding
# ----------------------------------------------------------------------------

class ATLHead(nn.Module):
    """Anterior temporal lobe: identity embedding head.

    Standard architecture: L2-normalized embedding (no margin in head; the
    margin loss is applied externally in the training loop because it needs
    target labels). This module just produces the 512-d unit-vector embedding.
    """
    def __init__(self, in_dim: int = 512, out_dim: int = 512):
        super().__init__()
        if in_dim != out_dim:
            self.project = nn.Linear(in_dim, out_dim)
        else:
            self.project = nn.Identity()
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, ffa_output: torch.Tensor) -> torch.Tensor:
        """Returns 512-d L2-normalized embedding."""
        x = self.project(ffa_output)
        x = self.bn(x)
        return F.normalize(x, p=2, dim=1)


# ----------------------------------------------------------------------------
# Custom Predictive-Coding feedback (hand-rolled per user preference)
# ----------------------------------------------------------------------------

class PredictivePredictor(nn.Module):
    """Top-down predictor: from layer_{n+1} activations → predict layer_n
    activations. Hand-rolled (not Predify).

    Implementation: deconvolution (transposed conv) that upsamples and reduces
    channels. Used in training only — adds reconstruction MSE loss to total loss.
    """
    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3,
                                          stride=upsample, padding=1,
                                          output_padding=upsample - 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, target_size: Optional[tuple[int, int]] = None) -> torch.Tensor:
        x = self.bn(self.deconv(x))
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


# ----------------------------------------------------------------------------
# Main HOLO-Net model
# ----------------------------------------------------------------------------

class HOLONet(nn.Module):
    """HOLO-Net: strict bio-fidelity face vision model.

    Forward returns a dict with intermediate representations at every layer,
    enabling per-layer EEG readout (Stage 3 multi-head EEG decoder).
    """
    def __init__(self, cfg: HOLONetConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = HOLONetConfig()
        self.cfg = cfg

        # LGN dual input
        self.lgn = LGN(cfg)

        # V1 input channel: 8 (magno upsampled) + 3 (parvo) = 11
        # If magno/parvo disabled, fallback to 3 (just parvo) or 8 (just magno)
        v1_in_ch = (8 if cfg.use_magno else 0) + (3 if cfg.use_parvo else 0)
        if v1_in_ch == 0:
            v1_in_ch = 3  # safety: at least use raw image
        self.v1 = CORnet_V1(in_ch=v1_in_ch, out_ch=cfg.v1_channels)

        # V2, V4, IT(MFP) — CORnet-S recurrent bottlenecks
        self.v2 = CORnetBlock(cfg.v1_channels, cfg.v2_channels, cfg.v2_cycles)
        self.v4 = CORnetBlock(cfg.v2_channels, cfg.v4_channels, cfg.v4_cycles)
        self.it = CORnetBlock(cfg.v4_channels, cfg.it_channels, cfg.it_cycles)

        # OFA branch off V4
        self.ofa = OFABranch(v4_dim=cfg.v4_channels) if cfg.use_ofa_branch else None

        # AFP block after MFP(IT)
        self.afp = AFPBlock(in_ch=cfg.it_channels, out_dim=cfg.afp_dim)

        # Orientation Gate
        if cfg.use_orientation_gate:
            self.orient_gate = OrientationGate(in_dim=cfg.afp_dim, soft=cfg.orientation_gate_soft)
        else:
            self.orient_gate = None

        # PFC-Gist MLP from magno
        if cfg.use_pfc_gist:
            self.pfc_gist = PFCGist(in_ch=8, gist_dim=cfg.gist_dim)
        else:
            self.pfc_gist = None

        # FFA module: recurrent self-attention with magno-gist query
        if cfg.use_ffa:
            assert cfg.gist_dim == cfg.afp_dim, \
                f"gist_dim ({cfg.gist_dim}) must equal afp_dim ({cfg.afp_dim}) for attention compatibility"
            self.ffa = FFAModule(dim=cfg.ffa_dim, n_heads=cfg.ffa_attn_heads, n_iters=cfg.ffa_iterations)
        else:
            self.ffa = None

        # ATL identity head
        self.atl = ATLHead(in_dim=cfg.ffa_dim if cfg.use_ffa else cfg.afp_dim,
                            out_dim=cfg.embed_dim)

        # Custom predictive-coding feedback predictors (top-down)
        # IT → V4, V4 → V2, V2 → V1 (each upsamples 2× and reduces channels)
        if cfg.use_pc_feedback:
            self.pc_it_to_v4 = PredictivePredictor(cfg.it_channels, cfg.v4_channels, upsample=2)
            self.pc_v4_to_v2 = PredictivePredictor(cfg.v4_channels, cfg.v2_channels, upsample=2)
            self.pc_v2_to_v1 = PredictivePredictor(cfg.v2_channels, cfg.v1_channels, upsample=2)

    def _build_v1_input(self, magno: torch.Tensor, parvo: torch.Tensor) -> torch.Tensor:
        """Build V1 input by concatenating magno (upsampled to parvo res) and parvo."""
        if self.cfg.use_magno and self.cfg.use_parvo:
            if magno.shape[-2:] != parvo.shape[-2:]:
                magno_up = F.interpolate(magno, size=parvo.shape[-2:], mode="bilinear",
                                         align_corners=False)
            else:
                magno_up = magno
            return torch.cat([magno_up, parvo], dim=1)
        if self.cfg.use_parvo:
            return parvo
        if self.cfg.use_magno:
            return F.interpolate(magno, size=parvo.shape[-2:] if parvo is not None
                                  else (self.cfg.image_size, self.cfg.image_size),
                                  mode="bilinear", align_corners=False)
        raise RuntimeError("Both magno and parvo are disabled; need at least one.")

    def forward(self, x: torch.Tensor, return_all: bool = True) -> dict:
        """Forward pass.
        x: (B, 3, H, W) input image
        Returns dict with keys:
          - "v1", "v2", "v4", "mfp(it)", "afp_spatial", "afp", "ffa", "atl"
          - "orientation_logits", "ofa_logits", "gist_face_logits" (aux outputs)
          - "pc_predictions": dict with predictor outputs for training loss
        """
        out = {}

        # LGN dual stream
        magno, parvo = self.lgn(x)
        out["magno"] = magno
        out["parvo"] = parvo

        # V1
        v1_in = self._build_v1_input(magno, parvo)
        v1 = self.v1(v1_in)
        out["v1"] = v1

        # V2, V4
        v2 = self.v2(v1)
        out["v2"] = v2
        v4 = self.v4(v2)
        out["v4"] = v4

        # OFA branch (auxiliary face/non-face)
        if self.ofa is not None:
            out["ofa_logits"] = self.ofa(v4)

        # IT (MFP)
        it = self.it(v4)
        out["mfp"] = it

        # AFP block
        afp_spatial = self.afp(it)  # (B, afp_dim, H, W)
        out["afp_spatial"] = afp_spatial

        # AFP pooled (1D) for downstream / identity training
        afp_pooled = F.adaptive_avg_pool2d(afp_spatial, 1).flatten(1)
        out["afp"] = afp_pooled

        # Orientation Gate
        if self.orient_gate is not None:
            orient_logits, gate = self.orient_gate(afp_spatial)
            out["orientation_logits"] = orient_logits
            afp_gated_spatial = afp_spatial * gate  # (B, D, H, W) gated
        else:
            afp_gated_spatial = afp_spatial

        # PFC-Gist
        if self.pfc_gist is not None:
            gist_token, gist_face_logits = self.pfc_gist(magno)  # (B, 1, gist_dim), (B, 2)
            out["gist_token"] = gist_token
            out["gist_face_logits"] = gist_face_logits
        else:
            # If gist disabled, use AFP global pool as query
            gist_token = afp_pooled.unsqueeze(1)

        # FFA: holistic binding via cross-attention
        if self.ffa is not None:
            ffa_out = self.ffa(gist_token, afp_gated_spatial)
            out["ffa"] = ffa_out  # (B, ffa_dim)
            atl_input = ffa_out
        else:
            atl_input = afp_pooled
            out["ffa"] = atl_input

        # ATL identity embedding
        embed = self.atl(atl_input)
        out["atl"] = embed
        out["embedding"] = embed  # alias for consistency

        # Predictive-coding feedback predictions (for training loss)
        if self.training and self.cfg.use_pc_feedback:
            pc = {}
            pc["v4_pred_from_it"] = self.pc_it_to_v4(it, target_size=v4.shape[-2:])
            pc["v2_pred_from_v4"] = self.pc_v4_to_v2(v4, target_size=v2.shape[-2:])
            pc["v1_pred_from_v2"] = self.pc_v2_to_v1(v2, target_size=v1.shape[-2:])
            out["pc_predictions"] = pc

        return out

    def get_layer_for_eeg_readout(self, name: str, output_dict: dict) -> torch.Tensor:
        """Return spatial-pooled 1D feature for a named layer (for EEG readout)."""
        if name in ["v1", "v2", "v4", "mfp"]:
            spatial = output_dict[name]
            return F.adaptive_avg_pool2d(spatial, 1).flatten(1)
        if name == "afp":
            return output_dict["afp"]
        if name in ["ffa", "atl"]:
            return output_dict[name]
        raise ValueError(f"Unknown layer: {name}")


# ----------------------------------------------------------------------------
# CLI sanity check: forward pass with random input, print all layer shapes
# ----------------------------------------------------------------------------

def _sanity_check():
    """Run a single forward pass and print all layer shapes."""
    cfg = HOLONetConfig()
    model = HOLONet(cfg)
    model.eval()
    print(f"\nHOLO-Net config: {cfg}\n")

    B = 4
    x = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    with torch.no_grad():
        out = model(x)

    print(f"\nInput shape: {x.shape}\n")
    print(f"--- Layer outputs (B={B}) ---")
    for k in ["magno", "parvo", "v1", "v2", "v4", "mfp", "afp_spatial",
              "afp", "ffa", "atl", "ofa_logits", "orientation_logits",
              "gist_token", "gist_face_logits", "embedding"]:
        if k in out:
            shape = tuple(out[k].shape) if hasattr(out[k], "shape") else type(out[k]).__name__
            print(f"  {k:24s}: {shape}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params / 1e6:.2f}M")

    # Verify embedding is L2-normalized
    emb = out["atl"]
    norms = emb.norm(p=2, dim=1)
    print(f"\nEmbedding norms (should all be 1.0): {norms}")

    # Ablation API test
    print("\n--- Ablation API test (disable PC + FFA + magno) ---")
    cfg_abl = HOLONetConfig(use_pc_feedback=False, use_ffa=False, use_magno=False)
    model_abl = HOLONet(cfg_abl)
    model_abl.eval()
    with torch.no_grad():
        out_abl = model_abl(x)
    print(f"  ATL embedding: {out_abl['atl'].shape}")
    print(f"  FFA: {out_abl['ffa'].shape} (= afp_pooled since FFA disabled)")
    print(f"  magno: {out_abl['magno'].shape}")
    n_params_abl = sum(p.numel() for p in model_abl.parameters())
    print(f"  ablation parameter count: {n_params_abl / 1e6:.2f}M (vs full {n_params / 1e6:.2f}M)")

    print("\n=== Sanity check DONE ===")


if __name__ == "__main__":
    _sanity_check()
