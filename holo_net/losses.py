"""Multi-task losses for HOLO-Net training.

Components:
  - L_identity: AdaFace quality-adaptive angular-margin (Kim 2022 CVPR)
  - L_predcode: Predictive coding reconstruction MSE at V1, V2, V4
  - L_orientation: upright/inverted cross-entropy
  - L_face_detect: face/non-face cross-entropy (OFA branch)
  - L_view_invariance: same-identity-different-pose contrastive
  - L_gist: PFC-Gist face/non-face cross-entropy

Total loss is a weighted sum (weights in HOLONetConfig).
"""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# AdaFace identity loss (Kim et al. 2022 CVPR, arXiv:2204.00964)
# ----------------------------------------------------------------------------

class AdaFaceLoss(nn.Module):
    """AdaFace quality-adaptive angular-margin softmax loss.

    Implements Eq. (8) of Kim et al. 2022.

    Includes margin warmup: linearly ramp margin from 0 to target margin over
    `warmup_steps`. This is critical for training stability from random init —
    without warmup, the initial cos(theta + 0.4) penalty pushes target logit
    deeply negative, creating a hard plateau early in training.
    """
    def __init__(self, embedding_dim: int = 512, num_classes: int = 360000,
                 margin: float = 0.4, scale: float = 64.0, h: float = 0.333,
                 ema_decay: float = 0.99, warmup_steps: int = 4000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.target_margin = margin
        self.scale = scale
        self.h = h
        self.warmup_steps = warmup_steps
        # Class centers (W matrix). Initialized via Xavier.
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        # EMA running statistics for embedding magnitude (pre-normalization)
        self.register_buffer("running_norm_mean", torch.tensor(1.0))
        self.register_buffer("running_norm_std", torch.tensor(0.1))
        self.ema_decay = ema_decay
        # Step counter for margin warmup
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    @property
    def margin(self) -> float:
        """Current margin value, ramped from 0 to target_margin over warmup_steps."""
        if self.warmup_steps <= 0:
            return self.target_margin
        progress = min(1.0, self.step_count.item() / self.warmup_steps)
        return self.target_margin * progress

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                pre_norm_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        embeddings: (B, embedding_dim), L2-normalized
        labels: (B,) class indices
        pre_norm_embeddings: (B, embedding_dim) raw embedding before normalization.
            If None, falls back to magnitude=1 (no adaptive margin).
        """
        # cos(theta) = embeddings @ normalized_weight.T
        W_norm = F.normalize(self.weight, p=2, dim=1)
        cos_theta = embeddings @ W_norm.T  # (B, num_classes)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        # Adaptive margin based on per-sample embedding norm (quality proxy)
        if pre_norm_embeddings is not None and self.training:
            norms = pre_norm_embeddings.norm(p=2, dim=1)  # (B,)
            with torch.no_grad():
                batch_mean = norms.mean()
                batch_std = norms.std()
                self.running_norm_mean = self.ema_decay * self.running_norm_mean + (1 - self.ema_decay) * batch_mean
                self.running_norm_std = self.ema_decay * self.running_norm_std + (1 - self.ema_decay) * batch_std
            g = (norms - self.running_norm_mean) / (self.running_norm_std + 1e-6) * self.h
            g_clip = g.clamp(-1, 1)
            m_adaptive = self.margin * g_clip  # (B,)
        else:
            m_adaptive = torch.full((cos_theta.shape[0],), self.margin,
                                     device=cos_theta.device)

        # Increment step counter for next iteration's margin (only during training)
        if self.training:
            self.step_count += 1

        # Compute theta for additive angular margin
        theta = torch.acos(cos_theta)  # (B, num_classes)
        # Apply additive margin only to target class:
        #   target logit = scale * cos(theta + m_adaptive)
        #   non-target logit = scale * cos(theta)  (unchanged)
        target_one_hot = F.one_hot(labels, num_classes=self.num_classes).bool()
        m_expand = m_adaptive.unsqueeze(1)  # (B, 1)
        target_cos_with_margin = torch.cos(theta + m_expand)  # (B, num_classes)
        cos_modified = torch.where(target_one_hot, target_cos_with_margin, cos_theta)
        # CORRECT formula (no extra subtraction; v1/v2 had buggy `- scale * m` term
        # that artificially depressed target logits and caused identity loss to
        # start at ~65 instead of ~1)
        logits = self.scale * cos_modified
        loss = F.cross_entropy(logits, labels)
        return loss


# ----------------------------------------------------------------------------
# Multi-task loss orchestrator
# ----------------------------------------------------------------------------

class HOLONetLoss(nn.Module):
    """Combines all 6 loss terms with config-specified weights.

    Usage:
      loss_module = HOLONetLoss(cfg, num_classes=360_000)
      loss_dict = loss_module(
          model_output_dict=out,            # from HOLONet.forward
          identity_labels=labels,
          orientation_labels=orient,         # 0=upright, 1=inverted
          face_detect_labels=is_face,        # 0=non-face, 1=face
          view_invar_pairs=(emb_a, emb_b),   # tuple of paired embeddings
                                              # for same-identity-different-pose
                                              # (or None to skip this loss)
      )
      total = loss_dict["total"]
      total.backward()
    """
    def __init__(self, cfg, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.identity_loss = AdaFaceLoss(
            embedding_dim=cfg.embed_dim,
            num_classes=num_classes,
            margin=0.4,
            scale=64.0,
            h=0.333,
        )

    def forward(
        self,
        model_output_dict: dict,
        identity_labels: torch.Tensor,
        orientation_labels: Optional[torch.Tensor] = None,
        face_detect_labels: Optional[torch.Tensor] = None,
        view_invar_pairs: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> dict:
        """Returns dict with all individual losses + total (weighted sum)."""
        out = model_output_dict
        losses = {}

        # L_identity (AdaFace, primary)
        embed = out["atl"]  # L2-normalized
        losses["identity"] = self.identity_loss(embed, identity_labels)

        # L_predcode (predictive-coding reconstruction MSE)
        if "pc_predictions" in out and self.cfg.use_pc_feedback:
            pc = out["pc_predictions"]
            pc_loss = 0.0
            count = 0
            if "v4_pred_from_it" in pc:
                pc_loss = pc_loss + F.mse_loss(pc["v4_pred_from_it"], out["v4"].detach())
                count += 1
            if "v2_pred_from_v4" in pc:
                pc_loss = pc_loss + F.mse_loss(pc["v2_pred_from_v4"], out["v2"].detach())
                count += 1
            if "v1_pred_from_v2" in pc:
                pc_loss = pc_loss + F.mse_loss(pc["v1_pred_from_v2"], out["v1"].detach())
                count += 1
            losses["predcode"] = pc_loss / max(count, 1)
        else:
            losses["predcode"] = torch.tensor(0.0, device=identity_labels.device)

        # L_orientation
        if "orientation_logits" in out and orientation_labels is not None:
            losses["orientation"] = F.cross_entropy(out["orientation_logits"], orientation_labels)
        else:
            losses["orientation"] = torch.tensor(0.0, device=identity_labels.device)

        # L_face_detect (OFA)
        if "ofa_logits" in out and face_detect_labels is not None:
            losses["face_detect"] = F.cross_entropy(out["ofa_logits"], face_detect_labels)
        else:
            losses["face_detect"] = torch.tensor(0.0, device=identity_labels.device)

        # L_view_invariance
        if view_invar_pairs is not None:
            emb_a, emb_b = view_invar_pairs
            # cosine distance between paired same-identity-different-pose embeddings
            cos = (F.normalize(emb_a, dim=1) * F.normalize(emb_b, dim=1)).sum(dim=1)
            losses["view_invariance"] = (1.0 - cos).mean()
        else:
            losses["view_invariance"] = torch.tensor(0.0, device=identity_labels.device)

        # L_gist (PFC face/non-face from magno)
        if "gist_face_logits" in out and face_detect_labels is not None:
            losses["gist"] = F.cross_entropy(out["gist_face_logits"], face_detect_labels)
        else:
            losses["gist"] = torch.tensor(0.0, device=identity_labels.device)

        # Total
        total = (
            self.cfg.w_identity * losses["identity"]
            + self.cfg.w_predcode * losses["predcode"]
            + self.cfg.w_orientation * losses["orientation"]
            + self.cfg.w_face_detect * losses["face_detect"]
            + self.cfg.w_view_invariance * losses["view_invariance"]
            + self.cfg.w_gist * losses["gist"]
        )
        losses["total"] = total
        return losses


# ----------------------------------------------------------------------------
# Sanity check
# ----------------------------------------------------------------------------

def _sanity_check_loss():
    """Verify forward + backward pass with synthetic data."""
    import sys
    sys.path.insert(0, ".")
    from holo_net.model import HOLONet, HOLONetConfig

    cfg = HOLONetConfig()
    model = HOLONet(cfg).train()
    loss_module = HOLONetLoss(cfg, num_classes=100)

    B = 4
    x = torch.randn(B, 3, cfg.image_size, cfg.image_size, requires_grad=False)
    ident_labels = torch.randint(0, 100, (B,))
    orient_labels = torch.randint(0, 2, (B,))
    face_labels = torch.randint(0, 2, (B,))

    out = model(x)
    print("Forward keys:", sorted(out.keys()))

    losses = loss_module(out, ident_labels, orient_labels, face_labels,
                          view_invar_pairs=None)
    print("\n--- Loss components ---")
    for k, v in losses.items():
        if torch.is_tensor(v):
            print(f"  {k:18s}: {v.item():.4f}")

    losses["total"].backward()
    print("\n--- Backward pass succeeded ---")

    # Check gradient flow into key components
    components_to_check = ["lgn.magno_proj", "v1.conv1", "v4.conv_input",
                           "afp.conv1", "ffa.attn.in_proj_weight",
                           "atl.project", "pfc_gist.mlp",
                           "pc_it_to_v4.deconv"]
    for cname in components_to_check:
        try:
            parts = cname.split(".")
            module = model
            for p in parts:
                module = getattr(module, p)
            if hasattr(module, "weight") and module.weight.grad is not None:
                grad_norm = module.weight.grad.norm().item()
                print(f"  grad norm @ {cname:30s}: {grad_norm:.6f}")
            elif hasattr(module, "in_proj_weight") and module.in_proj_weight.grad is not None:
                grad_norm = module.in_proj_weight.grad.norm().item()
                print(f"  grad norm @ {cname:30s}: {grad_norm:.6f}")
            else:
                print(f"  no weight at {cname}")
        except Exception as e:
            print(f"  SKIP {cname}: {e}")

    print("\n=== Loss sanity check DONE ===")


if __name__ == "__main__":
    _sanity_check_loss()
