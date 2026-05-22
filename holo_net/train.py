"""HOLO-Net training script.

Stages:
  1. (optional) ImageNet pretrain — skipped if using CORnet-S released ckpt
  2. Face fine-tune on Glint360K with multi-task losses
  3. (later) Multi-layer EEG decoder training

For sanity / overfit test mode, run with --overfit_test which freezes after
N steps on a single batch and reports loss curve.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from holo_net.model import HOLONet, HOLONetConfig
from holo_net.losses import HOLONetLoss
from holo_net.data import make_glint360k_dataloader


def setup_model_and_loss(num_classes: int = 360232, device: str = "cuda") -> tuple[HOLONet, HOLONetLoss]:
    cfg = HOLONetConfig()
    model = HOLONet(cfg).to(device)
    loss_module = HOLONetLoss(cfg, num_classes=num_classes).to(device)
    return model, loss_module


def make_optimizer(model: HOLONet, loss_module: HOLONetLoss, lr: float = 1e-4):
    """AdamW with weight decay. Combine model + loss module params (AdaFace
    weight matrix is in loss_module)."""
    params = list(model.parameters()) + list(loss_module.parameters())
    return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)


def single_batch_overfit_test(
    num_steps: int = 50,
    batch_size: int = 16,
    num_classes_subset: int = 1000,
    device: str = "cuda",
):
    """Sanity check: take 1 batch, train repeatedly, verify loss drops.

    A working model+loss+optimizer pipeline should overfit a single batch
    of size 16 within ~30-50 steps (total loss should drop from ~50 to <5).
    """
    print(f"\n=== Single-batch overfit test ({num_steps} steps, B={batch_size}) ===\n")
    # Use small num_classes for fast overfit (don't allocate full 360K classifier)
    model, loss_module = setup_model_and_loss(num_classes=num_classes_subset, device=device)
    optimizer = make_optimizer(model, loss_module, lr=3e-4)
    model.train()

    # Build a single batch (random tensors as proxy for real data)
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    x_pair = torch.randn(batch_size, 3, 224, 224, device=device)
    identity = torch.randint(0, num_classes_subset, (batch_size,), device=device)
    orient = torch.randint(0, 2, (batch_size,), device=device)
    is_face = torch.ones(batch_size, dtype=torch.long, device=device)

    loss_curve = []
    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass primary batch
        out = model(x)
        # Forward pass paired batch (for view-invariance)
        out_pair = model(x_pair)

        losses = loss_module(
            out, identity, orient, is_face,
            view_invar_pairs=(out["atl"], out_pair["atl"]),
        )
        total = losses["total"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_curve.append({
            k: v.item() if torch.is_tensor(v) else v
            for k, v in losses.items()
        })
        if step % 5 == 0:
            print(f"step {step:3d}: total={losses['total'].item():.4f}  "
                  f"identity={losses['identity'].item():.4f}  "
                  f"predcode={losses['predcode'].item():.4f}  "
                  f"orientation={losses['orientation'].item():.4f}  "
                  f"face_detect={losses['face_detect'].item():.4f}  "
                  f"view_invar={losses['view_invariance'].item():.4f}  "
                  f"gist={losses['gist'].item():.4f}")

    # Diagnostic: should loss have dropped substantially
    initial_total = loss_curve[0]["total"]
    final_total = loss_curve[-1]["total"]
    drop = initial_total - final_total
    print(f"\n  Loss drop: {initial_total:.4f} → {final_total:.4f}  (Δ {drop:.4f})")
    if drop > 30:
        print(f"  ✓ Pipeline can overfit single batch (drop > 30)")
        return True
    else:
        print(f"  ⚠ Loss did not drop sufficiently — pipeline may have bug")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit_test", action="store_true",
                        help="run single-batch overfit sanity")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if args.overfit_test:
        single_batch_overfit_test(num_steps=args.num_steps,
                                   batch_size=args.batch_size)
        return

    # TODO: full training loop (Stage 1 ImageNet, Stage 2 face fine-tune)
    print("Full training loop not yet implemented — use --overfit_test for sanity.")


if __name__ == "__main__":
    main()
