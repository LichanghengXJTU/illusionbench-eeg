"""HOLO-Net training script.

Stages:
  1. (skipped) ImageNet pretrain — use CORnet-S released ckpt for V1-IT init
  2. Face fine-tune on Glint360K with multi-task losses (this script)
  3. (later, separate script) Multi-layer EEG decoder training

Modes:
  --overfit_test     single-batch sanity test (verified in tick 42)
  --train            full Stage 2 training loop
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from holo_net.model import HOLONet, HOLONetConfig
from holo_net.losses import HOLONetLoss
from holo_net.data import make_glint360k_dataloader


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def setup_model_and_loss(num_classes: int, device: str = "cuda",
                          minimal: bool = False,
                          enable: set | None = None) -> tuple[HOLONet, HOLONetLoss]:
    """Build model + loss.
    minimal=True disables all bio-fidelity additions (LGN-Magno, OFA, FFA,
    Orientation Gate, PC feedback, PFC-Gist) → CORnet-S + AFP + AdaFace baseline.
    enable={...} starts from the minimal base and selectively re-enables named
    components (for component-isolation ablations): any of
    {magno, ofa, orient, gist, ffa, pc}.
    """
    cfg = HOLONetConfig()
    if minimal or enable:
        cfg.use_magno = False
        cfg.use_ofa_branch = False
        cfg.use_orientation_gate = False
        cfg.use_pfc_gist = False
        cfg.use_ffa = False
        cfg.use_pc_feedback = False
        cfg.w_predcode = 0.0
        cfg.w_orientation = 0.0
        cfg.w_face_detect = 0.0
        cfg.w_view_invariance = 0.0
        cfg.w_gist = 0.0
    if enable:
        if "magno" in enable:
            cfg.use_magno = True
        if "ofa" in enable:
            cfg.use_ofa_branch = True
            cfg.w_face_detect = 0.05
        if "orient" in enable:
            cfg.use_orientation_gate = True
            cfg.w_orientation = 0.1
        if "gist" in enable:
            cfg.use_pfc_gist = True
            cfg.w_gist = 0.05
        if "ffa" in enable:
            cfg.use_ffa = True
        if "pc" in enable:
            cfg.use_pc_feedback = True
            cfg.w_predcode = 0.1
    model = HOLONet(cfg).to(device)
    loss_module = HOLONetLoss(cfg, num_classes=num_classes).to(device)
    return model, loss_module


def make_optimizer(model: HOLONet, loss_module: HOLONetLoss, lr: float = 0.1,
                   optimizer_type: str = "sgd"):
    """Optimizer. SGD with momentum is standard for AdaFace at scale.
    AdamW lr=1e-4 was tried and found too slow for 360K-class classification.
    """
    params = list(model.parameters()) + list(loss_module.parameters())
    if optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    raise ValueError(f"unknown optimizer_type: {optimizer_type}")


def init_from_cornet(model: HOLONet) -> int:
    """Try to initialize CORnet-S backbone (V1, V2, V4, IT) from released ckpt.

    Returns number of layers successfully initialized.
    """
    try:
        import torch.hub
        cornet_url = "https://github.com/dicarlolab/CORnet/releases/download/v1.0/cornet_s_epoch43.pth.tar"
        # Try local cached path first
        local_paths = [
            "/workspace/eeg_repos/CORnet/cornet/cornet_s_epoch43.pth.tar",
            "/workspace/models/cornet_s_epoch43.pth.tar",
            "/root/.cache/torch/hub/checkpoints/cornet_s_epoch43.pth.tar",
        ]
        loaded = None
        for p in local_paths:
            if os.path.exists(p):
                loaded = torch.load(p, map_location="cpu", weights_only=False)
                print(f"  loaded CORnet ckpt from {p}")
                break
        if loaded is None:
            print(f"  CORnet ckpt not found locally; skipping init (training from scratch)")
            return 0
        # CORnet checkpoint has state_dict under "state_dict" key
        sd = loaded["state_dict"] if "state_dict" in loaded else loaded
        # Keys have "module." prefix from DataParallel
        sd_clean = {k.replace("module.", ""): v for k, v in sd.items()}
        # HOLO-Net's V1/V2/V4/IT may have different naming; do best-effort partial load
        # For now just report what's in the ckpt and let user verify
        print(f"  CORnet ckpt has {len(sd_clean)} keys, e.g.: {list(sd_clean.keys())[:5]}")
        return 0  # for now don't actually load to avoid mismatch (TODO: implement key remapping)
    except Exception as e:
        print(f"  init_from_cornet failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# Single-batch overfit sanity (kept from tick 42)
# ---------------------------------------------------------------------------

def single_batch_overfit_test(num_steps: int = 30, batch_size: int = 16,
                               num_classes_subset: int = 1000, device: str = "cuda"):
    print(f"\n=== Single-batch overfit test ({num_steps} steps, B={batch_size}) ===\n")
    model, loss_module = setup_model_and_loss(num_classes=num_classes_subset, device=device)
    optimizer = make_optimizer(model, loss_module, lr=3e-4)
    model.train()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    x_pair = torch.randn(batch_size, 3, 224, 224, device=device)
    identity = torch.randint(0, num_classes_subset, (batch_size,), device=device)
    orient = torch.randint(0, 2, (batch_size,), device=device)
    is_face = torch.ones(batch_size, dtype=torch.long, device=device)
    for step in range(num_steps):
        optimizer.zero_grad()
        out = model(x)
        out_pair = model(x_pair)
        losses = loss_module(out, identity, orient, is_face,
                              view_invar_pairs=(out["atl"], out_pair["atl"]))
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 5 == 0:
            print(f"  step {step}: total={losses['total'].item():.4f}")
    print(f"\nOverfit drop: {losses['total'].item():.4f} (target < 5)\n")


# ---------------------------------------------------------------------------
# Full Stage 2 face fine-tune training loop
# ---------------------------------------------------------------------------

def train_stage2(args):
    """Stage 2: face fine-tune on Glint360K with multi-task loss."""
    device = "cuda"
    print(f"\n=== HOLO-Net Stage 2 — face fine-tune ===")
    print(f"Glint360K shards: {args.shard_pattern}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size}, lr={args.lr}, max_steps={args.max_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    ckpt_path = output_dir / "checkpoint.pt"

    # Model + loss
    enable = set(c.strip() for c in args.enable.split(",") if c.strip()) if args.enable else None
    model, loss_module = setup_model_and_loss(num_classes=args.num_classes,
                                                device=device, minimal=args.minimal,
                                                enable=enable)
    if enable:
        print(f"  ABLATION MODE: minimal base + enabled components: {sorted(enable)}")
    elif args.minimal:
        print("  MINIMAL MODE: LGN-Magno, OFA, FFA, OrientGate, PC, PFC all disabled")
    # Optional CORnet init (currently no-op pending key remapping; trains from scratch)
    init_from_cornet(model)

    optimizer = make_optimizer(model, loss_module, lr=args.lr,
                                 optimizer_type=args.optimizer)
    # Linear warmup followed by cosine decay
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=args.use_amp)

    # Override view_invariance weight to reduce interference with orient gate
    loss_module.cfg.w_view_invariance = args.w_view_invariance

    # Dataloader
    dl = make_glint360k_dataloader(
        shard_pattern=args.shard_pattern,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        p_inverted=0.5,
        pair_for_view_invariance=True,
        max_class_id=(args.num_classes - 1 if args.num_classes < 360232 else None),
    )

    # Training loop
    model.train()
    step = 0
    t0 = time.time()
    last_t = t0
    running_total = 0.0
    log_records = []

    print(f"\nStarting training. Logging to {log_path}\n")
    for batch in dl:
        if step >= args.max_steps:
            break

        x = batch["image"].to(device, non_blocking=True)
        x_pair = batch["image_paired"].to(device, non_blocking=True)
        identity = batch["identity"].to(device, non_blocking=True)
        orientation = batch["orientation"].to(device, non_blocking=True)
        is_face = batch["is_face"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=args.use_amp, dtype=torch.float16):
            out = model(x)
            # For view-invariance, only need ATL embedding from paired view
            with torch.no_grad():
                out_pair_emb = model(x_pair)["atl"]
            losses = loss_module(out, identity, orientation, is_face,
                                  view_invar_pairs=(out["atl"], out_pair_emb))

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(loss_module.parameters()), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_total += losses["total"].item()

        # Log every 50 steps
        if step % 50 == 0:
            elapsed = time.time() - t0
            step_time = (time.time() - last_t) / max(50, 1) if step > 0 else 0
            avg_total = running_total / max(1, step % 50 + 1)
            record = {
                "step": step,
                "elapsed_s": elapsed,
                "step_time_s": step_time,
                "total": losses["total"].item(),
                "identity": losses["identity"].item(),
                "predcode": losses["predcode"].item(),
                "orientation": losses["orientation"].item(),
                "face_detect": losses["face_detect"].item(),
                "view_invariance": losses["view_invariance"].item(),
                "gist": losses["gist"].item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_records.append(record)
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"step {step:6d}  total={record['total']:.4f}  "
                  f"id={record['identity']:.4f}  pc={record['predcode']:.4f}  "
                  f"ori={record['orientation']:.4f}  fd={record['face_detect']:.4f}  "
                  f"vi={record['view_invariance']:.4f}  g={record['gist']:.4f}  "
                  f"lr={record['lr']:.2e}  step_t={step_time:.2f}s  elapsed={elapsed/60:.1f}min")
            last_t = time.time()
            running_total = 0.0

        # Checkpoint every 5000 steps
        if step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "loss_module": loss_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            print(f"  Saved checkpoint at step {step} to {ckpt_path}")

        step += 1

    # Final checkpoint
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "loss_module": loss_module.state_dict(),
    }, ckpt_path)
    print(f"\nTraining DONE. Final checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["overfit_test", "train"], default="train")
    parser.add_argument("--shard_pattern", default="/workspace/glint360k/glint360k-{0000..1384}.tar.gz")
    parser.add_argument("--output_dir", default="/workspace/holo_net_stage2")
    parser.add_argument("--num_classes", type=int, default=360232)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--ckpt_every", type=int, default=5000)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--w_view_invariance", type=float, default=0.0,
                        help="weight for view-invariance aux loss. v2 with 0.05 caused embedding collapse — disabled by default until contrastive negatives are added.")
    parser.add_argument("--minimal", action="store_true",
                        help="Strip all bio-fidelity additions; CORnet-S + AFP + AdaFace baseline only")
    parser.add_argument("--enable", default="",
                        help="comma-list of components to re-enable on the minimal base "
                             "(for component-isolation ablations): any of magno,ofa,orient,gist,ffa,pc")
    parser.add_argument("--seed", type=int, default=20260521,
                        help="seed for model init + main-process RNG (the WebDataset "
                             "shardshuffle + multi-worker data stream stays stochastic)")
    args = parser.parse_args()

    import numpy as np
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    print(f"[seed] torch/cuda/numpy seeded with {args.seed}")

    if args.mode == "overfit_test":
        single_batch_overfit_test()
    else:
        train_stage2(args)


if __name__ == "__main__":
    main()
