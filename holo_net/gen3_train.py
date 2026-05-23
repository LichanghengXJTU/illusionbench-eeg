"""holo_net/gen3_train.py — Gen 3 from-scratch HOLO-Net backbone trainer.

Recipe (first attempt: minimal DINO + orient-aux, no multi-crop to reduce
collapse risk):
  - ViT-S/14 random init
  - 2 global views per image (224×224), NO local views in loss
  - DINO loss: KL(teacher_global_i || student_global_j) for all i ≠ j
    + centering on teacher + sharpening (teacher_temp 0.04, student 0.1)
  - Teacher = EMA of student (momentum 0.996 → 1.0 cosine schedule)
  - Orient aux loss: CE on the orient_view → orient classification head
    (weight α=0.1)
  - AdamW lr 5e-4 cosine, weight_decay 0.04
  - Warmup 10 epochs
  - AMP fp16

Pre-registered checkpoints (will be filed in EXPERIMENTS/E053 before launching):
  - Healthy descent expected: L_dino < ln(out_dim) by step 5000
    (ln(8192)=9.01; we want L_dino < 9.0 by epoch 1 = step 5000)
  - If L_dino stays ≥ 9.0 after step 5000 → COLLAPSE, kill + diagnose
  - L_orient should descend from ln(2)=0.693 to < 0.3 by step 5000 (orient
    aux is easy; if not converging, model isn't training at all)

Run:
  cd /workspace/illusionbench-eeg
  HF_HOME=/workspace/.hf_cache HF_DATASETS_CACHE=/workspace/.hf_cache/datasets \
    TORCH_HOME=/workspace/.torch_cache nohup .venv/bin/python -m holo_net.gen3_train \
    --output_dir /workspace/runs/2026-05-24_gen3_v4.0_seed20260521 \
    --epochs 50 --batch_size 256 --num_workers 12 > /workspace/runs/gen3.log 2>&1 &
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from holo_net.gen3_data import Gen3DataConfig, ImageNet1KGen3, collate, _worker_init
from holo_net.gen3_model import Gen3Backbone, Gen3ModelConfig


# ---- DINO loss (from DINOv2 paper) ----

class DINOLoss(nn.Module):
    """KL between teacher (sharpened + centered) and student outputs."""

    def __init__(self, out_dim: int, teacher_temp: float = 0.04,
                 student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs: list[torch.Tensor],
                teacher_outputs: list[torch.Tensor]) -> torch.Tensor:
        """student_outputs[i] : (B, out_dim) for global view i.
           teacher_outputs[i] : (B, out_dim) for global view i (no_grad)."""
        student_softmax = [F.log_softmax(s / self.student_temp, dim=-1)
                            for s in student_outputs]
        teacher_softmax = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
                            for t in teacher_outputs]
        total_loss = 0.0
        n_terms = 0
        for ti, t in enumerate(teacher_softmax):
            for si, s in enumerate(student_softmax):
                if si == ti:
                    continue
                total_loss = total_loss - (t * s).sum(dim=-1).mean()
                n_terms += 1
        total_loss = total_loss / max(1, n_terms)
        # Update center (EMA over teacher outputs)
        with torch.no_grad():
            all_teacher = torch.cat(teacher_outputs, dim=0)   # (G*B, out_dim)
            batch_center = all_teacher.mean(dim=0, keepdim=True)
            self.center = self.center * self.center_momentum + \
                          batch_center * (1.0 - self.center_momentum)
        return total_loss


# ---- Cosine schedules ----

def cosine_schedule(base: float, final: float, total_steps: int,
                     warmup_steps: int = 0, warmup_start: float = 0.0):
    """Returns an array of length total_steps."""
    warmup = np.linspace(warmup_start, base, warmup_steps) if warmup_steps > 0 \
             else np.array([], dtype=np.float32)
    cosine_steps = total_steps - warmup_steps
    cosine = final + 0.5 * (base - final) * \
             (1 + np.cos(np.pi * np.arange(cosine_steps) / cosine_steps))
    return np.concatenate([warmup, cosine])


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float):
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(momentum).add_(sp.data, alpha=1.0 - momentum)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    ckpt_path = out_dir / "checkpoint.pt"
    print(f"[device] {device}\n[out] {out_dir}")

    # --- data ---
    dcfg = Gen3DataConfig(cache_dir=args.cache_dir)
    ds = ImageNet1KGen3(dcfg)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=True, drop_last=True, pin_memory=True,
                    persistent_workers=(args.num_workers > 0),
                    worker_init_fn=_worker_init, collate_fn=collate)
    steps_per_epoch = len(dl)
    total_steps = steps_per_epoch * args.epochs
    print(f"[data] {len(ds):,} samples; batch {args.batch_size}; "
          f"{steps_per_epoch} steps/epoch × {args.epochs} ep = {total_steps} steps")

    # --- model: student + teacher ---
    mcfg = Gen3ModelConfig(dino_out_dim=args.dino_out_dim)
    student = Gen3Backbone(mcfg).to(device)
    teacher = Gen3Backbone(mcfg).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[model] ViT-S/14 + DINO head + orient head  "
          f"params={n_params/1e6:.1f}M  out_dim={mcfg.dino_out_dim}")

    # --- loss ---
    dino_loss_fn = DINOLoss(out_dim=mcfg.dino_out_dim,
                              teacher_temp=args.teacher_temp,
                              student_temp=args.student_temp).to(device)
    orient_loss_fn = nn.CrossEntropyLoss()

    # --- opt ---
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    warmup_steps = args.warmup_epochs * steps_per_epoch
    lr_sched = cosine_schedule(args.lr, args.lr * 0.01, total_steps,
                                 warmup_steps=warmup_steps, warmup_start=1e-6)
    mom_sched = cosine_schedule(args.teacher_momentum, 1.0, total_steps)

    # --- training loop ---
    print(f"\n[train] starting; log → {log_path}\n")
    t0 = time.time()
    step = 0
    last_log_step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            if args.max_steps and step >= args.max_steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_sched[min(step, total_steps - 1)]

            global_views = [v.to(device, non_blocking=True)
                            for v in batch["global_views"]]
            orient_view = batch["orient_view"].to(device, non_blocking=True)
            orient_label = batch["orient_label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16,
                                            enabled=(device == "cuda")):
                # Student: 2 global views
                student_outs = [student(v, mode="dino_global")["dino_proj"]
                                for v in global_views]
                # Teacher: same 2 global views (no_grad in teacher)
                with torch.no_grad():
                    teacher_outs = [teacher(v, mode="dino_global")["dino_proj"]
                                    for v in global_views]
                l_dino = dino_loss_fn(student_outs, teacher_outs)

                # Orient aux: on student
                orient_logits = student(orient_view, mode="orient")["orient_logits"]
                l_orient = orient_loss_fn(orient_logits, orient_label)
                orient_acc = (orient_logits.argmax(-1) == orient_label).float().mean().item()

                loss = l_dino + args.orient_weight * l_orient

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                opt.step()

            # Teacher EMA update
            mom = mom_sched[min(step, total_steps - 1)]
            update_teacher(student, teacher, momentum=mom)

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                steps_recent = max(1, step - last_log_step)
                last_log_step = step
                rec = {
                    "step": step, "epoch": epoch,
                    "L_dino": float(l_dino.item()),
                    "L_orient": float(l_orient.item()),
                    "L_total": float(loss.item()),
                    "orient_acc": orient_acc,
                    "lr": opt.param_groups[0]["lr"],
                    "teacher_momentum": float(mom),
                    "elapsed_s": elapsed,
                    "step_time_s": elapsed / max(1, step),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"step {step:6d} ep {epoch:2d}  "
                      f"L_dino {l_dino.item():.3f}  L_orient {l_orient.item():.3f}  "
                      f"orient_acc {orient_acc:.3f}  lr {opt.param_groups[0]['lr']:.2e}  "
                      f"mom {mom:.4f}  elap {elapsed/60:.1f}min", flush=True)

            # Save ckpt every save_every steps
            if step > 0 and step % args.save_every == 0:
                torch.save({"step": step, "epoch": epoch,
                            "student": student.state_dict(),
                            "teacher": teacher.state_dict(),
                            "dino_loss": dino_loss_fn.state_dict(),
                            "opt": opt.state_dict(),
                            "scaler": scaler.state_dict() if scaler else None,
                            "args": vars(args)}, ckpt_path)
                print(f"  [save] checkpoint at step {step}", flush=True)

            step += 1
        if args.max_steps and step >= args.max_steps:
            break

    # Final save
    torch.save({"step": step, "epoch": epoch,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "dino_loss": dino_loss_fn.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "args": vars(args)}, ckpt_path)
    print(f"\n[train] DONE — {step} steps; ckpt at {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",
                   default="/workspace/runs/2026-05-24_gen3_v4.0_seed20260521")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--dino_out_dim", type=int, default=8192)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--teacher_temp", type=float, default=0.04)
    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--teacher_momentum", type=float, default=0.996)
    p.add_argument("--orient_weight", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=3.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=0,
                   help="0 = until --epochs done")
    p.add_argument("--cache_dir", default="/workspace/.hf_cache")
    p.add_argument("--seed", type=int, default=20260521)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
