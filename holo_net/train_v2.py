"""holo_net/train_v2.py — DINOv2-style SSL + FTPC training for HOLO-Net v2.1.

Composition:
  - Student / teacher: two HOLONetV2 instances; teacher is a momentum EMA
    of student (no gradient).
  - DINO loss (Caron et al. 2021 / DINOv2 Oquab et al. 2024): cross-entropy
    between teacher's softmax(centered) and student's log_softmax over all
    teacher-global × student-(global|local) view pairs, excluding self pairs.
  - PC loss: ‖AFP_spatial − T‖² on face-detected samples (weight 0.1).
  - Template EMA: face-mask-filtered AFP_spatial accumulated into the
    `face_template.template_ema` buffer (decay 0.999).

Defaults: 100 epochs ImageNet-1K, batch 256, AdamW lr 5e-4, AMP fp16,
warmup 10 ep, cosine to 0; checkpoint every 5 epochs.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from holo_net.model_v2 import HOLONetV2, HOLONetV2Config
from holo_net.data_v2 import ImageNetFTPCConfig, make_dataloader


# ---------------------------------------------------------------------------
# DINO loss
# ---------------------------------------------------------------------------

class DINOLoss(nn.Module):
    """DINO/DINOv2 self-distillation loss.

    Maintains a centering buffer (EMA of teacher mean logits) to prevent
    collapse. Forward expects:
      teacher_out: (n_global × B, out_dim) — teacher applied to each global view.
      student_out: ((n_global + n_local) × B, out_dim) — student applied to all views.

    Loss = mean over all (t_idx, s_idx) with t_idx != s_idx of
            −sum_d  softmax_d((teacher_t − c)/T_t) * log_softmax_d(student_s / T_s)
    """

    def __init__(self, out_dim: int, n_global: int, n_local: int,
                 student_temp: float = 0.1, teacher_temp: float = 0.04,
                 center_momentum: float = 0.9):
        super().__init__()
        self.out_dim = out_dim
        self.n_global, self.n_local = n_global, n_local
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor) -> None:
        """EMA the teacher's per-dim mean logits into the centering buffer."""
        batch_center = teacher_out.float().mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum)

    def forward(self, student_out: torch.Tensor,
                teacher_out: torch.Tensor) -> torch.Tensor:
        teacher_centered = (teacher_out.float() - self.center) / self.teacher_temp
        teacher_p = F.softmax(teacher_centered, dim=-1).detach()
        teacher_chunks = teacher_p.chunk(self.n_global, dim=0)

        student_log_p = F.log_softmax(student_out.float() / self.student_temp, dim=-1)
        student_chunks = student_log_p.chunk(self.n_global + self.n_local, dim=0)

        total, n_pairs = 0.0, 0
        for t_idx in range(self.n_global):
            for s_idx in range(self.n_global + self.n_local):
                if t_idx == s_idx:
                    continue
                term = -(teacher_chunks[t_idx] * student_chunks[s_idx]).sum(dim=-1).mean()
                total = total + term
                n_pairs += 1
        return total / n_pairs


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def cosine_warmup_lr(step: int, max_steps: int, warmup_steps: int,
                     peak_lr: float, end_lr: float = 0.0) -> float:
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return end_lr + (peak_lr - end_lr) * 0.5 * (
        1.0 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    print(f"[seed] {args.seed}")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    ckpt_path = out_dir / "checkpoint.pt"

    # Models — student + teacher (EMA copy, no grad)
    mcfg = HOLONetV2Config()
    student = HOLONetV2(mcfg).to(device)
    teacher = HOLONetV2(mcfg).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[model] HOLO-Net v2.1 (FTPC); student params = {n_params/1e6:.2f}M")

    # DataLoader
    dl_cfg = ImageNetFTPCConfig(
        cache_dir="/workspace/.hf_cache",
        dataset_name="evanarlian/imagenet_1k_resized_256",
        split="train",
        n_global=2, n_local=8,
        global_size=224, local_size=96,
        detect_face=True)
    dl = make_dataloader(dl_cfg, batch_size=args.batch_size,
                         num_workers=args.num_workers)
    n_per_epoch = len(dl)
    total_steps = args.epochs * n_per_epoch
    warmup_steps = args.warmup_epochs * n_per_epoch
    print(f"[data] {len(dl.dataset):,} samples × {args.epochs} ep / batch {args.batch_size} "
          f"→ {n_per_epoch} steps/ep, total {total_steps:,}")
    print(f"[lr] AdamW peak {args.lr}; warmup {warmup_steps} steps → cosine to 0")

    # Optimizer + loss + AMP
    optim = torch.optim.AdamW(student.parameters(),
                               lr=0.0, weight_decay=args.weight_decay)
    dino_loss = DINOLoss(out_dim=mcfg.ssl_out_dim,
                          n_global=dl_cfg.n_global, n_local=dl_cfg.n_local,
                          student_temp=args.student_temp,
                          teacher_temp=args.teacher_temp).to(device)
    scaler = GradScaler(enabled=args.use_amp)

    # Optional: resume
    start_epoch, start_step = 0, 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ck["student"])
        teacher.load_state_dict(ck["teacher"])
        if "dino_loss" in ck: dino_loss.load_state_dict(ck["dino_loss"])
        if "optim" in ck: optim.load_state_dict(ck["optim"])
        start_epoch = int(ck.get("epoch", 0))
        start_step = int(ck.get("step", 0))
        print(f"[resume] from epoch {start_epoch} step {start_step}")

    # Training loop
    step = start_step
    t0 = time.time()
    last_t = t0
    print(f"\n[train] starting; logging to {log_path}\n")

    for epoch in range(start_epoch, args.epochs):
        for batch in dl:
            if step >= total_steps:
                break

            # LR schedule
            lr = cosine_warmup_lr(step, total_steps, warmup_steps, args.lr)
            for g in optim.param_groups:
                g["lr"] = lr

            # Move to GPU
            gv = [v.to(device, non_blocking=True) for v in batch["global_views"]]
            lv = [v.to(device, non_blocking=True) for v in batch["local_views"]]
            face_mask = batch["face_mask"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with autocast(enabled=args.use_amp, dtype=torch.float16):
                # Teacher — only global views, no grad
                with torch.no_grad():
                    teacher_outs = [teacher(g, return_ssl=True)["ssl_out"]
                                    for g in gv]
                    teacher_out = torch.cat(teacher_outs, dim=0)

                # Student — global views (keep afp_spatial for PC + template EMA)
                student_g = [student(g, return_ssl=True) for g in gv]
                student_l = [student(l, return_ssl=True) for l in lv]
                student_ssl = ([o["ssl_out"] for o in student_g]
                               + [o["ssl_out"] for o in student_l])
                student_out = torch.cat(student_ssl, dim=0)

                loss_dino = dino_loss(student_out, teacher_out)

                # PC loss on the first global view's AFP_spatial, on face samples only
                afp_spatial_g0 = student_g[0]["afp_spatial"]
                loss_pc = student.pc_loss(afp_spatial_g0, face_mask)
                loss_total = loss_dino + mcfg.pc_loss_weight * loss_pc

            scaler.scale(loss_total).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            scaler.step(optim)
            scaler.update()

            # Centering update (no grad)
            dino_loss.update_center(teacher_out)

            # Teacher EMA from student (no grad)
            with torch.no_grad():
                m = args.teacher_momentum
                for sp, tp in zip(student.parameters(), teacher.parameters()):
                    tp.data.mul_(m).add_(sp.data, alpha=1.0 - m)

            # Template EMA on face samples (no grad inside the buffer update)
            student.face_template.update_ema(afp_spatial_g0[face_mask].detach())

            # Log every 50 steps
            if step % 50 == 0:
                now = time.time()
                step_t = (now - last_t) / 50 if step > start_step else 0.0
                last_t = now
                elapsed = now - t0
                rec = {"step": step, "epoch": epoch, "lr": lr,
                       "loss_dino": float(loss_dino.item()),
                       "loss_pc": float(loss_pc.item()),
                       "loss_total": float(loss_total.item()),
                       "n_faces_in_batch": int(face_mask.sum().item()),
                       "step_time_s": step_t, "elapsed_s": elapsed,
                       "template_ema_norm":
                            float(student.face_template.template_ema.norm().item()),
                       "template_norm":
                            float(student.face_template.template.norm().item())}
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"step {step:6d} ep {epoch:3d} lr {lr:.2e} "
                      f"L_dino {loss_dino.item():.3f} L_pc {loss_pc.item():.4f} "
                      f"faces {int(face_mask.sum().item())} "
                      f"step_t {step_t:.2f}s elapsed {elapsed/60:.1f}min "
                      f"T_ema {student.face_template.template_ema.norm().item():.2f}")

            step += 1

        # End-of-epoch checkpoint
        if (epoch + 1) % args.ckpt_every_epoch == 0 or epoch == args.epochs - 1:
            torch.save({
                "step": step, "epoch": epoch + 1,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "dino_loss": dino_loss.state_dict(),
                "optim": optim.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"[ckpt] saved epoch {epoch+1} → {ckpt_path}")

        if step >= total_steps:
            break

    # Final
    torch.save({
        "step": step, "epoch": args.epochs,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    print(f"\n[train] DONE — final checkpoint at {ckpt_path}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/workspace/holonetv2")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--teacher_temp", type=float, default=0.04)
    p.add_argument("--teacher_momentum", type=float, default=0.996)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--ckpt_every_epoch", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
