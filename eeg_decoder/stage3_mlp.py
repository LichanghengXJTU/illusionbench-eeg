"""eeg_decoder/stage3_mlp.py — non-linear sensitivity check on Stage-3 ridge.
Per-subject 2-layer MLP ATM-EEG (1024) → DINOv2 (384) trained with cosine
loss, same train/test split as `stage3_ridge.py`.

Purpose: test the E047 [CONJECTURE] that a non-linear mapping could recover
PWI structure that linear (ridge) decoding destroys. If MLP retrieval >>
ridge OR MLP IllusionBench transfer PWI < 0.5, the EEG-bottleneck-fundamental
conclusion weakens. If MLP ≈ ridge, the conclusion strengthens.

Hyperparams (kept simple, no sweep — this is a sensitivity check):
  - Architecture: 1024 → 512 → 384 with GELU + dropout 0.1
  - Loss: 1 - cosine_similarity
  - Optimizer: AdamW lr 1e-3, weight_decay 1e-4
  - Batch: 256, 50 epochs, early stop patience 5 on validation cosine
  - 90/10 train/val split (deterministic)

Output (per the tick-84 transfer pipeline expectations):
  runs/2026-05-23_stage3-mlp_seed20260521/
    summary.json
    mlp_states.pt        # per-subject state dict
    per_dim_preservation.pt   # same key/format as ridge's, for tick-85
                                IllusionBench transfer re-use
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eeg_decoder.data import load_atm_eeg, load_image_features
from eeg_decoder.stage3_ridge import retrieval_topk, per_dim_preservation


class MLP(nn.Module):
    def __init__(self, d_in: int = 1024, d_hidden: int = 512,
                 d_out: int = 384, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(pred, target, dim=1).mean()


def train_one_subject(X_train: torch.Tensor, Y_train: torch.Tensor,
                        X_test: torch.Tensor, Y_test: torch.Tensor,
                        device: str, epochs: int, batch_size: int,
                        lr: float, weight_decay: float, dropout: float,
                        patience: int, seed: int
                        ) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Train MLP on one subject's data with early stopping; return summary
    + best-model predictions on train + test."""
    g = torch.Generator().manual_seed(seed)
    n = X_train.shape[0]
    perm = torch.randperm(n, generator=g)
    n_val = int(n * 0.1)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr = X_train[train_idx], Y_train[train_idx]
    Xva, Yva = X_train[val_idx].to(device), Y_train[val_idx].to(device)

    model = MLP(d_in=X_train.shape[1], d_hidden=512, d_out=Y_train.shape[1],
                 dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size,
                         shuffle=True, generator=g, drop_last=False)
    best_val = -1.0
    best_state = None
    bad = 0
    history = []
    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = cosine_loss(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.shape[0]
        model.eval()
        with torch.no_grad():
            val_cos = F.cosine_similarity(model(Xva), Yva, dim=1).mean().item()
        history.append({"epoch": ep, "train_loss": running / len(Xtr),
                          "val_cos": val_cos})
        if val_cos > best_val + 1e-4:
            best_val = val_cos
            best_state = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # Restore best
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Y_pred_train = model(X_train.to(device)).cpu()
        Y_pred_test = model(X_test.to(device)).cpu()
    summary = {"epochs_trained": len(history),
               "best_val_cos": best_val,
               "final_train_loss": history[-1]["train_loss"]}
    return summary, Y_pred_train, Y_pred_test, best_state


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"[device] {device}")

    Y_train = load_image_features("dinov2_vits14", "train").float()
    Y_test = load_image_features("dinov2_vits14", "test").float()
    print(f"[target] DINOv2 train {tuple(Y_train.shape)} | test {tuple(Y_test.shape)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    summary = {"per_subject": {}, "config": vars(args)}
    states = {}
    per_dim_pres = {}

    t_total = time.time()
    for sub in range(1, 11):
        sub_tag = f"sub-{sub:02d}"
        print(f"\n=== {sub_tag} ===")
        X_train = load_atm_eeg(sub, "train", average_reps=True).float()
        X_test = load_atm_eeg(sub, "test").float()
        Y_train_sub = Y_train.clone()
        t0 = time.time()
        sub_sum, Y_pred_train, Y_pred_test, state = train_one_subject(
            X_train, Y_train_sub, X_test, Y_test, device,
            args.epochs, args.batch_size, args.lr, args.weight_decay,
            args.dropout, args.patience, args.seed + sub
        )
        train_t = time.time() - t0

        true_idx = torch.arange(200)
        topk = retrieval_topk(Y_pred_test, Y_test, true_idx,
                                ks=tuple(int(k) for k in args.ks.split(",")))
        r_d = per_dim_preservation(Y_pred_test, Y_test)
        per_dim_pres[sub_tag] = r_d.cpu()
        states[sub_tag] = state

        print(f"  trained {sub_sum['epochs_trained']} epochs in {train_t:.1f}s   "
              f"val_cos {sub_sum['best_val_cos']:.3f}")
        print(f"  retrieval: top1 {topk['top1']:.3f}  top5 {topk['top5']:.3f}  "
              f"top10 {topk['top10']:.3f}  mean_rank {topk['mean_rank']:.1f}")
        print(f"  per-dim r: mean {r_d.mean().item():.3f}  std {r_d.std().item():.3f}  "
              f"min {r_d.min().item():.3f}  max {r_d.max().item():.3f}  "
              f">0.1 frac {(r_d > 0.1).float().mean().item():.3f}")

        summary["per_subject"][sub_tag] = {
            **sub_sum, **topk,
            "r_d_mean": float(r_d.mean().item()),
            "r_d_std": float(r_d.std().item()),
            "r_d_min": float(r_d.min().item()),
            "r_d_max": float(r_d.max().item()),
            "frac_r_d_gt_0p1": float((r_d > 0.1).float().mean().item()),
            "train_time_s": train_t,
        }
    total_t = time.time() - t_total

    pr = summary["per_subject"]
    summary["aggregate"] = {
        "top1_mean": float(np.mean([pr[s]["top1"] for s in pr])),
        "top1_std": float(np.std([pr[s]["top1"] for s in pr])),
        "top5_mean": float(np.mean([pr[s]["top5"] for s in pr])),
        "top5_std": float(np.std([pr[s]["top5"] for s in pr])),
        "top10_mean": float(np.mean([pr[s]["top10"] for s in pr])),
        "top10_std": float(np.std([pr[s]["top10"] for s in pr])),
        "r_d_mean_mean": float(np.mean([pr[s]["r_d_mean"] for s in pr])),
        "frac_r_d_gt_0p1_mean": float(np.mean([pr[s]["frac_r_d_gt_0p1"] for s in pr])),
        "total_time_s": total_t,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(states, out_dir / "mlp_states.pt")
    torch.save(per_dim_pres, out_dir / "per_dim_preservation.pt")

    print("\n=== AGGREGATE (10 subjects, MLP) ===")
    agg = summary["aggregate"]
    print(f"  top-1   = {agg['top1_mean']:.3f} ± {agg['top1_std']:.3f}")
    print(f"  top-5   = {agg['top5_mean']:.3f} ± {agg['top5_std']:.3f}")
    print(f"  top-10  = {agg['top10_mean']:.3f} ± {agg['top10_std']:.3f}")
    print(f"  per-dim r_d mean of means: {agg['r_d_mean_mean']:.3f}")
    print(f"  frac dims with r_d > 0.1 (mean over subs): {agg['frac_r_d_gt_0p1_mean']:.3f}")
    print(f"  total wallclock: {total_t:.1f}s")
    print(f"\n[save] summary → {out_dir}/summary.json")
    print(f"[save] mlp states → {out_dir}/mlp_states.pt")
    print(f"[save] per-dim preservation → {out_dir}/per_dim_preservation.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",
                   default="/workspace/runs/2026-05-23_stage3-mlp_seed20260521")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--ks", default="1,5,10")
    p.add_argument("--seed", type=int, default=20260521)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
