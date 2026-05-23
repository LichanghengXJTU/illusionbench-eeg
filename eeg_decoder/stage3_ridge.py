"""eeg_decoder/stage3_ridge.py — per-subject ridge ATM-EEG → DINOv2 mapping,
with closed-form fit, K-fold λ selection, and top-K retrieval evaluation
on the THINGS-EEG2 test set.

Math:
  Solve W = argmin_W ||X W - Y||² + λ ||W||²  (X = EEG, Y = DINOv2 image features)
  Closed form (X (n, d_in), Y (n, d_out)):
    W = (X^T X + λ I)^{-1} X^T Y           shape (d_in, d_out)
  Predict Y_test = X_test @ W

  Retrieval at test: cos_sim(Y_pred[i], Y_true_pool[j]) for j in 200 test stims;
  top-K = fraction of i where the true index is in top K of sorted similarities.

Per-subject design (per EEG_DECODER_STAGE3_DESIGN.md §2):
  - Each of 10 subjects has its own EEG noise; fit a separate ridge.
  - λ chosen by 5-fold CV on training set (cosine similarity criterion).
  - No cross-subject leakage.

Output:
  - eeg_decoder/runs/2026-MM-DD_stage3-ridge_seed20260521/
      ├─ summary.json          per-subject λ_best, top-K, residual stats
      ├─ ridge_weights.pt      dict {sub-XX: W (1024, 384), λ}
      └─ per_dim_preservation.pt  dict {sub-XX: r_d (384,)} — tick-84 input
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eeg_decoder.data import load_atm_eeg, load_image_features


def fit_ridge_closed(X: torch.Tensor, Y: torch.Tensor, lam: float
                       ) -> torch.Tensor:
    """Closed-form ridge. X (n, d_in), Y (n, d_out) → W (d_in, d_out)."""
    XtX = X.t() @ X
    n, d = X.shape
    XtX.diagonal().add_(lam)
    L = torch.linalg.cholesky(XtX)
    W = torch.cholesky_solve(X.t() @ Y, L)
    return W


def kfold_select_lambda(X: torch.Tensor, Y: torch.Tensor,
                          lams: list[float], k: int = 5,
                          device: str = "cuda") -> tuple[float, dict]:
    """K-fold CV; criterion is mean cosine similarity between predicted and
    true Y on held-out fold."""
    n = X.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(20260521))
    folds = torch.chunk(idx, k)
    by_lam: dict[float, list[float]] = {lam: [] for lam in lams}
    for fi in range(k):
        val = folds[fi]
        train = torch.cat([folds[fj] for fj in range(k) if fj != fi])
        Xv, Yv = X[val].to(device), Y[val].to(device)
        Xt, Yt = X[train].to(device), Y[train].to(device)
        for lam in lams:
            W = fit_ridge_closed(Xt, Yt, lam)
            Yp = Xv @ W
            cos = F.cosine_similarity(Yp, Yv, dim=1).mean().item()
            by_lam[lam].append(cos)
    means = {lam: float(np.mean(scores)) for lam, scores in by_lam.items()}
    best = max(means, key=means.get)
    return best, {"per_lam_cos": means, "criterion": "cosine"}


def retrieval_topk(Y_pred: torch.Tensor, Y_pool: torch.Tensor,
                     true_idx: torch.Tensor, ks: list[int] = (1, 5, 10)
                     ) -> dict[str, float]:
    """Top-K retrieval. Y_pred (n, d), Y_pool (m, d), true_idx (n,) gives
    the index in [0, m) of the correct match per row."""
    Yp = F.normalize(Y_pred, dim=1)
    Ypool = F.normalize(Y_pool, dim=1)
    sim = Yp @ Ypool.t()                  # (n, m)
    rank = (sim.argsort(dim=1, descending=True) == true_idx[:, None]).int().argmax(dim=1)
    out = {}
    for k in ks:
        out[f"top{k}"] = float((rank < k).float().mean().item())
    out["mean_rank"] = float(rank.float().mean().item())
    return out


def per_dim_preservation(Y_pred: torch.Tensor, Y_true: torch.Tensor
                          ) -> torch.Tensor:
    """Per-dim Pearson correlation across stimuli. (n, d) × (n, d) → (d,)."""
    Yp = Y_pred - Y_pred.mean(dim=0, keepdim=True)
    Yt = Y_true - Y_true.mean(dim=0, keepdim=True)
    num = (Yp * Yt).sum(dim=0)
    den = Yp.norm(dim=0) * Yt.norm(dim=0)
    return num / den.clamp_min(1e-8)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"[device] {device}")

    # --- target (image-side DINOv2 features) ---
    Y_train = load_image_features("dinov2_vits14", "train").float()   # (16540, 384)
    Y_test = load_image_features("dinov2_vits14", "test").float()     # (200, 384)
    print(f"[target] DINOv2 train {tuple(Y_train.shape)} | test {tuple(Y_test.shape)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    lams = [float(x) for x in args.lambdas.split(",")]
    print(f"[ridge] λ grid: {lams}")

    summary: dict = {"per_subject": {},
                     "config": {"lambdas": lams,
                                 "kfold": args.kfold,
                                 "seed": args.seed,
                                 "average_reps": args.average_reps}}
    ridge_weights: dict = {}
    per_dim_pres: dict = {}

    for sub in range(1, 11):
        sub_tag = f"sub-{sub:02d}"
        print(f"\n=== {sub_tag} ===")
        X_train = load_atm_eeg(sub, "train", average_reps=args.average_reps).float()
        X_test = load_atm_eeg(sub, "test").float()
        Y_train_sub = (Y_train if args.average_reps
                        else Y_train.repeat_interleave(4, dim=0))
        assert X_train.shape[0] == Y_train_sub.shape[0], \
            f"X {X_train.shape[0]} vs Y {Y_train_sub.shape[0]}"
        print(f"  X_train {tuple(X_train.shape)}  Y_train {tuple(Y_train_sub.shape)}  "
              f"X_test {tuple(X_test.shape)}")

        # λ selection
        t0 = time.time()
        lam_best, cv_info = kfold_select_lambda(X_train, Y_train_sub, lams,
                                                   k=args.kfold, device=device)
        cv_t = time.time() - t0
        print(f"  λ_best = {lam_best:g}   (CV {cv_t:.1f}s)   "
              f"cos by λ: {cv_info['per_lam_cos']}")

        # Final fit on all train data
        Xt = X_train.to(device)
        Yt = Y_train_sub.to(device)
        W = fit_ridge_closed(Xt, Yt, lam_best)            # (1024, 384)
        ridge_weights[sub_tag] = {"W": W.cpu(), "lambda": lam_best}

        # Test eval
        Xte = X_test.to(device)
        Y_pred_test = (Xte @ W).cpu()
        true_idx = torch.arange(200)
        topk = retrieval_topk(Y_pred_test, Y_test, true_idx,
                                ks=tuple(int(k) for k in args.ks.split(",")))
        print(f"  retrieval: top1 {topk['top1']:.3f}  top5 {topk['top5']:.3f}  "
              f"top10 {topk['top10']:.3f}  mean_rank {topk['mean_rank']:.1f}")

        # Per-dim preservation (for tick-84 IllusionBench transfer)
        r_d = per_dim_preservation(Y_pred_test, Y_test)
        per_dim_pres[sub_tag] = r_d.cpu()
        print(f"  per-dim r: mean {r_d.mean().item():.3f}  std {r_d.std().item():.3f}  "
              f"min {r_d.min().item():.3f}  max {r_d.max().item():.3f}  "
              f">0.1 frac {(r_d > 0.1).float().mean().item():.3f}")

        # Train residual (sanity — should be ~ near max corr)
        Y_pred_train = (Xt @ W).cpu()
        train_cos = F.cosine_similarity(Y_pred_train, Y_train_sub, dim=1).mean().item()

        summary["per_subject"][sub_tag] = {
            "lambda": lam_best,
            "cv_cos_by_lam": cv_info["per_lam_cos"],
            **topk,
            "train_cos_mean": train_cos,
            "r_d_mean": float(r_d.mean().item()),
            "r_d_std": float(r_d.std().item()),
            "r_d_min": float(r_d.min().item()),
            "r_d_max": float(r_d.max().item()),
            "frac_r_d_gt_0p1": float((r_d > 0.1).float().mean().item()),
        }

    # Aggregate summary
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
    }

    # Save
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(ridge_weights, out_dir / "ridge_weights.pt")
    torch.save(per_dim_pres, out_dir / "per_dim_preservation.pt")

    print("\n=== AGGREGATE (10 subjects) ===")
    agg = summary["aggregate"]
    print(f"  top-1   = {agg['top1_mean']:.3f} ± {agg['top1_std']:.3f}")
    print(f"  top-5   = {agg['top5_mean']:.3f} ± {agg['top5_std']:.3f}")
    print(f"  top-10  = {agg['top10_mean']:.3f} ± {agg['top10_std']:.3f}")
    print(f"  per-dim r_d mean of means: {agg['r_d_mean_mean']:.3f}")
    print(f"  frac dims with r_d > 0.1 (mean over subs): {agg['frac_r_d_gt_0p1_mean']:.3f}")
    print(f"\n[save] summary → {out_dir}/summary.json")
    print(f"[save] ridge weights → {out_dir}/ridge_weights.pt")
    print(f"[save] per-dim preservation → {out_dir}/per_dim_preservation.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",
                   default="/workspace/runs/2026-05-23_stage3-ridge_seed20260521")
    p.add_argument("--lambdas",
                   default="100,1000,10000,100000,1000000",
                   help="λ values to sweep over via k-fold CV")
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--ks", default="1,5,10", help="top-K values to report")
    p.add_argument("--average_reps", action="store_true",
                   help="Average the 4 train reps per stim before ridge (faster, "
                        "potentially smoother). Default off = use 66160 trials.")
    p.add_argument("--seed", type=int, default=20260521)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
