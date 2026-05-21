"""analysis/route_a_preservation.py — Q005 Route A core analysis.

For each CLIP-ViT-H/14 dimension d ∈ [0, 1024):
  preservation[d]  = average Pearson correlation across 10 THINGS-EEG2 subjects
                     between CLIP_img[:, d] and ATM_S_eeg_pred[:, d] across 200
                     test concepts. Measures how well EEG bottleneck preserves
                     dimension d of the CLIP target space.
  thatcher_load[d] = mean absolute Thatcher-perturbation magnitude in dim d
                     computed from FFHQ Thatcher CLIP-H/14 embeddings (E002),
                     normalized by the CLIP-target-space scale on dim d.

Then compute the Spearman correlation between preservation[d] and thatcher_load[d]
across all 1024 dims, with permutation null. The sign and magnitude answer Q005:

  preservation × thatcher_load correlation
  +    → Thatcher-loaded dimensions are MORE preserved by EEG (signal survives)
  -    → Thatcher-loaded dimensions are LESS preserved (EEG destroys it)
  ~0  → independent (no relationship)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


def load_atm_emb_eeg(atm_root: Path):
    """Returns dict with:
       clip_img: (200, 1024) tensor — CLIP-ViT-H/14 target image features
       eeg_preds: (10, 200, 1024) — ATM_S decoded per subject (sub-01..sub-10)
    """
    clip_test = torch.load(atm_root / "ViT-H-14_features_test.pt",
                           weights_only=False, map_location="cpu")
    clip_img = clip_test["img_features"].float()  # (200, 1024)

    eeg_preds = []
    for s in range(1, 11):
        p = atm_root / "emb_eeg" / f"ATM_S_eeg_features_sub-{s:02d}_test.pt"
        t = torch.load(p, weights_only=False, map_location="cpu").float()
        eeg_preds.append(t)
    eeg_preds = torch.stack(eeg_preds, dim=0)  # (10, 200, 1024)
    return {"clip_img": clip_img, "eeg_preds": eeg_preds}


def compute_preservation(clip_img: torch.Tensor, eeg_preds: torch.Tensor) -> np.ndarray:
    """Per-dimension Pearson correlation between CLIP target and ATM EEG output,
    averaged across subjects. Returns (1024,) numpy array of correlation coeffs."""
    n_sub, n_concept, n_dim = eeg_preds.shape
    preservation = np.zeros((n_sub, n_dim), dtype=np.float64)
    clip_img_np = clip_img.numpy()
    for s in range(n_sub):
        eeg_s = eeg_preds[s].numpy()
        for d in range(n_dim):
            x = clip_img_np[:, d]
            y = eeg_s[:, d]
            sx = x.std()
            sy = y.std()
            if sx < 1e-12 or sy < 1e-12:
                preservation[s, d] = np.nan
                continue
            preservation[s, d] = np.corrcoef(x, y)[0, 1]
    return preservation.mean(axis=0)


def compute_thatcher_loading(thatcher_npz_path: Path, manifest_csv: Path) -> np.ndarray:
    """For each FFHQ identity i:
      delta_i = CLIP(V1_upright_normal_i) - CLIP(V2_upright_thatched_i)  in R^1024
    Returns (1024,) array of mean absolute delta per dim, normalized by std
    across all V1 stimuli on that dim.
    """
    npz = np.load(thatcher_npz_path, allow_pickle=True)
    stim_ids = [str(s) for s in npz["stim_ids"]]
    embs = npz["embeddings"]  # (N, 1024)
    assert embs.shape[1] == 1024, f"expected 1024-dim, got {embs.shape}"

    manifest = pd.read_csv(manifest_csv)
    manifest = manifest.set_index("stim_id").reindex(stim_ids).reset_index()

    by_cond = {}
    for cond in ["V1_upright_normal", "V2_upright_thatched"]:
        mask = (manifest["condition"] == cond).to_numpy()
        ids = manifest.loc[mask, "identity_id"].to_numpy()
        order = np.argsort(ids)
        by_cond[cond] = embs[mask][order]  # (N_id, 1024)
        if len(by_cond[cond]) == 0:
            raise RuntimeError(f"no rows for {cond}")

    v1 = by_cond["V1_upright_normal"]
    v2 = by_cond["V2_upright_thatched"]
    assert v1.shape == v2.shape, f"shape mismatch {v1.shape} vs {v2.shape}"

    delta = v1 - v2  # (N_id, 1024)
    mean_abs = np.abs(delta).mean(axis=0)  # (1024,)

    v1_std = v1.std(axis=0) + 1e-12  # natural-image scale per dim
    loading = mean_abs / v1_std
    return loading, mean_abs, v1_std


def permutation_null(preservation, loading, n_perm=10000, seed=20260521):
    """Spearman correlation under random permutation of one axis."""
    rng = np.random.default_rng(seed)
    observed, _ = spearmanr(preservation, loading)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = spearmanr(preservation, rng.permutation(loading))[0]
    p_two = (np.abs(null) >= np.abs(observed)).mean()
    return observed, null, p_two


def main(args):
    atm_root = Path(args.atm_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] ATM embeddings from {atm_root}")
    data = load_atm_emb_eeg(atm_root)
    print(f"  clip_img: {tuple(data['clip_img'].shape)}")
    print(f"  eeg_preds: {tuple(data['eeg_preds'].shape)}")

    print("[compute] per-dim EEG preservation correlation")
    preservation = compute_preservation(data["clip_img"], data["eeg_preds"])
    print(f"  preservation: mean={preservation.mean():.3f}  median={np.median(preservation):.3f}  "
          f"min={preservation.min():.3f}  max={preservation.max():.3f}")

    print("[compute] Thatcher loading from FFHQ E002 P04 CLIP-H/14")
    loading, mean_abs, v1_std = compute_thatcher_loading(Path(args.clip_npz), Path(args.manifest))
    print(f"  loading: mean={loading.mean():.3f}  median={np.median(loading):.3f}  "
          f"min={loading.min():.3f}  max={loading.max():.3f}")

    print("[compute] Spearman(preservation, thatcher_load) + permutation null")
    rho, null, p_two = permutation_null(preservation, loading,
                                        n_perm=args.n_perm, seed=args.seed)
    null_mean = null.mean()
    null_ci = (np.percentile(null, 2.5), np.percentile(null, 97.5))

    pearson_r, pearson_p = pearsonr(preservation, loading)

    # Also: are HIGH-loading dims preserved better or worse than LOW-loading dims?
    high_mask = loading >= np.percentile(loading, 75)
    low_mask  = loading <= np.percentile(loading, 25)
    hi_pres = preservation[high_mask].mean()
    lo_pres = preservation[low_mask].mean()

    summary = {
        "spearman_rho_observed": float(rho),
        "spearman_perm_p_two_sided": float(p_two),
        "spearman_perm_null_mean": float(null_mean),
        "spearman_perm_null_ci95_lo": float(null_ci[0]),
        "spearman_perm_null_ci95_hi": float(null_ci[1]),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "preservation_mean": float(preservation.mean()),
        "preservation_median": float(np.median(preservation)),
        "loading_mean": float(loading.mean()),
        "loading_median": float(np.median(loading)),
        "high_loading_quartile_preservation_mean": float(hi_pres),
        "low_loading_quartile_preservation_mean":  float(lo_pres),
        "n_perm": int(args.n_perm),
        "n_dims": int(len(preservation)),
        "n_subjects": int(data["eeg_preds"].shape[0]),
        "n_concepts_test": int(data["eeg_preds"].shape[1]),
    }

    print()
    print("==== RESULT ====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("================")

    np.savez(out_dir / "route_a_arrays.npz",
             preservation=preservation,
             thatcher_loading=loading,
             thatcher_mean_abs=mean_abs,
             v1_std=v1_std)
    with open(out_dir / "route_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / 'route_a_arrays.npz'} and route_a_summary.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atm_root", required=True,
                   help="dir with ViT-H-14_features_test.pt + emb_eeg/ATM_S_eeg_features_sub-XX_test.pt")
    p.add_argument("--clip_npz", required=True,
                   help="P04 CLIP-H/14 embedding NPZ from E002 (thatcher_ffhq)")
    p.add_argument("--manifest", required=True, help="thatcher_manifest.csv from E002")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260521)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
