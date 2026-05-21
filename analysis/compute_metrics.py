"""analysis/compute_metrics.py — compute ISI from extracted embeddings.

For each model NPZ, compute:
  - within-identity distance d_within(orientation) = D(V_normal_i, V_thatched_i) per identity
  - cross-identity baseline d_between(orientation) over pairs of distinct identities (same condition)
  - d'(orientation) = (mean_within - mean_between) / pooled_std
  - ISI = d'_upright / d'_inverted
  - bootstrap 95% CI on ISI by resampling identities with replacement

Human reference (Carbon 2005, Thompson 1980): ISI ≈ 4-5.
Pure image-statistics expectation: ISI ≈ 1 (no orientation × thatcher interaction).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Embeddings expected L2-normalized; returns (N, N) cosine distance."""
    sim = embeddings @ embeddings.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def dprime(within: np.ndarray, between: np.ndarray) -> float:
    """Signal-detection-style d' with pooled std."""
    if len(within) < 2 or len(between) < 2:
        return float("nan")
    m_w = within.mean()
    m_b = between.mean()
    pooled_var = 0.5 * (within.var(ddof=1) + between.var(ddof=1))
    if pooled_var <= 0:
        return float("nan")
    return float((m_w - m_b) / np.sqrt(pooled_var))


def cross_id_distances(D: np.ndarray, idx_list: list[int]) -> np.ndarray:
    """All upper-triangular pair distances between rows in idx_list."""
    sub = D[np.ix_(idx_list, idx_list)]
    n = len(idx_list)
    iu = np.triu_indices(n, k=1)
    return sub[iu]


def compute_isi(embeddings: np.ndarray, manifest: pd.DataFrame,
                n_bootstrap: int = 1000, seed: int = 20260521) -> dict:
    """Compute Illusion Sensitivity Index for the Thatcher paradigm.

    Primary metric (ISI_pert): mean Thatcher-perturbation distance in upright vs inverted.
        ISI_pert = mean_i d(V1_i, V2_i) / mean_i d(V3_i, V4_i)
        Interpretation: how much more does the embedding "move" when Thatcher is applied
        to an upright face vs an inverted face. Pixel-baseline → 1.0 (orientation-invariant).
        Humans/perceptual systems → >> 1 (upright Thatcher is detectable, inverted is not).

    Secondary metric (ISI_zscore): a normalized version. We z-score the Thatcher distance
    per-orientation using the cross-identity distance distribution as the noise floor, then
    take the ratio. This controls for orientation-dependent embedding "spread".
    """
    rng = np.random.default_rng(seed)

    stim_to_idx = {sid: i for i, sid in enumerate(manifest["stim_id"].to_list())}
    D = cosine_distance_matrix(embeddings)
    identities = sorted(manifest["identity_id"].unique().tolist())

    def lookup(ident_id: int, cond: str) -> int:
        sid = f"thatcher_id{ident_id:04d}_{cond}"
        return stim_to_idx[sid]

    # Per-identity Thatcher perturbation distances
    d_up_pert = np.array([D[lookup(i, "V1_upright_normal"), lookup(i, "V2_upright_thatched")]
                          for i in identities])
    d_inv_pert = np.array([D[lookup(i, "V3_inverted_normal"), lookup(i, "V4_inverted_thatched")]
                           for i in identities])

    # Cross-identity baseline distances per orientation (the "noise floor")
    up_norm_idx = [lookup(i, "V1_upright_normal") for i in identities]
    inv_norm_idx = [lookup(i, "V3_inverted_normal") for i in identities]
    d_up_between = cross_id_distances(D, up_norm_idx)
    d_inv_between = cross_id_distances(D, inv_norm_idx)

    # Primary: ratio of mean perturbation distances
    isi_pert = float(d_up_pert.mean() / d_inv_pert.mean()) if d_inv_pert.mean() > 1e-12 else float("nan")

    # Secondary: z-scored perturbation ratio
    z_up = d_up_pert / max(d_up_between.std(ddof=1), 1e-12)
    z_inv = d_inv_pert / max(d_inv_between.std(ddof=1), 1e-12)
    isi_zscore = float(z_up.mean() / z_inv.mean()) if z_inv.mean() > 1e-12 else float("nan")

    # Bootstrap CI: resample identities with replacement
    n_ids = len(identities)
    isi_pert_boot = np.empty(n_bootstrap)
    isi_zscore_boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_ids, n_ids)
        wu = d_up_pert[idx]
        wi = d_inv_pert[idx]
        bu = rng.choice(d_up_between, size=len(d_up_between), replace=True)
        bi = rng.choice(d_inv_between, size=len(d_inv_between), replace=True)
        isi_pert_boot[b] = wu.mean() / wi.mean() if wi.mean() > 1e-12 else np.nan
        zu = wu / max(bu.std(ddof=1), 1e-12)
        zi = wi / max(bi.std(ddof=1), 1e-12)
        isi_zscore_boot[b] = zu.mean() / zi.mean() if zi.mean() > 1e-12 else np.nan

    def pct(arr, q):
        v = arr[~np.isnan(arr)]
        return float(np.percentile(v, q)) if len(v) else float("nan")

    return {
        "isi_pert": isi_pert,
        "isi_pert_ci_lo": pct(isi_pert_boot, 2.5),
        "isi_pert_ci_hi": pct(isi_pert_boot, 97.5),
        "isi_zscore": isi_zscore,
        "isi_zscore_ci_lo": pct(isi_zscore_boot, 2.5),
        "isi_zscore_ci_hi": pct(isi_zscore_boot, 97.5),
        "mean_d_pert_upright": float(d_up_pert.mean()),
        "mean_d_pert_inverted": float(d_inv_pert.mean()),
        "std_d_pert_upright": float(d_up_pert.std(ddof=1)),
        "std_d_pert_inverted": float(d_inv_pert.std(ddof=1)),
        "mean_d_between_upright": float(d_up_between.mean()),
        "mean_d_between_inverted": float(d_inv_between.mean()),
        "n_identities": int(n_ids),
    }


def main(args):
    manifest = pd.read_csv(args.manifest)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    npz_files = sorted(Path(args.embedding_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"no NPZ files in {args.embedding_dir}")

    for npz_path in npz_files:
        model_id = npz_path.stem
        npz = np.load(npz_path, allow_pickle=True)
        stim_ids = np.array([str(s) for s in npz["stim_ids"]])
        embeddings = npz["embeddings"]
        # Reorder manifest to match the embedding order
        m_indexed = manifest.set_index("stim_id").reindex(stim_ids).reset_index()

        metrics = compute_isi(embeddings, m_indexed,
                              n_bootstrap=args.n_bootstrap, seed=args.seed)
        metrics["model_id"] = model_id
        metrics["embedding_dim"] = int(embeddings.shape[1])
        results.append(metrics)
        print(f"{model_id:24s} ISI_pert={metrics['isi_pert']:6.3f} "
              f"[{metrics['isi_pert_ci_lo']:.3f}, {metrics['isi_pert_ci_hi']:.3f}]  "
              f"d_up={metrics['mean_d_pert_upright']:.4f}  d_inv={metrics['mean_d_pert_inverted']:.4f}")

    df = pd.DataFrame(results)
    # Put model_id first
    cols = ["model_id"] + [c for c in df.columns if c != "model_id"]
    df = df[cols]
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_dir", required=True, help="dir with one NPZ per model")
    p.add_argument("--manifest", required=True, help="thatcher_manifest.csv")
    p.add_argument("--output", required=True, help="output CSV path")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260521)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
