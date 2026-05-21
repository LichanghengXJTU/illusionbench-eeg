"""analysis/verification_2afc.py — 2AFC behavioral sanity check for ISI_pert.

For each model + paradigm:
1. Calibrate Siamese cosine-distance threshold τ on (V1_i, V1_j) i≠j between-
   identity pairs at a fixed percentile (default 50% = median).
2. Treat "are V_a, V_b the same person?" as a thresholded decision:
   model says "different" iff cosine_distance(V_a, V_b) > τ.
3. Compute "different rate" (= fraction of pairs classified as 'different')
   separately for each orientation/condition pair:
   - Thatcher: upright (V1, V2) vs inverted (V3, V4)
   - Composite: aligned (V1, V2) vs misaligned (V3, V4)
   - Part-Whole: whole (V1, V2) vs part (V3, V4)
4. ISI_2AFC = different_rate(orientation_A) / different_rate(orientation_B)
5. Correlate ISI_2AFC with ISI_pert across all models.

If ISI_2AFC and ISI_pert rank-correlate strongly across models, ISI_pert is a
valid behavioral analog. If they diverge, the gap is informative.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise cosine distance between rows of a and b. Assumes L2-normalized."""
    return 1.0 - (a * b).sum(axis=-1).clip(-1.0, 1.0)


def cross_id_distances(embs: np.ndarray) -> np.ndarray:
    """Upper-triangular pairwise cosine distances over identity set."""
    sim = embs @ embs.T
    sim = np.clip(sim, -1.0, 1.0)
    n = embs.shape[0]
    iu = np.triu_indices(n, k=1)
    return 1.0 - sim[iu]


def auc_a_vs_b(a: np.ndarray, b: np.ndarray) -> float:
    """ROC-AUC for binary detection: probability that a random sample from a
    exceeds a random sample from b. = P(a > b). Equivalent to Mann-Whitney."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a, b = np.asarray(a), np.asarray(b)
    # Vectorized comparison; ties count as 0.5
    n = 0; tot = 0
    for x in a:
        n += len(b)
        tot += (x > b).sum() + 0.5 * (x == b).sum()
    return float(tot) / n if n else float("nan")


def compute_2afc(embeddings: np.ndarray, manifest: pd.DataFrame,
                  pair_a_conds=("V1_upright_normal", "V2_upright_thatched"),
                  pair_b_conds=("V3_inverted_normal", "V4_inverted_thatched"),
                  thresh_pct: float = 50.0) -> dict:
    """Two-pronged verification analog:
      (A) Detection AUC = P(d_pair_a > d_pair_b) across identities. >0.5 ⇒ model
          treats orientation-A modification as more "different-person-like" than
          orientation-B modification.
      (B) Threshold-based diff-rate analog with τ at thresh_pct of between-
          identity distance. We also report at low (5%, 10%) thresholds so the
          metric isn't dominated by τ being above all within-id distances.
    """
    stim_to_idx = {sid: i for i, sid in enumerate(manifest["stim_id"].to_list())}
    identities = sorted(manifest["identity_id"].unique().tolist())

    def lookup(ident_id: int, cond: str) -> int:
        sid = f"thatcher_id{ident_id:04d}_{cond}"
        return stim_to_idx[sid]

    d_a = np.array([cosine_distance(
        embeddings[lookup(i, pair_a_conds[0])][None, :],
        embeddings[lookup(i, pair_a_conds[1])][None, :])[0]
        for i in identities])
    d_b = np.array([cosine_distance(
        embeddings[lookup(i, pair_b_conds[0])][None, :],
        embeddings[lookup(i, pair_b_conds[1])][None, :])[0]
        for i in identities])

    # Between-identity reference distribution: V1 pairs across identities
    v1_embs = np.stack([embeddings[lookup(i, pair_a_conds[0])] for i in identities])
    d_ref = cross_id_distances(v1_embs)
    if len(d_ref) == 0:
        return {"isi_2afc": float("nan"), "auc_a_vs_b": float("nan")}

    # (A) ROC-AUC for orientation discrimination (modification detectability)
    auc = auc_a_vs_b(d_a, d_b)

    # (B) Threshold-based diff-rate, sweeping multiple percentiles
    out = {
        "n_ref_pairs": int(len(d_ref)),
        "n_identities": len(identities),
        "mean_d_a": float(d_a.mean()),
        "mean_d_b": float(d_b.mean()),
        "mean_d_ref": float(d_ref.mean()),
        "auc_a_vs_b": auc,
    }
    for pct in (5, 10, 25, 50):
        tau = float(np.percentile(d_ref, pct))
        da_diff = float((d_a > tau).mean())
        db_diff = float((d_b > tau).mean())
        out[f"tau_{pct}"] = tau
        out[f"diff_rate_a_thr{pct}"] = da_diff
        out[f"diff_rate_b_thr{pct}"] = db_diff
        out[f"isi_2afc_thr{pct}"] = (da_diff / db_diff) if db_diff > 1e-9 else float("nan")

    return out


def main(args):
    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    isi_pert_df = pd.read_csv(args.isi_csv).set_index("model_id")

    rows = []
    npz_files = sorted(Path(args.embedding_dir).glob("*.npz"))
    for npz_path in npz_files:
        model_id = npz_path.stem
        npz = np.load(npz_path, allow_pickle=True)
        stim_ids = np.array([str(s) for s in npz["stim_ids"]])
        embs = npz["embeddings"]
        m_indexed = manifest.set_index("stim_id").reindex(stim_ids).reset_index()

        r = compute_2afc(embs, m_indexed,
                         pair_a_conds=tuple(args.pair_a.split(",")),
                         pair_b_conds=tuple(args.pair_b.split(",")),
                         thresh_pct=args.thresh_pct)
        r["model_id"] = model_id
        if model_id in isi_pert_df.index:
            r["isi_pert"] = float(isi_pert_df.loc[model_id, "isi_pert"])
        rows.append(r)
        print(f"{model_id:30s} ISI_pert={r.get('isi_pert', float('nan')):6.3f}  "
              f"AUC(A>B)={r['auc_a_vs_b']:.3f}  "
              f"diff_rate@thr5: A={r['diff_rate_a_thr5']:.3f} B={r['diff_rate_b_thr5']:.3f}")

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"verification_2afc_{args.paradigm_tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Rank correlations with ISI_pert across models
    valid = df.dropna(subset=["isi_pert"])
    if len(valid) >= 4:
        print(f"\nAcross {len(valid)} models:")
        summary = {"paradigm_tag": args.paradigm_tag, "n_models": int(len(valid))}
        for col in ["auc_a_vs_b", "isi_2afc_thr5", "isi_2afc_thr10", "isi_2afc_thr25"]:
            sub = valid.dropna(subset=[col])
            if len(sub) < 4:
                continue
            sp, sp_p = spearmanr(sub["isi_pert"], sub[col])
            pr, pr_p = pearsonr(sub["isi_pert"], sub[col])
            print(f"  vs {col:20s}: Spearman ρ={sp:+.3f} p={sp_p:.4f} | Pearson r={pr:+.3f} p={pr_p:.4f} (n={len(sub)})")
            summary[f"spearman_rho_{col}"] = float(sp)
            summary[f"spearman_p_{col}"] = float(sp_p)
            summary[f"pearson_r_{col}"] = float(pr)
            summary[f"pearson_p_{col}"] = float(pr_p)
            summary[f"n_valid_{col}"] = int(len(sub))
        with open(out_dir / f"verification_2afc_{args.paradigm_tag}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--isi_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--paradigm_tag", required=True, help="e.g., 'thatcher' or 'composite'")
    p.add_argument("--pair_a", default="V1_upright_normal,V2_upright_thatched")
    p.add_argument("--pair_b", default="V3_inverted_normal,V4_inverted_thatched")
    p.add_argument("--thresh_pct", type=float, default=50.0,
                   help="percentile of between-identity distance for tau")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
