"""analysis/fusion_simulation.py — multi-anchor (CLIP + CORnet) fusion simulation.

Build a fused embedding by concatenating two independent priors, optionally
with scale-rebalancing so neither dominates the cosine distance. Then compute
ISI/CSI/PWI on the fused space and compare with the pure priors.

Theory recap:
- For two L2-normalized embeddings a (D_a-dim) and b (D_b-dim), the concat
  [a, b] has norm √2; cos([a1,b1]/√2, [a2,b2]/√2) = (cos_a + cos_b) / 2.
- Distance d_fused = 1 - cos_fused = (d_a + d_b) / 2.
- Therefore ISI_fused = (mean_a + mean_b)_up / (mean_a + mean_b)_inv, NOT
  simple mean of ISI_a and ISI_b — it depends on the relative scales.
- To balance contributions, we can re-normalize each prior's distances to
  zero mean / unit std before fusing (z-scored fusion) or rescale by mean
  cross-identity distance ("magnitude-rebalanced fusion").
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cosine_distance_matrix(emb: np.ndarray) -> np.ndarray:
    sim = emb @ emb.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def cross_id_distances_from_matrix(D: np.ndarray, idx_list: list[int]) -> np.ndarray:
    sub = D[np.ix_(idx_list, idx_list)]
    n = len(idx_list)
    iu = np.triu_indices(n, k=1)
    return sub[iu]


def compute_isi_from_distance_matrix(D: np.ndarray, manifest: pd.DataFrame,
                                     stim_to_idx: dict) -> tuple[float, dict]:
    """Compute ISI_pert given a precomputed cosine-distance matrix D."""
    identities = sorted(manifest["identity_id"].unique().tolist())
    def lookup(ident_id: int, cond: str) -> int:
        sid = f"thatcher_id{ident_id:04d}_{cond}"
        return stim_to_idx[sid]
    d_up = np.array([D[lookup(i, "V1_upright_normal"), lookup(i, "V2_upright_thatched")]
                     for i in identities])
    d_inv = np.array([D[lookup(i, "V3_inverted_normal"), lookup(i, "V4_inverted_thatched")]
                      for i in identities])
    isi = d_up.mean() / d_inv.mean() if d_inv.mean() > 1e-12 else float("nan")
    return isi, {"mean_d_up": float(d_up.mean()),
                 "mean_d_inv": float(d_inv.mean()),
                 "n_identities": len(identities)}


def main(args):
    manifest = pd.read_csv(args.manifest)
    stim_to_idx = {sid: i for i, sid in enumerate(manifest["stim_id"].to_list())}

    # Load and align both embeddings (preserve stim_id order from manifest)
    def load_npz(p):
        npz = np.load(p, allow_pickle=True)
        sids = np.array([str(s) for s in npz["stim_ids"]])
        e = npz["embeddings"].astype(np.float64)
        order = [np.where(sids == s)[0][0] for s in manifest["stim_id"].to_list()]
        return e[order]

    e_clip = load_npz(args.clip_npz)
    e_cornet = load_npz(args.cornet_npz)
    print(f"loaded CLIP {e_clip.shape}, CORnet {e_cornet.shape}")

    # Pre-normalize each to unit (should already be unit but enforce)
    e_clip = e_clip / np.linalg.norm(e_clip, axis=1, keepdims=True).clip(1e-9)
    e_cornet = e_cornet / np.linalg.norm(e_cornet, axis=1, keepdims=True).clip(1e-9)

    results = []

    # 1. Pure CLIP
    D_clip = cosine_distance_matrix(e_clip)
    isi_clip, info_clip = compute_isi_from_distance_matrix(D_clip, manifest, stim_to_idx)
    print(f"\nPure CLIP:    ISI = {isi_clip:.3f}  d_up={info_clip['mean_d_up']:.4f}  d_inv={info_clip['mean_d_inv']:.4f}")
    results.append({"variant": "pure_clip", "isi": isi_clip, **info_clip})

    # 2. Pure CORnet
    D_cornet = cosine_distance_matrix(e_cornet)
    isi_cornet, info_cornet = compute_isi_from_distance_matrix(D_cornet, manifest, stim_to_idx)
    print(f"Pure CORnet:  ISI = {isi_cornet:.3f}  d_up={info_cornet['mean_d_up']:.4f}  d_inv={info_cornet['mean_d_inv']:.4f}")
    results.append({"variant": "pure_cornet", "isi": isi_cornet, **info_cornet})

    # 3. Equal-weight concat (each component is unit-norm) → fused norm √2 → renormalize
    e_fused = np.concatenate([e_clip, e_cornet], axis=1)
    e_fused = e_fused / np.linalg.norm(e_fused, axis=1, keepdims=True).clip(1e-9)
    D_fused = cosine_distance_matrix(e_fused)
    isi_fused, info_fused = compute_isi_from_distance_matrix(D_fused, manifest, stim_to_idx)
    print(f"50/50 concat: ISI = {isi_fused:.3f}  d_up={info_fused['mean_d_up']:.4f}  d_inv={info_fused['mean_d_inv']:.4f}")
    results.append({"variant": "concat_50_50", "isi": isi_fused, **info_fused})

    # 4. Magnitude-rebalanced concat: rescale each prior such that mean cross-identity
    #    distance is equal. Then cosine on the fused embedding gives each prior equal
    #    weight in the discriminability scale.
    identities = sorted(manifest["identity_id"].unique().tolist())
    v1_idx = [stim_to_idx[f"thatcher_id{i:04d}_V1_upright_normal"] for i in identities]
    d_ref_clip = cross_id_distances_from_matrix(D_clip, v1_idx).mean()
    d_ref_cornet = cross_id_distances_from_matrix(D_cornet, v1_idx).mean()
    print(f"  mean cross-id distances: CLIP={d_ref_clip:.4f}, CORnet={d_ref_cornet:.4f}")
    # Scale CORnet by sqrt(d_ref_clip / d_ref_cornet) so its distance contribution to the
    # fused embedding matches CLIP. We rescale the vector by this factor in linear space.
    scale = np.sqrt(d_ref_clip / d_ref_cornet)
    e_cornet_rescaled = e_cornet * scale
    e_fused_balanced = np.concatenate([e_clip, e_cornet_rescaled], axis=1)
    e_fused_balanced = e_fused_balanced / np.linalg.norm(e_fused_balanced, axis=1, keepdims=True).clip(1e-9)
    D_balanced = cosine_distance_matrix(e_fused_balanced)
    isi_bal, info_bal = compute_isi_from_distance_matrix(D_balanced, manifest, stim_to_idx)
    print(f"Mag-balanced: ISI = {isi_bal:.3f}  d_up={info_bal['mean_d_up']:.4f}  d_inv={info_bal['mean_d_inv']:.4f}")
    results.append({"variant": "concat_mag_balanced", "isi": isi_bal,
                    "scale_cornet": float(scale), **info_bal})

    # 5. 25% CLIP + 75% CORnet (boost CORnet) and vice versa
    for w_cornet in [0.3, 0.5, 0.7, 0.9]:
        e_combo = np.concatenate([(1 - w_cornet) * e_clip,
                                  w_cornet * e_cornet_rescaled], axis=1)
        e_combo = e_combo / np.linalg.norm(e_combo, axis=1, keepdims=True).clip(1e-9)
        D_c = cosine_distance_matrix(e_combo)
        isi_c, info_c = compute_isi_from_distance_matrix(D_c, manifest, stim_to_idx)
        print(f"w_cornet={w_cornet:.2f} (rebalanced): ISI = {isi_c:.3f}")
        results.append({"variant": f"w_cornet_{w_cornet:.2f}_rebalanced",
                        "isi": isi_c, **info_c})

    df = pd.DataFrame(results)
    out_csv = Path(args.output_dir) / f"fusion_{args.paradigm_tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip_npz", required=True)
    p.add_argument("--cornet_npz", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--paradigm_tag", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
