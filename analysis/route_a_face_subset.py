"""analysis/route_a_face_subset.py — restrict Route A preservation to face-like concepts.

We do not have a labeled concept list for THINGS-EEG2 test split, but we DO have
CLIP-ViT-H/14-LAION text features for each of the 200 test concepts in
`ViT-H-14_features_test.pt` text_features. We use semantic similarity to a
small set of face-related text queries to identify the top-K face-related test
concepts in an unsupervised way, then re-run Route A preservation only on those
concepts.

Sanity: the top-K face concepts should have substantially HIGH semantic-similarity
to "face/person/human" queries. We can inspect the top-K text features by
projecting back via CLIP text encoder (using the same `laion/CLIP-ViT-H-14-laion2B`
on our server).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


FACE_QUERIES = [
    "a photo of a face",
    "a photo of a person",
    "a photo of a human",
    "a photo of a man",
    "a photo of a woman",
    "a photo of a girl",
    "a photo of a boy",
    "a photo of a baby",
]


def get_clip_text_query_embeddings(device="cuda", dtype=torch.float16):
    """Returns (Q, 1024) tensor of normalized text features for FACE_QUERIES."""
    from transformers import CLIPModel, CLIPProcessor
    hf_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model = CLIPModel.from_pretrained(hf_id, torch_dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(hf_id)
    inputs = processor(text=FACE_QUERIES, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = F.normalize(feats, dim=-1)
    out = feats.cpu().float()
    del model
    torch.cuda.empty_cache()
    return out  # (Q, 1024)


def identify_face_concepts(text_features_200: torch.Tensor,
                           face_query_embs: torch.Tensor,
                           top_k: int = 30):
    """Score each of 200 concepts by max-over-queries cosine similarity to a face
    query. Returns indices of the top_k highest-scoring concepts + their scores."""
    # normalize text_features_200 if not already (they are pre-normalized in CLIP)
    tf = F.normalize(text_features_200.float(), dim=-1)
    fq = F.normalize(face_query_embs.float(), dim=-1)
    sim = tf @ fq.T  # (200, Q)
    score = sim.max(dim=1).values  # (200,)
    top_idx = torch.argsort(score, descending=True)[:top_k].cpu().numpy()
    return top_idx, score.cpu().numpy()


def compute_preservation_subset(clip_img: torch.Tensor, eeg_preds: torch.Tensor,
                                 concept_idx: np.ndarray) -> np.ndarray:
    """Same as compute_preservation but restricted to concept_idx."""
    n_sub, _, n_dim = eeg_preds.shape
    preservation = np.zeros((n_sub, n_dim), dtype=np.float64)
    clip_sub = clip_img[concept_idx].numpy()
    for s in range(n_sub):
        eeg_sub = eeg_preds[s, concept_idx].numpy()
        for d in range(n_dim):
            x = clip_sub[:, d]
            y = eeg_sub[:, d]
            if x.std() < 1e-12 or y.std() < 1e-12:
                preservation[s, d] = np.nan
            else:
                preservation[s, d] = np.corrcoef(x, y)[0, 1]
    return preservation.mean(axis=0)


def main(args):
    atm_root = Path(args.atm_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ATM data
    clip_test = torch.load(atm_root / "ViT-H-14_features_test.pt",
                           weights_only=False, map_location="cpu")
    clip_img = clip_test["img_features"].float()
    text_features = clip_test["text_features"].float()  # (200, 1024)

    eeg_preds = []
    for s in range(1, 11):
        p = atm_root / "emb_eeg" / f"ATM_S_eeg_features_sub-{s:02d}_test.pt"
        eeg_preds.append(torch.load(p, weights_only=False, map_location="cpu").float())
    eeg_preds = torch.stack(eeg_preds, dim=0)

    # Get face query embeddings via CLIP-ViT-H/14 text encoder
    print("[clip-text] computing face-query embeddings via CLIP-ViT-H/14 LAION")
    face_q = get_clip_text_query_embeddings(device="cuda")
    print(f"  face_query_embs shape: {tuple(face_q.shape)}")

    # Identify top-K face-related test concepts
    top_idx, scores = identify_face_concepts(text_features, face_q, top_k=args.top_k)
    print(f"[face-concepts] top-{args.top_k} face-related test concepts:")
    print(f"  indices: {top_idx[:20].tolist()}{'...' if len(top_idx)>20 else ''}")
    print(f"  max sim scores: {scores[top_idx][:10].round(3).tolist()}")
    print(f"  min sim score in top-K: {scores[top_idx[-1]]:.3f}")
    print(f"  median score across ALL 200: {np.median(scores):.3f}")

    # Compute preservation for face subset + for matched-size random subset
    rng = np.random.default_rng(args.seed)
    nonface_pool = np.setdiff1d(np.arange(200), top_idx)
    random_idx = rng.choice(nonface_pool, size=args.top_k, replace=False)

    pres_face   = compute_preservation_subset(clip_img, eeg_preds, top_idx)
    pres_random = compute_preservation_subset(clip_img, eeg_preds, random_idx)
    pres_all    = compute_preservation_subset(clip_img, eeg_preds, np.arange(200))

    print()
    print(f"=== preservation comparison (K={args.top_k}) ===")
    print(f"  face subset:   mean r = {np.nanmean(pres_face):.3f}  median = {np.nanmedian(pres_face):.3f}")
    print(f"  random subset: mean r = {np.nanmean(pres_random):.3f}  median = {np.nanmedian(pres_random):.3f}")
    print(f"  ALL 200:       mean r = {np.nanmean(pres_all):.3f}  median = {np.nanmedian(pres_all):.3f}")

    # Permutation: is face-subset preservation different from random-subset?
    diff_obs = np.nanmean(pres_face) - np.nanmean(pres_random)
    n_perm = args.n_perm
    null_diffs = np.empty(n_perm)
    all_dims = np.arange(len(pres_face))
    for i in range(n_perm):
        perm = rng.permutation(all_dims)
        # randomize per-dim assignment between face and random
        mask = rng.random(len(perm)) > 0.5
        a = np.where(mask, pres_face, pres_random)
        b = np.where(mask, pres_random, pres_face)
        null_diffs[i] = np.nanmean(a) - np.nanmean(b)
    p_two = (np.abs(null_diffs) >= abs(diff_obs)).mean()

    print(f"\n  face - random difference: {diff_obs:+.4f}, perm p_two-sided = {p_two:.3f}")

    # Save artifacts
    np.savez(out_dir / "route_a_face_subset_arrays.npz",
             pres_face=pres_face, pres_random=pres_random, pres_all=pres_all,
             face_idx=top_idx, random_idx=random_idx,
             face_query_scores=scores)
    summary = {
        "top_k": int(args.top_k),
        "n_perm": int(n_perm),
        "pres_face_mean": float(np.nanmean(pres_face)),
        "pres_face_median": float(np.nanmedian(pres_face)),
        "pres_random_mean": float(np.nanmean(pres_random)),
        "pres_random_median": float(np.nanmedian(pres_random)),
        "pres_all_mean": float(np.nanmean(pres_all)),
        "face_minus_random": float(diff_obs),
        "perm_p_two_sided": float(p_two),
        "top_idx": top_idx.tolist(),
        "max_face_query_scores_sample": [float(scores[i]) for i in top_idx[:10]],
    }
    with open(out_dir / "route_a_face_subset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atm_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_k", type=int, default=30,
                   help="how many face-most-similar test concepts to take")
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--n_perm", type=int, default=10000)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
