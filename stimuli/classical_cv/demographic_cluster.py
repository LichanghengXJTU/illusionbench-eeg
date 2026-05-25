"""stimuli/classical_cv/demographic_cluster.py — k-means demographic clustering.

Pure classical CV descriptors (no DL): cluster filtered FFHQ identities so
composite/partwhole paradigms can be paired within-cell (avoiding cross-
demographic skin/lighting mismatch that previously dominated PWI signal).

Descriptors per identity:
  - HSV mean of face oval (skin tone)
  - face bbox aspect ratio (rough age/sex proxy)
  - IOD / face width ratio (rough sex proxy)
  - eye-line-y / face-bbox-height (face proportions)
  - cheek local std (skin smoothness, age proxy)

These are deliberately simple — k-means with k=4-8 clusters partitions the
filtered set into "look similar enough to pair" groups. Random pairing
WITHIN a cluster avoids the worst cross-demographic mismatches without
needing demographic labels.
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans


def face_descriptor(rgb: np.ndarray, landmarks: np.ndarray,
                     bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Return a feature vector for clustering."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    # Skin-tone region: face oval, but exclude the eye/mouth strips.
    # Approximation: use the cheek region (lower half of face bbox,
    # central 60% width).
    cheek_y1 = int(y1 + bh * 0.55); cheek_y2 = int(y1 + bh * 0.85)
    cheek_x1 = int(x1 + bw * 0.20); cheek_x2 = int(x1 + bw * 0.80)
    cheek_y1, cheek_y2 = max(0, cheek_y1), min(h, cheek_y2)
    cheek_x1, cheek_x2 = max(0, cheek_x1), min(w, cheek_x2)
    cheek = rgb[cheek_y1:cheek_y2, cheek_x1:cheek_x2]
    if cheek.size == 0:
        cheek_hsv_mean = np.zeros(3)
        cheek_std = 0.0
    else:
        cheek_hsv = cv2.cvtColor(cheek, cv2.COLOR_RGB2HSV)
        cheek_hsv_mean = cheek_hsv.reshape(-1, 3).mean(axis=0)
        cheek_std = float(cheek.std())

    le = landmarks[36:42].mean(axis=0)
    re = landmarks[42:48].mean(axis=0)
    iod = float(np.linalg.norm(le - re))
    iod_fw = iod / max(bw, 1e-6)
    eye_y_frac = float((le[1] + re[1]) / 2 - y1) / max(bh, 1e-6)

    aspect = bw / max(bh, 1e-6)

    return np.array([
        cheek_hsv_mean[0] / 180.0,  # H normalized
        cheek_hsv_mean[1] / 255.0,  # S
        cheek_hsv_mean[2] / 255.0,  # V (lightness)
        iod_fw,                      # ~ sex proxy
        eye_y_frac,                  # face proportions
        aspect,                      # bbox shape
        cheek_std / 64.0,            # skin smoothness ~ age proxy
    ], dtype=np.float32)


def cluster_identities(records: list[dict], k: int = 6,
                        seed: int = 20260525) -> dict[int, int]:
    """Run k-means; return {identity_id → cluster_id}."""
    feats = np.stack([r["descriptor"] for r in records])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(feats)
    return {r["identity_id"]: int(c) for r, c in zip(records, labels)}


def within_cluster_pairs(cluster_map: dict[int, int],
                          seed: int = 20260525
                          ) -> dict[int, int]:
    """For each identity, pick a within-cluster partner (random, non-self,
    deterministic). Returns {i: j}."""
    rng = np.random.default_rng(seed)
    by_cluster: dict[int, list[int]] = {}
    for ident, c in cluster_map.items():
        by_cluster.setdefault(c, []).append(ident)
    pairs: dict[int, int] = {}
    for c, members in by_cluster.items():
        if len(members) == 1:
            # singleton cluster: pair with self-overflow (no pair possible)
            pairs[members[0]] = members[0]   # caller can skip self-pairs
            continue
        # Random non-self pairing inside cluster: shift by 1
        shuffled = list(members)
        rng.shuffle(shuffled)
        for i in range(len(shuffled)):
            pairs[shuffled[i]] = shuffled[(i + 1) % len(shuffled)]
    return pairs
