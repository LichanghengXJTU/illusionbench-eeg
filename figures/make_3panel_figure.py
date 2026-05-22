"""Make a 3-panel combined RSA heatmap (Thatcher, Composite, Part-Whole) for paper Figure 3.
Uses E033 + E034 RSA correlation CSVs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import os

paradigms = [
    ("Thatcher", "outputs/tables/e033_rsa_correlation.csv"),
    ("Composite", "outputs/tables/e034_rsa_composite.csv"),
    ("Part-Whole", "outputs/tables/e034_rsa_partwhole.csv"),
]


def cluster_order(R):
    D = 1.0 - R
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    return dendrogram(Z, no_plot=True)["leaves"]


def clean(lbl):
    parts = lbl.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("P"):
        return parts[1].replace("_", " ")
    return lbl.replace("_", " ")


fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Use Thatcher clustering as canonical ordering for consistency
df_t = pd.read_csv(paradigms[0][1], index_col=0)
R_t = df_t.values
names_t = df_t.index.tolist()
order_t = cluster_order(R_t)
labels_t = [clean(names_t[i]) for i in order_t]

for ax, (label, csv_path) in zip(axes, paradigms):
    df = pd.read_csv(csv_path, index_col=0)
    R = df.values
    names = df.index.tolist()
    # Match Thatcher's ordering to enable cross-paradigm comparison
    common = sorted(set(names) & set(names_t))
    idx_map = {n: i for i, n in enumerate(names)}
    idx_in_p = [idx_map[names_t[i]] for i in order_t if names_t[i] in common]
    R_ord = R[np.ix_(idx_in_p, idx_in_p)]
    labels = [clean(names_t[i]) for i in order_t if names_t[i] in common]
    im = ax.imshow(R_ord, cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title(f"{label}", fontsize=14)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

# Single colorbar
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Spearman r (RSA)")

fig.suptitle("Figure 3 — RSA correlation across 25 visual priors, per paradigm\n"
             "(prior order = Thatcher hierarchical clustering)",
             fontsize=14, y=1.02)

out_path = "figures/exports/figure3_rsa_3panel.png"
os.makedirs("figures/exports", exist_ok=True)
fig.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
