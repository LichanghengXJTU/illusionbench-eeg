"""figures/compare_isi_bars.py — side-by-side ISI comparison across two stimulus sets.

Given two ISI CSVs (e.g., LFW vs FFHQ), draws bars side-by-side per model so the
effect of stimulus quality on ISI magnitude is immediately visible.

Also marks the human reference band (Carbon 2005, ISI ≈ 4-5) and image-stat
baseline (ISI = 1).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PRIOR_CLASS = {
    "P02_clip_b32": "CLIP", "P03_clip_l14": "CLIP", "P04_clip_h14": "CLIP",
    "P05_clip_g14": "CLIP", "P06_clip_bigG14": "CLIP",
    "P07_dinov2_base": "DINOv2 (SSL)", "P08_dinov2_large": "DINOv2 (SSL)",
    "P09_dinov2_giant": "DINOv2 (SSL)",
    "P10_mae_huge": "MAE (SSL)", "P11_sdxl_vae": "VAE (pixel-stat)",
    "P17_facenet_vggface2": "Face-trained", "P18_facenet_casiawebface": "Face-trained",
    "N02_untrained_vit": "untrained / control", "N03_pixel": "untrained / control",
}
CLASS_ORDER = ["untrained / control", "VAE (pixel-stat)", "Face-trained", "MAE (SSL)", "DINOv2 (SSL)", "CLIP"]


def main(args):
    df1 = pd.read_csv(args.csv1).assign(stimset=args.label1)
    df2 = pd.read_csv(args.csv2).assign(stimset=args.label2)
    df = pd.concat([df1, df2], ignore_index=True)
    df["class"] = df["model_id"].map(PRIOR_CLASS).fillna("other")
    df["class_order"] = df["class"].map({c: i for i, c in enumerate(CLASS_ORDER)})

    models = (df.groupby("model_id")["isi_pert"].mean()
                .reset_index()
                .merge(df[["model_id", "class_order"]].drop_duplicates(), on="model_id")
                .sort_values(["class_order", "isi_pert"], kind="stable")["model_id"].tolist())

    n = len(models)
    x = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * n + 2), 5.5))

    d1 = df1.set_index("model_id").reindex(models)
    d2 = df2.set_index("model_id").reindex(models)

    bars1 = ax.bar(x - width / 2, d1["isi_pert"], width, label=args.label1,
                   color="#9ecae1", edgecolor="black", linewidth=0.6, zorder=2)
    bars2 = ax.bar(x + width / 2, d2["isi_pert"], width, label=args.label2,
                   color="#fdae6b", edgecolor="black", linewidth=0.6, zorder=2)

    ax.errorbar(x - width / 2, d1["isi_pert"],
                yerr=[d1["isi_pert"] - d1["isi_pert_ci_lo"], d1["isi_pert_ci_hi"] - d1["isi_pert"]],
                fmt="none", ecolor="black", capsize=3, lw=1.0, zorder=3)
    ax.errorbar(x + width / 2, d2["isi_pert"],
                yerr=[d2["isi_pert"] - d2["isi_pert_ci_lo"], d2["isi_pert_ci_hi"] - d2["isi_pert"]],
                fmt="none", ecolor="black", capsize=3, lw=1.0, zorder=3)

    ax.axhline(1.0, color="black", lw=0.7, ls="--", zorder=1)
    ax.axhspan(4.0, 5.0, color="#1f77b4", alpha=0.10, zorder=0)
    ax.axhline(4.0, color="#1f77b4", lw=0.5, ls=":", zorder=1)
    ax.axhline(5.0, color="#1f77b4", lw=0.5, ls=":", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("ISI_pert = mean d(upright N,T) / mean d(inverted N,T)")
    ax.set_title(f"Thatcher ISI: {args.label1} vs {args.label2}  (200 identities each, 95% bootstrap CI)")
    ymax = max(d1["isi_pert_ci_hi"].max(), d2["isi_pert_ci_hi"].max(), 5.2)
    ax.set_ylim(0, ymax + 0.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    handles = [
        Patch(facecolor="#9ecae1", edgecolor="black", label=args.label1),
        Patch(facecolor="#fdae6b", edgecolor="black", label=args.label2),
        plt.Line2D([0], [0], color="black", ls="--", label="image-stat baseline (ISI=1)"),
        Patch(facecolor="#1f77b4", alpha=0.10, label="human range (Carbon 2005, 4-5)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    out_png = Path(args.output_prefix + ".png")
    out_pdf = Path(args.output_prefix + ".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}\nsaved: {out_pdf}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv1", required=True)
    p.add_argument("--csv2", required=True)
    p.add_argument("--label1", default="LFW (224)")
    p.add_argument("--label2", default="FFHQ (1024)")
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
