"""figures/headline_isi_bars.py — bar plot of ISI_pert per model with 95% CIs.

Inputs:  outputs/tables/thatcher_isi_v1.csv
Outputs: outputs/figures/headline_isi_bars.png and .pdf
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorize each model into a prior class for color coding.
PRIOR_CLASS = {
    # CLIP family (image-text contrastive on web data)
    "P02_clip_b32": "CLIP",
    "P03_clip_l14": "CLIP",
    "P04_clip_h14": "CLIP",
    "P05_clip_g14": "CLIP",
    "P06_clip_bigG14": "CLIP",
    # DINOv2 (self-supervised)
    "P07_dinov2_base": "DINOv2 (SSL)",
    "P08_dinov2_large": "DINOv2 (SSL)",
    "P09_dinov2_giant": "DINOv2 (SSL)",
    # MAE (pixel-prediction self-supervised)
    "P10_mae_huge": "MAE (SSL)",
    # SDXL VAE (pixel-statistics)
    "P11_sdxl_vae": "VAE (pixel-stat)",
    # Negative controls
    "N02_untrained_vit": "untrained / control",
    "N03_pixel": "untrained / control",
}

CLASS_COLOR = {
    "CLIP": "#d62728",
    "DINOv2 (SSL)": "#ff7f0e",
    "MAE (SSL)": "#bcbd22",
    "VAE (pixel-stat)": "#7f7f7f",
    "untrained / control": "#cccccc",
}

# Sort within-class by ISI ascending; classes ordered low-to-high baseline expectation
CLASS_ORDER = ["untrained / control", "VAE (pixel-stat)", "MAE (SSL)", "DINOv2 (SSL)", "CLIP"]


def main(args):
    df = pd.read_csv(args.input_csv)
    df["class"] = df["model_id"].map(PRIOR_CLASS).fillna("other")
    df["class_order"] = df["class"].map({c: i for i, c in enumerate(CLASS_ORDER)})
    df = df.sort_values(["class_order", "isi_pert"], kind="stable").reset_index(drop=True)

    n = len(df)
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * n + 2), 5.0))

    xs = np.arange(n)
    isi = df["isi_pert"].to_numpy()
    lo = df["isi_pert_ci_lo"].to_numpy()
    hi = df["isi_pert_ci_hi"].to_numpy()
    err_lo = isi - lo
    err_hi = hi - isi

    colors = [CLASS_COLOR.get(c, "#9467bd") for c in df["class"]]

    bars = ax.bar(xs, isi, color=colors, edgecolor="black", linewidth=0.7, zorder=2)
    ax.errorbar(xs, isi, yerr=[err_lo, err_hi],
                fmt="none", ecolor="black", capsize=3, lw=1.2, zorder=3)

    # Reference lines
    ax.axhline(1.0, color="black", lw=0.8, ls="--", zorder=1, label="image-stat baseline (ISI=1)")
    ax.axhspan(4.0, 5.0, color="#1f77b4", alpha=0.10, zorder=0,
               label="human reference (Carbon 2005, ISI ≈ 4–5)")
    ax.axhline(4.0, color="#1f77b4", lw=0.6, ls=":", zorder=1)
    ax.axhline(5.0, color="#1f77b4", lw=0.6, ls=":", zorder=1)

    ax.set_xticks(xs)
    ax.set_xticklabels(df["model_id"].to_list(), rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Illusion Sensitivity Index (ISI_pert)\n= mean d(upright N,T) / mean d(inverted N,T)")
    ax.set_title("Thatcher ISI per visual prior  (200 LFW identities, 95% bootstrap CI)", fontsize=11)
    ax.set_ylim(0, max(isi.max() + 1.0, 5.2))
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Legend with class colors + reference lines
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor="black", label=c) for c in CLASS_ORDER if c in df["class"].unique()]
    handles.append(plt.Line2D([0], [0], color="black", ls="--", label="image-stat baseline"))
    handles.append(Patch(facecolor="#1f77b4", alpha=0.10, label="human range (4-5)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.95)

    plt.tight_layout()
    out_png = Path(args.output_prefix + ".png")
    out_pdf = Path(args.output_prefix + ".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}\nsaved: {out_pdf}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_prefix", required=True, help="output path WITHOUT extension")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
