"""figures/three_paradigm_panel.py — three-paradigm summary panel (Thatcher / Composite / Part-Whole).

For each prior in REGISTRY, show ISI/CSI/PWI side-by-side as grouped bars per
prior, color-coded by prior class.
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
    "P19_siglip_base_384": "SigLIP/MetaCLIP",
    "P20_siglip_so400m": "SigLIP/MetaCLIP",
    "P21_metaclip_h14": "SigLIP/MetaCLIP",
    "P07_dinov2_base": "DINOv2 (SSL)", "P08_dinov2_large": "DINOv2 (SSL)",
    "P09_dinov2_giant": "DINOv2 (SSL)",
    "P10_mae_huge": "MAE (SSL)", "P11_sdxl_vae": "VAE (pixel-stat)",
    "P17_facenet_vggface2": "Face-trained", "P18_facenet_casiawebface": "Face-trained",
    "N02_untrained_vit": "untrained / control", "N03_pixel": "untrained / control",
}
CLASS_ORDER = ["untrained / control", "VAE (pixel-stat)", "Face-trained",
               "MAE (SSL)", "DINOv2 (SSL)", "SigLIP/MetaCLIP", "CLIP"]
CLASS_COLOR = {
    "CLIP": "#d62728", "SigLIP/MetaCLIP": "#8c564b", "DINOv2 (SSL)": "#ff7f0e",
    "MAE (SSL)": "#bcbd22", "VAE (pixel-stat)": "#7f7f7f",
    "Face-trained": "#2ca02c", "untrained / control": "#cccccc",
}


def main(args):
    th = pd.read_csv(args.thatcher_csv).set_index("model_id")["isi_pert"]
    cp = pd.read_csv(args.composite_csv).set_index("model_id")["isi_pert"]
    pw = pd.read_csv(args.partwhole_csv).set_index("model_id")["isi_pert"]

    th_ci_lo = pd.read_csv(args.thatcher_csv).set_index("model_id")["isi_pert_ci_lo"]
    th_ci_hi = pd.read_csv(args.thatcher_csv).set_index("model_id")["isi_pert_ci_hi"]
    cp_ci_lo = pd.read_csv(args.composite_csv).set_index("model_id")["isi_pert_ci_lo"]
    cp_ci_hi = pd.read_csv(args.composite_csv).set_index("model_id")["isi_pert_ci_hi"]
    pw_ci_lo = pd.read_csv(args.partwhole_csv).set_index("model_id")["isi_pert_ci_lo"]
    pw_ci_hi = pd.read_csv(args.partwhole_csv).set_index("model_id")["isi_pert_ci_hi"]

    # Sort by class order, within class by Thatcher ISI
    df = pd.DataFrame({"th": th, "cp": cp, "pw": pw}).reset_index()
    df["class"] = df["model_id"].map(PRIOR_CLASS).fillna("other")
    df["class_order"] = df["class"].map({c: i for i, c in enumerate(CLASS_ORDER)})
    df = df.sort_values(["class_order", "th"], kind="stable").reset_index(drop=True)
    models = df["model_id"].tolist()
    n = len(models)
    x = np.arange(n)
    width = 0.27

    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.7 * n + 2), 11),
                             sharex=True)
    for ax, series, ci_lo, ci_hi, title, ref_line, ref_band in [
        (axes[0], th, th_ci_lo, th_ci_hi,
         "Thatcher ISI (face inversion × local feature rotation)",
         1.0, (4, 5)),
        (axes[1], cp, cp_ci_lo, cp_ci_hi,
         "Composite CSI (spatial integration: aligned vs misaligned bottom-swap)",
         1.0, None),
        (axes[2], pw, pw_ci_lo, pw_ci_hi,
         "Part-Whole PWI (whole-face context dampening of eye-swap)",
         1.0, None),
    ]:
        vals = series.reindex(models).to_numpy()
        lo = ci_lo.reindex(models).to_numpy()
        hi = ci_hi.reindex(models).to_numpy()
        colors = [CLASS_COLOR[PRIOR_CLASS.get(m, "untrained / control")] for m in models]
        ax.bar(x, vals, width=0.72, color=colors, edgecolor="black", linewidth=0.6)
        ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="none",
                    ecolor="black", capsize=3, lw=1.0)
        ax.axhline(ref_line, color="black", lw=0.7, ls="--")
        if ref_band:
            ax.axhspan(ref_band[0], ref_band[1], color="#1f77b4", alpha=0.08)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("ISI / CSI / PWI")
        ax.grid(axis="y", alpha=0.3)
        ymax = max(vals.max(), hi.max(), 1.2)
        ax.set_ylim(0, ymax + 0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(models, rotation=40, ha="right", fontsize=8)

    handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor="black", label=c)
               for c in CLASS_ORDER if c in df["class"].unique()]
    handles.append(plt.Line2D([0], [0], color="black", ls="--", label="image-stat baseline = 1"))
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, ncol=2, framealpha=0.95)

    plt.tight_layout()
    out_png = Path(args.output_prefix + ".png")
    out_pdf = Path(args.output_prefix + ".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}\nsaved: {out_pdf}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--thatcher_csv", required=True)
    p.add_argument("--composite_csv", required=True)
    p.add_argument("--partwhole_csv", required=True)
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
