"""figures/face_vs_randombbox_polished.py — publication-quality Figure 2.

Face-feature Thatcher vs random-bbox controls, polished with consistent
aesthetic with `three_paradigm_polished.py`.
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
    "P07_dinov2_base": "DINOv2", "P08_dinov2_large": "DINOv2",
    "P09_dinov2_giant": "DINOv2",
    "P10_mae_huge": "MAE", "P11_sdxl_vae": "VAE",
    "P17_facenet_vggface2": "Face-trained",
    "P18_facenet_casiawebface": "Face-trained",
    "N02_untrained_vit": "Control", "N03_pixel": "Control",
}
CLASS_ORDER = ["Control", "VAE", "Face-trained", "MAE", "DINOv2", "SigLIP/MetaCLIP", "CLIP"]
CLASS_COLOR = {
    "CLIP": "#d62728", "SigLIP/MetaCLIP": "#a55a4b",
    "DINOv2": "#ff7f0e", "MAE": "#bcbd22", "VAE": "#7f7f7f",
    "Face-trained": "#2ca02c", "Control": "#cccccc",
}
PRETTY_NAME = {
    "P02_clip_b32": "CLIP-B/32", "P03_clip_l14": "CLIP-L/14",
    "P04_clip_h14": "CLIP-H/14", "P05_clip_g14": "CLIP-g/14",
    "P06_clip_bigG14": "CLIP-bigG/14",
    "P19_siglip_base_384": "SigLIP-B", "P20_siglip_so400m": "SigLIP-SO400M",
    "P21_metaclip_h14": "MetaCLIP-H/14",
    "P07_dinov2_base": "DINOv2-B", "P08_dinov2_large": "DINOv2-L",
    "P09_dinov2_giant": "DINOv2-G",
    "P10_mae_huge": "MAE-H", "P11_sdxl_vae": "SDXL-VAE",
    "P17_facenet_vggface2": "FaceNet-VGG2", "P18_facenet_casiawebface": "FaceNet-CASIA",
    "N02_untrained_vit": "ViT-untrained", "N03_pixel": "Pixel",
}


def main(args):
    df1 = pd.read_csv(args.face_csv).set_index("model_id")
    df2 = pd.read_csv(args.rb_csv).set_index("model_id")
    models = [m for m in PRIOR_CLASS if m in df1.index and m in df2.index]
    rows = []
    for m in models:
        rows.append({"model_id": m, "class": PRIOR_CLASS[m],
                     "f_isi": df1.loc[m, "isi_pert"],
                     "f_lo": df1.loc[m, "isi_pert_ci_lo"],
                     "f_hi": df1.loc[m, "isi_pert_ci_hi"],
                     "r_isi": df2.loc[m, "isi_pert"],
                     "r_lo": df2.loc[m, "isi_pert_ci_lo"],
                     "r_hi": df2.loc[m, "isi_pert_ci_hi"]})
    d = pd.DataFrame(rows).set_index("model_id")
    d["class_order"] = d["class"].map({c: i for i, c in enumerate(CLASS_ORDER)})
    d = d.sort_values(["class_order", "f_isi"], kind="stable")
    models = d.index.tolist()
    n = len(models)
    x = np.arange(n)
    width = 0.40

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fvals = d["f_isi"].to_numpy()
    rvals = d["r_isi"].to_numpy()
    colors_face = [CLASS_COLOR[c] for c in d["class"]]
    colors_rand = [tuple(list(plt.cm.colors.to_rgb(c)) + [0.45]) for c in colors_face]

    ax.bar(x - width / 2, fvals, width, color=colors_face,
           edgecolor="black", linewidth=0.8, zorder=2, label="Face Thatcher")
    ax.bar(x + width / 2, rvals, width, color=colors_rand,
           edgecolor="black", linewidth=0.5, zorder=2,
           hatch="///", label="Random-bbox (matched size, non-feature locations)")
    ax.errorbar(x - width / 2, fvals,
                yerr=[fvals - d["f_lo"], d["f_hi"] - fvals],
                fmt="none", ecolor="black", capsize=3, lw=1.0, zorder=3)
    ax.errorbar(x + width / 2, rvals,
                yerr=[rvals - d["r_lo"], d["r_hi"] - rvals],
                fmt="none", ecolor="black", capsize=3, lw=1.0, zorder=3)

    ax.axhline(1.0, color="black", lw=1.0, ls="--", zorder=1)
    ax.axhspan(4.0, 5.0, color="#1f77b4", alpha=0.10, zorder=0)
    ax.axhline(4.0, color="#1f77b4", lw=0.6, ls=":")
    ax.axhline(5.0, color="#1f77b4", lw=0.6, ls=":")

    ax.set_title("Face-feature-specificity of the Thatcher ISI signal "
                 "(Claim 3, E003)",
                 fontsize=13, weight="bold", loc="left", pad=8)
    ax.text(0.99, 0.95,
            "Random-location bboxes (matched size, non-feature regions) reduce "
            "CLIP ISI by 59–74%, leaving 1.2–2.0\n"
            "Face-Thatcher ISI gap is robust across bbox scales 0.5×–1.5× (E025)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.95))

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_NAME[m] for m in models],
                       rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("ISI_pert", fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ymax = max(d["f_hi"].max(), d["r_hi"].max(), 5.2)
    ax.set_ylim(0, ymax + 0.3)
    ax.tick_params(axis="y", labelsize=9)

    handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor="black", label=c)
               for c in CLASS_ORDER if c in d["class"].unique()]
    handles.append(Patch(facecolor="lightgray", edgecolor="black",
                          hatch="///", label="random-bbox (E003)"))
    handles.append(plt.Line2D([0], [0], color="black", ls="--",
                              label="pixel baseline = 1"))
    handles.append(Patch(facecolor="#1f77b4", alpha=0.10,
                          label="human Thatcher range (4–5)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2,
              framealpha=0.95)

    plt.tight_layout()
    out_png = Path(args.output_prefix + ".png")
    out_pdf = Path(args.output_prefix + ".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}\nsaved: {out_pdf}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--face_csv", required=True)
    p.add_argument("--rb_csv", required=True)
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
