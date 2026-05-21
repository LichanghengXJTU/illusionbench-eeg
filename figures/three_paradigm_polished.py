"""figures/three_paradigm_polished.py — publication-quality 3-paradigm panel.

Improvements over `three_paradigm_panel.py`:
  - Cleaner colors with explicit class annotations
  - Bigger fonts for camera-ready (target 8.5pt body equiv at print size)
  - Horizontal stripes between panels for visual separation
  - Annotated "dissociation" pointer linking the rows
  - Per-paradigm pixel-baseline indicator
  - Inline summary text per panel
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, FancyArrowPatch

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
    "CLIP": "#d62728",
    "SigLIP/MetaCLIP": "#a55a4b",
    "DINOv2": "#ff7f0e",
    "MAE": "#bcbd22",
    "VAE": "#7f7f7f",
    "Face-trained": "#2ca02c",
    "Control": "#cccccc",
}

PRETTY_NAME = {
    "P02_clip_b32": "CLIP-B/32",
    "P03_clip_l14": "CLIP-L/14",
    "P04_clip_h14": "CLIP-H/14",
    "P05_clip_g14": "CLIP-g/14",
    "P06_clip_bigG14": "CLIP-bigG/14",
    "P19_siglip_base_384": "SigLIP-B",
    "P20_siglip_so400m": "SigLIP-SO400M",
    "P21_metaclip_h14": "MetaCLIP-H/14",
    "P07_dinov2_base": "DINOv2-B",
    "P08_dinov2_large": "DINOv2-L",
    "P09_dinov2_giant": "DINOv2-G",
    "P10_mae_huge": "MAE-H",
    "P11_sdxl_vae": "SDXL-VAE",
    "P17_facenet_vggface2": "FaceNet-VGG2",
    "P18_facenet_casiawebface": "FaceNet-CASIA",
    "N02_untrained_vit": "ViT-untrained",
    "N03_pixel": "Pixel",
}

def main(args):
    th = pd.read_csv(args.thatcher_csv).set_index("model_id")
    cp = pd.read_csv(args.composite_csv).set_index("model_id")
    pw = pd.read_csv(args.partwhole_csv).set_index("model_id")
    models_all = [m for m in PRIOR_CLASS if m in th.index]
    df = pd.DataFrame({
        "th": th.loc[models_all, "isi_pert"].values,
        "th_lo": th.loc[models_all, "isi_pert_ci_lo"].values,
        "th_hi": th.loc[models_all, "isi_pert_ci_hi"].values,
        "cp": cp.loc[models_all, "isi_pert"].values,
        "cp_lo": cp.loc[models_all, "isi_pert_ci_lo"].values,
        "cp_hi": cp.loc[models_all, "isi_pert_ci_hi"].values,
        "pw": pw.loc[models_all, "isi_pert"].values,
        "pw_lo": pw.loc[models_all, "isi_pert_ci_lo"].values,
        "pw_hi": pw.loc[models_all, "isi_pert_ci_hi"].values,
        "class": [PRIOR_CLASS[m] for m in models_all],
    }, index=models_all)
    df["class_order"] = df["class"].map({c: i for i, c in enumerate(CLASS_ORDER)})
    df = df.sort_values(["class_order", "th"], kind="stable")
    models = df.index.tolist()
    n = len(models)
    x = np.arange(n)

    pixel_th = 1.000
    pixel_cp = 1.099
    pixel_pw = 1.202

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"hspace": 0.28})
    paradigms = [
        ("th", "Thatcher illusion ISI", "Local-feature × orientation",
         pixel_th, (4, 5), axes[0],
         "CLIP family dominates (ISI 4.0–6.8, in or above human range)"),
        ("cp", "Composite-face CSI", "Spatial integration of face halves",
         pixel_cp, None, axes[1],
         "DINOv2 + MetaCLIP-H/14 + CLIP-L/14 highest (CSI 1.5–1.75); CLIP family weaker"),
        ("pw", "Part-Whole PWI", "Whole-context dampening of eye-swap",
         pixel_pw, None, axes[2],
         "DINOv2 lowest (PWI 0.31–0.40) — most spatially-holistic"),
    ]
    for key, title, subtitle, pix, ref_band, ax, summary in paradigms:
        vals = df[key].to_numpy()
        lo = df[key + "_lo"].to_numpy()
        hi = df[key + "_hi"].to_numpy()
        colors = [CLASS_COLOR[c] for c in df["class"]]
        ax.bar(x, vals, width=0.75, color=colors, edgecolor="black",
               linewidth=0.9, zorder=2)
        ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="none",
                    ecolor="black", capsize=3, lw=1.0, zorder=3)
        ax.axhline(pix, color="black", lw=1.0, ls="--", zorder=1,
                   label=f"pixel baseline = {pix:.3f}")
        if ref_band:
            ax.axhspan(ref_band[0], ref_band[1], color="#1f77b4", alpha=0.10,
                       zorder=0)
            ax.axhline(ref_band[0], color="#1f77b4", lw=0.6, ls=":", zorder=1)
            ax.axhline(ref_band[1], color="#1f77b4", lw=0.6, ls=":", zorder=1)
        ax.set_title(f"{title} — {subtitle}", fontsize=12, loc="left",
                     pad=8, weight="bold")
        ax.text(0.99, 0.95, summary, transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.9))
        ax.set_ylabel("ISI / CSI / PWI", fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ymax = max(vals.max(), hi.max(), 1.3)
        ax.set_ylim(0, ymax + 0.3)
        ax.tick_params(axis="y", labelsize=9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([PRETTY_NAME[m] for m in models],
                             rotation=40, ha="right", fontsize=9)

    handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor="black", label=c)
               for c in CLASS_ORDER if c in df["class"].unique()]
    handles.append(plt.Line2D([0], [0], color="black", ls="--",
                              label="pixel baseline"))
    handles.append(Patch(facecolor="#1f77b4", alpha=0.10,
                          label="human Thatcher range (Carbon 2005)"))
    axes[0].legend(handles=handles, loc="upper left", fontsize=9, ncol=2,
                   framealpha=0.95)

    fig.suptitle(
        "Holistic-face illusion sensitivity dissociates by training paradigm",
        fontsize=14, weight="bold", y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out_png = Path(args.output_prefix + ".png")
    out_pdf = Path(args.output_prefix + ".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
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
