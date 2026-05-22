"""figures/scaling_law.py — Figure 5: ISI scaling with image-encoder parameter count.

Two emergence curves: image-text contrastive (CLIP / SigLIP / MetaCLIP) vs
vision-only SSL (DINOv2). Plot log10(image-encoder params) on x-axis.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Image-encoder parameter counts (approximate, in millions).
# Sources: original papers + open_clip / HF model cards.
IMAGE_ENCODER_PARAMS_M = {
    # Image-text contrastive family
    "P02_clip_b32":      88,     # CLIP ViT-B/32 image encoder
    "P03_clip_l14":      304,    # CLIP ViT-L/14
    "P04_clip_h14":      632,    # CLIP ViT-H/14 LAION-2B
    "P05_clip_g14":      1012,   # CLIP ViT-g/14 LAION-2B (~1B)
    "P06_clip_bigG14":   1845,   # CLIP ViT-bigG/14 LAION-2B (~1.8B)
    "P19_siglip_base_384": 88,    # SigLIP-base-384
    "P20_siglip_so400m":   400,   # SigLIP SO400M
    "P21_metaclip_h14":    632,
    # DINOv2 self-supervised
    "P07_dinov2_base":   86,
    "P08_dinov2_large":  300,
    "P09_dinov2_giant":  1100,
    # MAE
    "P10_mae_huge":      632,
    # Face-AM angular-margin family (new from E030-E032)
    "P17_facenet_vggface2":     27,
    "P18_facenet_casiawebface": 27,
    "P23_arcface_auraface":     65,     # ResNet-100 (AuraFace)
    "P23rgb_arcface_auraface_rgb": 65,
    "P24_adaface_ir101_ms1mv2":   65,   # IR-101 65.2M
    "P25_arcface_ir101_webface4m": 65,
    "P26_adaface_ir50_casia":     44,   # IR-50 43.6M
    "P28_adaface_ir50_webface4m": 44,
    "P29_adaface_ir50_ms1mv2":    44,
    # Bio-inspired
    "P22_cornet_s":               53,   # CORnet-S
}

FAMILY = {
    "P02_clip_b32": "image-text", "P03_clip_l14": "image-text",
    "P04_clip_h14": "image-text", "P05_clip_g14": "image-text",
    "P06_clip_bigG14": "image-text",
    "P19_siglip_base_384": "image-text", "P20_siglip_so400m": "image-text",
    "P21_metaclip_h14": "image-text",
    "P07_dinov2_base": "DINOv2-SSL", "P08_dinov2_large": "DINOv2-SSL",
    "P09_dinov2_giant": "DINOv2-SSL",
    "P10_mae_huge": "MAE-SSL",
    "P17_facenet_vggface2": "Face-triplet", "P18_facenet_casiawebface": "Face-triplet",
    "P23_arcface_auraface": "Face-AM", "P23rgb_arcface_auraface_rgb": "Face-AM",
    "P24_adaface_ir101_ms1mv2": "Face-AM", "P25_arcface_ir101_webface4m": "Face-AM",
    "P26_adaface_ir50_casia": "Face-AM", "P28_adaface_ir50_webface4m": "Face-AM",
    "P29_adaface_ir50_ms1mv2": "Face-AM",
    "P22_cornet_s": "Bio-inspired",
}
FAMILY_COLOR = {"image-text": "#d62728", "DINOv2-SSL": "#ff7f0e",
                "MAE-SSL": "#bcbd22", "Face-triplet": "#2ca02c",
                "Face-AM": "#1f9c4d", "Bio-inspired": "#9467bd"}

PRETTY_NAME = {
    "P02_clip_b32": "CLIP-B/32", "P03_clip_l14": "CLIP-L/14",
    "P04_clip_h14": "CLIP-H/14", "P05_clip_g14": "CLIP-g/14",
    "P06_clip_bigG14": "CLIP-bigG/14",
    "P19_siglip_base_384": "SigLIP-B", "P20_siglip_so400m": "SigLIP-SO400M",
    "P21_metaclip_h14": "MetaCLIP-H/14",
    "P07_dinov2_base": "DINOv2-B", "P08_dinov2_large": "DINOv2-L",
    "P09_dinov2_giant": "DINOv2-G",
    "P10_mae_huge": "MAE-H",
    "P17_facenet_vggface2": "FaceNet-VGG2", "P18_facenet_casiawebface": "FaceNet-CASIA",
    "P22_cornet_s": "CORnet-S",
    "P23_arcface_auraface": "ArcFace-R100 BGR",
    "P23rgb_arcface_auraface_rgb": "ArcFace-R100 RGB",
    "P24_adaface_ir101_ms1mv2": "AdaFace-IR101 MS1MV2",
    "P25_arcface_ir101_webface4m": "ArcFace-IR101 WF4M",
    "P26_adaface_ir50_casia": "AdaFace-IR50 CASIA",
    "P28_adaface_ir50_webface4m": "AdaFace-IR50 WF4M",
    "P29_adaface_ir50_ms1mv2": "AdaFace-IR50 MS1MV2",
}


def main(args):
    th = pd.read_csv(args.thatcher_csv).set_index("model_id")

    rows = []
    for mid, fam in FAMILY.items():
        if mid not in IMAGE_ENCODER_PARAMS_M or mid not in th.index:
            continue
        rows.append({
            "model_id": mid,
            "family": fam,
            "params_M": IMAGE_ENCODER_PARAMS_M[mid],
            "isi": th.loc[mid, "isi_pert"],
            "isi_lo": th.loc[mid, "isi_pert_ci_lo"],
            "isi_hi": th.loc[mid, "isi_pert_ci_hi"],
        })
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Per-family curves
    for fam in ["image-text", "DINOv2-SSL"]:
        sub = d[d.family == fam].sort_values("params_M")
        x = np.log10(sub.params_M.to_numpy())
        y = sub.isi.to_numpy()
        ylo = sub.isi_lo.to_numpy()
        yhi = sub.isi_hi.to_numpy()
        col = FAMILY_COLOR[fam]
        ax.errorbar(x, y, yerr=[y - ylo, yhi - y], fmt="o",
                    color=col, markersize=10, capsize=4, lw=1.2,
                    markeredgecolor="black", markeredgewidth=0.8,
                    zorder=3, label=f"{fam} (n={len(sub)})")
        # Linear fit on log-params
        if len(sub) >= 3:
            slope, intercept = np.polyfit(x, y, 1)
            x_extrap = np.linspace(min(1.5, x.min() - 0.3), max(3.4, x.max() + 0.3), 100)
            ax.plot(x_extrap, slope * x_extrap + intercept,
                    color=col, ls="--", lw=1.2, alpha=0.7, zorder=2,
                    label=f"  fit: slope={slope:.2f}, ISI(10^x M params)")

    # MAE as single scatter (not enough for fit)
    sub_mae = d[d.family == "MAE-SSL"]
    if len(sub_mae):
        ax.errorbar(np.log10(sub_mae.params_M.to_numpy()),
                    sub_mae.isi.to_numpy(),
                    yerr=[sub_mae.isi.to_numpy() - sub_mae.isi_lo.to_numpy(),
                          sub_mae.isi_hi.to_numpy() - sub_mae.isi.to_numpy()],
                    fmt="s", color=FAMILY_COLOR["MAE-SSL"], markersize=10,
                    markeredgecolor="black", markeredgewidth=0.8,
                    capsize=4, lw=1.2, zorder=3, label="MAE-SSL (n=1)")
    # Face-AM scatter (multiple priors at similar size — show data-inversion)
    sub_am = d[d.family == "Face-AM"]
    if len(sub_am):
        ax.errorbar(np.log10(sub_am.params_M.to_numpy()),
                    sub_am.isi.to_numpy(),
                    yerr=[sub_am.isi.to_numpy() - sub_am.isi_lo.to_numpy(),
                          sub_am.isi_hi.to_numpy() - sub_am.isi.to_numpy()],
                    fmt="D", color=FAMILY_COLOR["Face-AM"], markersize=9,
                    markeredgecolor="black", markeredgewidth=0.8,
                    capsize=4, lw=1.2, zorder=3,
                    label=f"Face-AM (n={len(sub_am)})")
    # Face-triplet scatter
    sub_tri = d[d.family == "Face-triplet"]
    if len(sub_tri):
        ax.errorbar(np.log10(sub_tri.params_M.to_numpy()),
                    sub_tri.isi.to_numpy(),
                    yerr=[sub_tri.isi.to_numpy() - sub_tri.isi_lo.to_numpy(),
                          sub_tri.isi_hi.to_numpy() - sub_tri.isi.to_numpy()],
                    fmt="^", color=FAMILY_COLOR["Face-triplet"], markersize=10,
                    markeredgecolor="black", markeredgewidth=0.8,
                    capsize=4, lw=1.2, zorder=3,
                    label=f"Face-triplet (n={len(sub_tri)})")
    # Bio-inspired (CORnet-S)
    sub_bio = d[d.family == "Bio-inspired"]
    if len(sub_bio):
        ax.errorbar(np.log10(sub_bio.params_M.to_numpy()),
                    sub_bio.isi.to_numpy(),
                    yerr=[sub_bio.isi.to_numpy() - sub_bio.isi_lo.to_numpy(),
                          sub_bio.isi_hi.to_numpy() - sub_bio.isi.to_numpy()],
                    fmt="p", color=FAMILY_COLOR["Bio-inspired"], markersize=10,
                    markeredgecolor="black", markeredgewidth=0.8,
                    capsize=4, lw=1.2, zorder=3,
                    label=f"Bio-inspired (n={len(sub_bio)})")

    # Annotate each point
    for _, r in d.iterrows():
        ax.annotate(PRETTY_NAME[r.model_id],
                    (np.log10(r.params_M) + 0.04, r.isi),
                    fontsize=8, va="center")

    ax.axhline(1.0, color="black", lw=0.7, ls=":", zorder=1)
    ax.axhspan(4.0, 5.0, color="#1f77b4", alpha=0.10, zorder=0)
    ax.axhline(4.0, color="#1f77b4", lw=0.5, ls=":")
    ax.axhline(5.0, color="#1f77b4", lw=0.5, ls=":")

    ax.set_xlabel(r"$\log_{10}$ image-encoder params (M)", fontsize=11)
    ax.set_ylabel("Thatcher ISI", fontsize=11)
    ax.set_title("Scaling of Thatcher ISI with image-encoder size, by training family",
                 fontsize=13, weight="bold", loc="left", pad=8)

    # Predictive annotation: where does image-text family cross ISI=4?
    sub_it = d[d.family == "image-text"].sort_values("params_M")
    if len(sub_it) >= 3:
        slope, intercept = np.polyfit(np.log10(sub_it.params_M.to_numpy()),
                                       sub_it.isi.to_numpy(), 1)
        crossing_x = (4.0 - intercept) / slope  # log10(M params) at ISI=4
        if crossing_x > 0 and crossing_x < 4:
            ax.axvline(crossing_x, color="#1f77b4", lw=0.8, ls=":",
                       alpha=0.5, zorder=1)
            ax.annotate(
                f"image-text family crosses\nlower human-range edge (ISI=4)\nat ~ 10^{crossing_x:.2f} M = {10**crossing_x:.0f}M params",
                xy=(crossing_x, 4.0), xytext=(crossing_x - 0.7, 3.2),
                fontsize=9, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="gray"))

    ax.grid(alpha=0.3, zorder=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_ylim(0, max(d.isi_hi.max() + 0.5, 7.5))
    ax.set_xlim(1.5, 3.5)

    plt.tight_layout()
    out_png = Path(args.output_prefix + ".png")
    out_pdf = Path(args.output_prefix + ".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}\nsaved: {out_pdf}")

    print("\n=== fit summary ===")
    for fam in ["image-text", "DINOv2-SSL"]:
        sub = d[d.family == fam].sort_values("params_M")
        if len(sub) < 3: continue
        slope, intercept = np.polyfit(np.log10(sub.params_M.to_numpy()),
                                       sub.isi.to_numpy(), 1)
        print(f"  {fam}: ISI ≈ {slope:.2f} × log10(M params) + {intercept:.2f}")
        print(f"    at 88M params (smallest typical): {slope * np.log10(88) + intercept:.2f}")
        print(f"    at 1.8B params (largest tested):  {slope * np.log10(1800) + intercept:.2f}")
        print(f"    extrapolation to 10B params:      {slope * np.log10(10000) + intercept:.2f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--thatcher_csv", required=True)
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
