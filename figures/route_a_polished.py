"""figures/route_a_polished.py — publication-quality Figure 4 (Route A scatter).

Per-CLIP-H/14 dim EEG preservation × Thatcher loading scatter with annotations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(args):
    data = np.load(args.arrays_npz)
    pres = data["preservation"]
    load = data["thatcher_loading"]
    with open(args.summary_json) as f:
        s = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(load, pres, alpha=0.35, s=18, c="#1f77b4",
               edgecolor="none", zorder=2)
    ax.axhline(0, color="black", lw=0.5, ls=":", zorder=1)
    ax.axhline(s["preservation_mean"], color="#d62728", lw=1.0, ls="--",
               zorder=2, label=f"mean preservation = {s['preservation_mean']:.3f}")

    slope, intercept = np.polyfit(load, pres, 1)
    xline = np.array([load.min(), load.max()])
    ax.plot(xline, slope * xline + intercept,
            color="red", lw=1.4, zorder=3,
            label=f"linear fit (slope={slope:.3f})")

    # Quartile lines
    q25 = np.percentile(load, 25)
    q75 = np.percentile(load, 75)
    ax.axvspan(load.min(), q25, color="#cccccc", alpha=0.25, zorder=0,
               label=f"low-loading quartile (mean r={s['low_loading_quartile_preservation_mean']:.3f})")
    ax.axvspan(q75, load.max(), color="#ffcc66", alpha=0.25, zorder=0,
               label=f"high-loading quartile (mean r={s['high_loading_quartile_preservation_mean']:.3f})")

    ax.set_xlabel("Thatcher loading per dim "
                  r"$\frac{|\mathrm{CLIP}(V_1)_d - \mathrm{CLIP}(V_2)_d|}{\sigma_d}$ "
                  "(mean over FFHQ identities)",
                  fontsize=10)
    ax.set_ylabel("EEG preservation per dim "
                  r"$\bar r_d(\mathrm{CLIP\_target},\, \mathrm{ATM\_EEG})$",
                  fontsize=10)
    ax.set_title("ATM EEG bottleneck: uniform per-dim attenuation, "
                 "no Thatcher-dim selectivity",
                 fontsize=13, weight="bold", loc="left", pad=8)

    summary_text = (
        f"Spearman ρ = {s['spearman_rho_observed']:.3f}, "
        f"permutation p = {s['spearman_perm_p_two_sided']:.3f}\n"
        f"High-loading quartile r = {s['high_loading_quartile_preservation_mean']:.3f}  "
        f"vs  Low-loading quartile r = {s['low_loading_quartile_preservation_mean']:.3f}\n"
        f"N = {s['n_subjects']} subjects × {s['n_dims']} CLIP-H/14 dims × "
        f"{s['n_concepts_test']} THINGS-EEG2 test concepts"
    )
    ax.text(0.99, 0.02, summary_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.95))

    ax.grid(alpha=0.3, zorder=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    plt.tight_layout()

    out_png = Path(args.output_prefix + ".png")
    out_pdf = Path(args.output_prefix + ".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved: {out_png}\nsaved: {out_pdf}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arrays_npz", required=True)
    p.add_argument("--summary_json", required=True)
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
