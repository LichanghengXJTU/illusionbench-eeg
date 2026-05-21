"""figures/route_a_scatter.py — scatter of per-dim EEG preservation × Thatcher loading."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(args):
    data = np.load(Path(args.arrays_npz))
    pres = data["preservation"]
    load = data["thatcher_loading"]
    with open(Path(args.summary_json)) as f:
        summary = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(load, pres, alpha=0.35, s=14, c="#1f77b4", edgecolor="none")
    ax.axhline(0, color="black", lw=0.5, ls=":")

    # Robust trend line (Spearman) shown as a flat reference (we found ~0 slope)
    slope, intercept = np.polyfit(load, pres, 1)
    xline = np.array([load.min(), load.max()])
    ax.plot(xline, slope * xline + intercept,
            color="red", lw=1.2, label=f"linear fit slope={slope:.3f}")

    rho = summary["spearman_rho_observed"]
    p   = summary["spearman_perm_p_two_sided"]
    hi  = summary["high_loading_quartile_preservation_mean"]
    lo  = summary["low_loading_quartile_preservation_mean"]

    ax.set_xlabel("Thatcher loading on dim d  (|V1−V2| / σ_natural, mean over FFHQ identities)")
    ax.set_ylabel("EEG preservation on dim d  (mean Pearson r across 10 subjects)")
    ax.set_title(
        f"Route A — per-CLIP-H/14 dim ({len(pres)} dims)\n"
        f"Spearman ρ = {rho:.3f}  (perm p={p:.3f})  |  High-load quartile r={hi:.3f}  vs  Low-load r={lo:.3f}"
    )

    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    plt.tight_layout()

    out_path = Path(args.output_prefix + ".png")
    out_pdf  = Path(args.output_prefix + ".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    fig.savefig(out_pdf,  bbox_inches="tight")
    print(f"saved: {out_path}\nsaved: {out_pdf}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arrays_npz", required=True)
    p.add_argument("--summary_json", required=True)
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
