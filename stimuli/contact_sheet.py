"""stimuli/contact_sheet.py — visual QC contact sheet for a paradigm.

Builds a grid: N_identities rows × N_conditions cols, with column headers.
Saves a single PNG/PDF for fast eyeballing.

Usage:
  python -m stimuli.contact_sheet \
      --identity_manifest data/stimuli/identity_manifest.csv \
      --stim_manifest data/stimuli/thatcher_manifest.csv \
      --conditions V1_upright_normal,V2_upright_thatched,V3_inverted_normal,V4_inverted_thatched \
      --n_identities 16 \
      --output outputs/figures/qc_thatcher_contactsheet.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def main(args):
    id_manifest = pd.read_csv(args.identity_manifest)
    stim_manifest = pd.read_csv(args.stim_manifest)
    conditions = [c.strip() for c in args.conditions.split(",")]

    rng = np.random.default_rng(args.seed)
    if args.sample == "first":
        ids = id_manifest["identity_id"].to_list()[: args.n_identities]
    else:
        ids = rng.choice(id_manifest["identity_id"].to_list(),
                         size=args.n_identities, replace=False).tolist()

    n_rows = len(ids)
    n_cols = len(conditions)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 2.2 * n_rows),
                             squeeze=False)

    # Build a lookup
    stim_index = stim_manifest.set_index(["identity_id", "condition"])

    for r, iid in enumerate(ids):
        lfw_name = id_manifest[id_manifest["identity_id"] == iid]["lfw_name"].iloc[0]
        for c, cond in enumerate(conditions):
            ax = axes[r][c]
            try:
                path = stim_index.loc[(iid, cond), "path"]
                img = Image.open(path).convert("RGB")
                ax.imshow(img)
            except KeyError:
                ax.text(0.5, 0.5, f"missing\n{cond}",
                        ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cond.replace("_", " "), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"id{iid:04d}\n{lfw_name[:14]}", fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=40)

    plt.suptitle(f"{args.title}  (n={n_rows} identities)", fontsize=12)
    plt.tight_layout(rect=[0.03, 0, 1, 0.98])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--identity_manifest", required=True)
    p.add_argument("--stim_manifest", required=True)
    p.add_argument("--conditions", required=True)
    p.add_argument("--n_identities", type=int, default=16)
    p.add_argument("--sample", choices=["first", "random"], default="first")
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--title", default="Thatcher stimulus QC")
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
