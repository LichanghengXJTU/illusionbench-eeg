"""stimuli/closeup_compare.py — large-format single-identity 4-condition comparison.

For visual QC: shows ONE identity's V1/V2/V3/V4 at full resolution.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def main(args):
    stim_manifest = pd.read_csv(args.stim_manifest)
    id_manifest = pd.read_csv(args.identity_manifest)
    ids = [int(x) for x in args.identities.split(",")]
    conditions = ["V1_upright_normal", "V2_upright_thatched",
                  "V3_inverted_normal", "V4_inverted_thatched"]
    stim_idx = stim_manifest.set_index(["identity_id", "condition"])

    n_rows = len(ids)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows),
                             squeeze=False)
    titles = ["V1 upright_normal", "V2 upright_thatched (← detect here)",
              "V3 inverted_normal", "V4 inverted_thatched (← should look normal)"]

    for r, iid in enumerate(ids):
        name = id_manifest[id_manifest["identity_id"] == iid]["lfw_name"].iloc[0]
        for c, cond in enumerate(conditions):
            ax = axes[r][c]
            path = stim_idx.loc[(iid, cond), "path"]
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=12)
            if c == 0:
                ax.set_ylabel(f"identity {iid:04d}\n{name}", fontsize=11)

    plt.suptitle("Thatcher stimuli — visual QC at full 224×224 resolution", fontsize=14)
    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved: {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--identity_manifest", required=True)
    p.add_argument("--stim_manifest", required=True)
    p.add_argument("--identities", required=True, help="comma-separated identity IDs")
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
