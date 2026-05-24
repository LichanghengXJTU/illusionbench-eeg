"""analysis/section6_verdict.py — centralized §6 PASS/FAIL evaluator.

Two threshold regimes are provided side-by-side so the team can compare:

(A) LEGACY — the thresholds used in the first paper outline. PWI ≤ 0.5 is
    a "whole-context dampening" framing that is OPPOSITE to Tanaka 1997.
    CSI ≥ 1.5 has no human anchor.

(B) TANAKA_ALIGNED — Tanaka-direction PWI (whole > part discriminability
    means holistic ⇒ PWI > 1). ISI threshold kept at ≥ 3.0 (Carbon 2005
    human edge ~4-5; we hold one CI below human). CSI replaced with
    pixel-corrected effect-size threshold matched to the DINOv2 best in
    the existing battery (so we still have a discriminating criterion;
    flagged in paper as "no published human ratio benchmark").

A model 'PASSES' overall iff it passes all four criteria simultaneously
under the chosen regime. We always report BOTH regimes so reviewers see
the impact of the framing.

Reads pre-extracted result CSVs (one per paradigm) produced by
analysis.compute_metrics. Does not re-extract embeddings.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# Per-paradigm pixel-baseline for normalization (matches existing pipeline).
PIXEL_BASELINE = {
    "thatcher":   1.000,
    "composite":  1.099,
    "partwhole":  1.202,
    "randombbox": 1.000,
}

# Display name for each paradigm's index.
INDEX_LABEL = {
    "thatcher":   "ISI",
    "composite":  "CSI",
    "partwhole":  "PWI",
    "randombbox": "ISIrbox",
}


@dataclass(frozen=True)
class Criterion:
    op: str        # ">=" or "<="
    threshold: float
    rationale: str  # one-sentence justification


# (A) Legacy thresholds — what the first paper draft used.
LEGACY = {
    "thatcher":   Criterion(">=", 3.0,
                            "below Carbon 2005 human ISI 4-5; defensible"),
    "composite":  Criterion(">=", 1.5,
                            "arbitrary; no published human anchor"),
    "partwhole":  Criterion("<=", 0.5,
                            "OPPOSITE to Tanaka 1997 direction; rewards "
                            "anti-holistic models"),
    "randombbox": Criterion("<=", 1.5,
                            "control: ISI on non-feature regions; OK"),
}

# (B) Tanaka-aligned thresholds — direction matches human psychophysics.
TANAKA_ALIGNED = {
    "thatcher":   Criterion(">=", 3.0,
                            "below Carbon 2005 human ISI 4-5; defensible"),
    "composite":  Criterion(">=", 1.5,
                            "Young 1987 direction (aligned > misaligned "
                            "discrimination); no human ratio benchmark — "
                            "kept matched to legacy for direct comparison"),
    "partwhole":  Criterion(">=", 1.0,
                            "Tanaka 1997 direction: whole-context > part-"
                            "isolated discrimination implies holistic "
                            "binding; PWI > 1 = matches human direction"),
    "randombbox": Criterion("<=", 1.5,
                            "control: same as legacy"),
}


def apply_criterion(value: float, c: Criterion) -> bool:
    if c.op == ">=":
        return value >= c.threshold
    if c.op == "<=":
        return value <= c.threshold
    raise ValueError(f"unknown op {c.op}")


def load_per_model_index(csv_path: Path, paradigm: str,
                          isi_col: str = "isi_pert",
                          model_col: str = "model_id") -> dict[str, float]:
    """Return {model_id: pixel-corrected index} for one paradigm."""
    df = pd.read_csv(csv_path)
    pix = PIXEL_BASELINE[paradigm]
    return {str(r[model_col]): float(r[isi_col]) / pix for _, r in df.iterrows()}


def assemble(per_paradigm: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Stack into a wide DataFrame: rows=model_id, cols=per-paradigm index."""
    models = sorted({m for d in per_paradigm.values() for m in d.keys()})
    rows = []
    for m in models:
        row = {"model_id": m}
        for para, idx_map in per_paradigm.items():
            row[INDEX_LABEL[para]] = idx_map.get(m, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def verdict_table(values_df: pd.DataFrame, criteria: dict[str, Criterion]
                  ) -> pd.DataFrame:
    """Add per-criterion PASS/FAIL columns and an overall 4/4 flag."""
    out = values_df.copy()
    pass_cols = []
    for para, c in criteria.items():
        label = INDEX_LABEL[para]
        verdict_col = f"{label}_pass"
        # If the index is NaN, we mark FAIL conservatively.
        out[verdict_col] = out[label].apply(
            lambda v: bool(apply_criterion(v, c)) if pd.notna(v) else False
        )
        pass_cols.append(verdict_col)
    out["n_pass"] = out[pass_cols].sum(axis=1).astype(int)
    out["all_four_pass"] = out["n_pass"] == 4
    return out


def render_comparison(per_paradigm: dict[str, dict[str, float]]) -> str:
    """Produce the side-by-side LEGACY vs TANAKA_ALIGNED verdict table."""
    values = assemble(per_paradigm)
    legacy = verdict_table(values, LEGACY)
    tanaka = verdict_table(values, TANAKA_ALIGNED)

    lines = []
    lines.append("# §6 Verdict — Legacy vs Tanaka-aligned thresholds\n")
    lines.append("## Threshold regimes\n")
    for name, criteria in [("LEGACY", LEGACY), ("TANAKA_ALIGNED", TANAKA_ALIGNED)]:
        lines.append(f"\n### {name}\n")
        for para, c in criteria.items():
            lines.append(f"- **{INDEX_LABEL[para]}** {c.op} {c.threshold}  "
                         f"— *{c.rationale}*")
    lines.append("\n## Per-model verdict\n")
    lines.append("| model_id | ISI | CSI | PWI | ISIrbox | "
                 "Legacy n/4 | Legacy 4/4 | Tanaka n/4 | Tanaka 4/4 | Δ |")
    lines.append("|---|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|")

    for _, row in values.iterrows():
        m = row["model_id"]
        leg = legacy.loc[legacy["model_id"] == m].iloc[0]
        tan = tanaka.loc[tanaka["model_id"] == m].iloc[0]
        delta = int(tan["n_pass"]) - int(leg["n_pass"])
        delta_str = f"{delta:+d}" if delta != 0 else "0"
        lines.append(
            f"| {m} | {row['ISI']:.3f} | {row['CSI']:.3f} | "
            f"{row['PWI']:.3f} | {row['ISIrbox']:.3f} | "
            f"{int(leg['n_pass'])}/4 | "
            f"{'✓' if leg['all_four_pass'] else '✗'} | "
            f"{int(tan['n_pass'])}/4 | "
            f"{'✓' if tan['all_four_pass'] else '✗'} | "
            f"{delta_str} |"
        )

    n_legacy = int(legacy["all_four_pass"].sum())
    n_tanaka = int(tanaka["all_four_pass"].sum())
    n_models = len(legacy)
    lines.append(f"\n**Summary**: {n_legacy}/{n_models} models PASS under "
                 f"LEGACY; {n_tanaka}/{n_models} models PASS under "
                 f"TANAKA_ALIGNED.\n")

    # Where the regimes disagree
    disagreements = []
    for _, lr in legacy.iterrows():
        m = lr["model_id"]
        tr = tanaka.loc[tanaka["model_id"] == m].iloc[0]
        for para in ["thatcher", "composite", "partwhole", "randombbox"]:
            label = INDEX_LABEL[para]
            if bool(lr[f"{label}_pass"]) != bool(tr[f"{label}_pass"]):
                disagreements.append((m, label,
                                       bool(lr[f"{label}_pass"]),
                                       bool(tr[f"{label}_pass"])))
    if disagreements:
        lines.append("\n## Disagreements between regimes\n")
        lines.append("| model_id | index | Legacy | Tanaka |")
        lines.append("|---|---|:-:|:-:|")
        for m, lab, lp, tp in disagreements:
            lines.append(f"| {m} | {lab} | "
                         f"{'PASS' if lp else 'FAIL'} | "
                         f"{'PASS' if tp else 'FAIL'} |")

    return "\n".join(lines)


def main(args):
    repo = Path(args.repo_root)
    per_paradigm: dict[str, dict[str, float]] = {}

    # Explicit per-paradigm CSVs (overrides --csv_pattern when given).
    explicit = {
        "thatcher":   args.thatcher_csv,
        "composite":  args.composite_csv,
        "partwhole":  args.partwhole_csv,
        "randombbox": args.randombbox_csv,
    }
    for para in ["thatcher", "composite", "partwhole", "randombbox"]:
        if explicit[para]:
            csv_path = Path(explicit[para])
            if not csv_path.is_absolute():
                csv_path = repo / csv_path
        else:
            csv_path = repo / "outputs/tables" / args.csv_pattern.format(paradigm=para)
        if not csv_path.exists():
            print(f"  [skip] missing {csv_path}")
            per_paradigm[para] = {}
            continue
        per_paradigm[para] = load_per_model_index(csv_path, para)

    if not any(per_paradigm.values()):
        raise SystemExit(
            f"No paradigm CSVs found under {repo}/outputs/tables/ "
            f"matching pattern {args.csv_pattern!r}"
        )

    report = render_comparison(per_paradigm)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(report)
    print(f"\nWrote: {out_path}")

    # Also dump structured JSON
    json_path = out_path.with_suffix(".json")
    values = assemble(per_paradigm)
    legacy = verdict_table(values, LEGACY)
    tanaka = verdict_table(values, TANAKA_ALIGNED)
    payload = {
        "models": list(values["model_id"]),
        "values": values.set_index("model_id").to_dict(orient="index"),
        "legacy_pass": legacy.set_index("model_id")[
            ["n_pass", "all_four_pass"]].to_dict(orient="index"),
        "tanaka_pass": tanaka.set_index("model_id")[
            ["n_pass", "all_four_pass"]].to_dict(orient="index"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote: {json_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", default=".",
                   help="repository root (default: cwd)")
    p.add_argument("--csv_pattern",
                   default="{paradigm}_isi.csv",
                   help="filename pattern under outputs/tables/, with "
                        "{paradigm} placeholder. e.g. '{paradigm}_ffhq_isi.csv'")
    p.add_argument("--thatcher_csv", default=None,
                   help="explicit Thatcher result CSV (overrides --csv_pattern)")
    p.add_argument("--composite_csv", default=None,
                   help="explicit composite result CSV")
    p.add_argument("--partwhole_csv", default=None,
                   help="explicit part-whole result CSV")
    p.add_argument("--randombbox_csv", default=None,
                   help="explicit random-bbox control result CSV")
    p.add_argument("--output", default="outputs/tables/section6_verdict.md")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
