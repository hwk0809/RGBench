#!/usr/bin/env python3
"""Compare your RGBench run against the paper's published baselines.

Walks an ``outputs/`` directory produced by ``scripts/run_benchmark.py``,
aggregates each cell's metrics the same way the paper does (mean across
samples of the eval-window summary row), then joins against
``results/paper_baselines.csv`` and prints a per-cell ranking plus a
per-simulator summary.

Usage:

    # After running the benchmark with your new simulator
    python scripts/compare_to_paper.py outputs/ --simulator-name MySim

    # Restrict to one mode or one metric
    python scripts/compare_to_paper.py outputs/ --mode fixed_point --metric cd_l1_r2s

The script makes one assumption: your run output follows the standard
hydra layout
``outputs/<garment>/<action>/<simulator>/<mode>/<robot>/sample_<NN>/<ts>/metrics.csv``
— matching what ``scripts/run_benchmark.py`` produces.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "results" / "paper_baselines.csv"

# Map paper_baselines.csv column names to metrics.csv column names
METRIC_MAP = {
    "cd_l1_s2r": "chamfer_l1_sim_to_real",
    "cd_l2_s2r": "chamfer_l2_sim_to_real",
    "cd_l1_r2s": "chamfer_l1_real_to_sim",
    "hd_s2r":    "one_sided_hausdorff_sim_to_real",
    "hd_r2s":    "one_sided_hausdorff_real_to_sim",
    "sim_stab":  "sim_stability_score",
    "z_err":     "z_mean_error",
}
DEFAULT_METRICS = ["cd_l1_s2r", "cd_l1_r2s", "hd_s2r", "hd_r2s"]  # paper Tables 4/5


def collect_user_runs(outputs_dir: Path) -> pd.DataFrame:
    """Walk outputs/ and read every metrics.csv into one long DataFrame.

    Returns columns: garment, action, simulator, mode, sample, <metric cols>.
    Each row is one sample run. ``iloc[-2]`` of each metrics.csv (the
    eval-window summary mean) is taken as the canonical per-run value.
    """
    rows = []
    for csv in outputs_dir.rglob("metrics.csv"):
        # Expected path:
        # outputs/<garment>/<action>/<sim>/<mode>/<robot>/sample_<NN>/<ts>/metrics.csv
        parts = csv.relative_to(outputs_dir).parts
        if len(parts) < 7:
            continue  # skip non-conforming paths
        garment, action, sim, mode, _robot, sample = parts[:6]
        if not sample.startswith("sample_"):
            continue
        try:
            df = pd.read_csv(csv)
        except Exception as exc:
            print(f"  WARN: failed to read {csv}: {exc}", file=sys.stderr)
            continue
        if len(df) < 2:
            continue  # need at least 2 rows (body + summary)
        # Paper rule: iloc[-2] is the eval-window summary mean
        summary = df.iloc[-2]
        row = {"garment": garment, "action": action, "simulator": sim,
               "mode": mode, "sample": sample}
        for short, full in METRIC_MAP.items():
            row[short] = summary.get(full)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_user(user_df: pd.DataFrame) -> pd.DataFrame:
    """Average across samples within each (garment, action, simulator, mode) cell."""
    group_cols = ["garment", "action", "simulator", "mode"]
    metric_cols = list(METRIC_MAP.keys())
    agg = user_df.groupby(group_cols, as_index=False)[metric_cols].mean()
    agg["n_samples"] = user_df.groupby(group_cols).size().values
    return agg


def compare(user_agg: pd.DataFrame, paper: pd.DataFrame,
            metric: str) -> pd.DataFrame:
    """Join user against paper baselines and compute per-cell ranks.

    Lower is better for all paper metrics. Rank 1 = best in cell.
    Each row carries its actual user simulator name (no override).
    """
    paper_pivot = paper.pivot_table(
        index=["garment", "action", "mode"],
        columns="simulator",
        values=metric,
    ).reset_index()
    paper_pivot.columns.name = None
    paper_pivot = paper_pivot.rename(columns={
        "pybullet":        "paper.pybullet",
        "isaacsim":        "paper.isaacsim",
        "mujoco_style3d":  "paper.garmentdyn",
    })

    user_subset = user_agg[["garment", "action", "mode", "simulator",
                            metric, "n_samples"]].copy()
    user_subset = user_subset.rename(columns={metric: "you", "simulator": "your_sim"})

    merged = user_subset.merge(paper_pivot, on=["garment", "action", "mode"], how="left")

    # Per-row rank (lower = better)
    score_cols = ["you", "paper.pybullet", "paper.isaacsim", "paper.garmentdyn"]
    def rank_row(row):
        scores = [(c, row[c]) for c in score_cols if pd.notna(row[c])]
        scores.sort(key=lambda x: x[1])
        ranks = {c: i + 1 for i, (c, _) in enumerate(scores)}
        return ranks.get("you"), len(scores)
    rk = merged.apply(rank_row, axis=1, result_type="expand")
    rk.columns = ["your_rank", "n_competitors"]
    merged = pd.concat([merged, rk], axis=1)
    return merged


def print_table(df: pd.DataFrame, metric: str) -> None:
    cols = ["garment", "action", "mode", "your_sim",
            "you", "paper.pybullet", "paper.isaacsim", "paper.garmentdyn",
            "your_rank", "n_competitors"]
    show = df[cols].copy()
    for c in ["you", "paper.pybullet", "paper.isaacsim", "paper.garmentdyn"]:
        show[c] = show[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "    -")
    show["rank"] = show.apply(
        lambda r: f"{int(r['your_rank'])}/{int(r['n_competitors'])}"
                  if pd.notna(r['your_rank']) else "n/a",
        axis=1)
    show = show.drop(columns=["your_rank", "n_competitors"])
    show.columns = ["garment", "action", "mode", "your_sim",
                    f"you.{metric}", "paper.pybullet", "paper.isaacsim",
                    "paper.garmentdyn", "rank"]
    print(show.to_string(index=False))


def summary(df: pd.DataFrame, metric: str) -> str:
    valid = df[df["your_rank"].notna()]
    if valid.empty:
        return "No cells could be ranked (no overlap between your runs and paper baselines)."
    out = [f"Metric: {metric}", f"Cells ranked: {len(valid)}"]
    for sim_name, group in valid.groupby("your_sim"):
        ranks = group["your_rank"].astype(int)
        out.append(f"\n[{sim_name}]  n={len(group)}, mean rank = {ranks.mean():.2f}/{int(group['n_competitors'].mean())}")
        for r in (1, 2, 3, 4):
            out.append(f"    rank {r}: {(ranks == r).sum()}")
        for paper_col, label in [("paper.pybullet", "PyBullet"),
                                 ("paper.isaacsim", "IsaacSim"),
                                 ("paper.garmentdyn", "GarmentDynamics")]:
            both = group.dropna(subset=["you", paper_col])
            if both.empty:
                continue
            better = (both["you"] < both[paper_col]).sum()
            out.append(f"    better than paper {label}: {better}/{len(both)}")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outputs_dir", help="Path to outputs/ produced by run_benchmark.py")
    p.add_argument("--baseline-csv", default=str(DEFAULT_BASELINE),
                   help=f"Path to paper_baselines.csv (default: {DEFAULT_BASELINE}).")
    p.add_argument("--metric", default="cd_l1_r2s",
                   choices=list(METRIC_MAP.keys()),
                   help="Metric column to rank on (default: cd_l1_r2s).")
    p.add_argument("--mode", default=None, choices=["fixed_point", "robot"],
                   help="Restrict comparison to one mode (default: both).")
    p.add_argument("--output-csv", default=None,
                   help="Optionally write the merged comparison table to this CSV.")
    args = p.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    if not outputs_dir.is_dir():
        print(f"ERROR: outputs_dir does not exist: {outputs_dir}", file=sys.stderr)
        return 2

    baseline_csv = Path(args.baseline_csv)
    if not baseline_csv.is_file():
        print(f"ERROR: paper baselines not found: {baseline_csv}", file=sys.stderr)
        return 2

    paper = pd.read_csv(baseline_csv)
    if args.mode:
        paper = paper[paper["mode"] == args.mode]

    print(f"Reading user runs from: {outputs_dir}")
    user_df = collect_user_runs(outputs_dir)
    if user_df.empty:
        print("No metrics.csv files found.", file=sys.stderr)
        return 1
    print(f"  Found {len(user_df)} sample runs across "
          f"{user_df['simulator'].nunique()} simulator(s), "
          f"{user_df['garment'].nunique()} garment(s), "
          f"{user_df['mode'].nunique()} mode(s).")

    user_agg = aggregate_user(user_df)
    if args.mode:
        user_agg = user_agg[user_agg["mode"] == args.mode]

    merged = compare(user_agg, paper, args.metric)

    print()
    print_table(merged, args.metric)
    print()
    print(summary(merged, args.metric))

    if args.output_csv:
        out = Path(args.output_csv)
        merged.to_csv(out, index=False)
        print(f"\nMerged comparison written to: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
