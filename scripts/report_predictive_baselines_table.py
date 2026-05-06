#!/usr/bin/env python3
"""Build predictive baseline markdown table (R2/RMSE/MAE) from baseline JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sml", default="results/baselines_sml2010_sml2010_10seed.json")
    p.add_argument("--energy", default="results/baselines_energy_efficiency_energy_10seed.json")
    p.add_argument("--naval", default="results/baselines_naval_propulsion_naval_10seed.json")
    p.add_argument("--out", default="results/predictive_baselines_full_metrics_20260503.md")
    return p.parse_args()


def _load_rows(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d.get("summary", [])
    out = []
    for r in rows:
        out.append({
            "model": r.get("('model', '')"),
            "r2_mean": r.get("('r2', 'mean')"),
            "r2_std": r.get("('r2', 'std')"),
            "rmse_mean": r.get("('rmse', 'mean')"),
            "mae_mean": r.get("('mae', 'mean')"),
        })
    return out


def main():
    args = parse_args()
    packs = [
        ("sml2010", Path(args.sml)),
        ("energy_efficiency", Path(args.energy)),
        ("naval_propulsion", Path(args.naval)),
    ]
    lines = [
        "# Predictive Baselines (R²/RMSE/MAE)",
        "",
        "| Dataset | Model | R² mean | R² std | RMSE mean | MAE mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ds, path in packs:
        if not path.exists():
            continue
        for row in _load_rows(path):
            lines.append(
                f"| {ds} | {row['model']} | {row['r2_mean']:.6f} | {row['r2_std']:.6f} | "
                f"{row['rmse_mean']:.6f} | {row['mae_mean']:.6f} |"
            )
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

