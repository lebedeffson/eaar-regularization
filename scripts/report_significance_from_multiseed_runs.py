#!/usr/bin/env python3
"""Paired significance from one multiseed JSON containing vanilla/ea metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon, ttest_rel
except Exception:
    wilcoxon = None
    ttest_rel = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--multiseed", required=True)
    p.add_argument("--metric-ea", required=True, help="run key for EA metric")
    p.add_argument("--metric-vanilla", required=True, help="run key for vanilla metric")
    p.add_argument("--out", default="")
    p.add_argument("--label", default="")
    return p.parse_args()


def cliffs_delta(x, y):
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    n = len(x) * len(y)
    return float((gt - lt) / n) if n else float("nan")


def main():
    a = parse_args()
    d = json.loads(Path(a.multiseed).read_text(encoding="utf-8"))
    runs = d.get("runs", [])
    ea_vals, v_vals, seeds = [], [], []
    for r in runs:
        ve = r.get(a.metric_ea)
        vv = r.get(a.metric_vanilla)
        if ve is None or vv is None:
            continue
        ve = float(ve)
        vv = float(vv)
        if not np.isfinite(ve) or not np.isfinite(vv):
            continue
        ea_vals.append(ve)
        v_vals.append(vv)
        seeds.append(int(r.get("seed", -1)))
    x = np.asarray(ea_vals, dtype=float)
    y = np.asarray(v_vals, dtype=float)
    dlt = x - y
    mean = float(np.mean(dlt)) if dlt.size else float("nan")
    std = float(np.std(dlt, ddof=1)) if dlt.size > 1 else 0.0
    med = float(np.median(dlt)) if dlt.size else float("nan")
    ci = (
        float(mean - 1.96 * std / np.sqrt(max(1, dlt.size))),
        float(mean + 1.96 * std / np.sqrt(max(1, dlt.size))),
    ) if dlt.size else (float("nan"), float("nan"))
    p_w = float(wilcoxon(x, y, zero_method="wilcox").pvalue) if (dlt.size >= 3 and wilcoxon is not None) else None
    p_t = float(ttest_rel(x, y).pvalue) if (dlt.size >= 3 and ttest_rel is not None) else None
    out = {
        "source": str(Path(a.multiseed).resolve()),
        "label": a.label or f"{a.metric_ea}-{a.metric_vanilla}",
        "metric_ea": a.metric_ea,
        "metric_vanilla": a.metric_vanilla,
        "unit_of_analysis": "seed-level paired comparison",
        "n": int(dlt.size),
        "seeds": seeds,
        "deltas": dlt.tolist(),
        "delta_mean": mean,
        "delta_std": std,
        "delta_median": med,
        "ci95": {"low": ci[0], "high": ci[1]},
        "wins": int(np.sum(dlt > 0)),
        "losses": int(np.sum(dlt < 0)),
        "wilcoxon_p": p_w,
        "ttest_p": p_t,
        "cohen_d": float(mean / (std + 1e-12)) if dlt.size > 1 else float("nan"),
        "cliffs_delta": cliffs_delta(x, y) if dlt.size else float("nan"),
    }
    out_path = Path(a.out) if a.out else Path("results") / f"significance_{Path(a.multiseed).stem}_{a.metric_ea}_vs_{a.metric_vanilla}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

