#!/usr/bin/env python3
"""Build per-class delta table (vanilla vs EAAR) from MLP classifier multiseed JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--multiseed", required=True)
    p.add_argument("--out", default="")
    return p.parse_args()


def main():
    args = parse_args()
    ms_path = Path(args.multiseed).resolve()
    data = json.loads(ms_path.read_text(encoding="utf-8"))
    rows = data.get("per_class_delta", [])
    if not rows:
        raise RuntimeError("No per_class_delta in multiseed file. Re-run multiseed with updated code.")

    df = pd.DataFrame(rows).sort_values("class").reset_index(drop=True)
    out = Path(args.out) if args.out else Path("results") / f"covtype_per_class_{ms_path.stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Covertype Per-Class Delta (EAAR - Vanilla)", "", f"Source: `{ms_path}`", ""]
    try:
        md.append(df.to_markdown(index=False))
    except Exception:
        md.append(df.to_csv(index=False))
    out.write_text("\n".join(md), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

