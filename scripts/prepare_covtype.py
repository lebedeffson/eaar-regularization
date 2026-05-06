#!/usr/bin/env python3
"""Download and prepare Covertype dataset to CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_covtype


def main():
    out = Path("data/covtype.csv").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = fetch_covtype(as_frame=True)
    X = ds.data.copy()
    y = ds.target.copy()
    df = pd.DataFrame(X)
    df["target"] = y.astype("int64")
    df.to_csv(out, index=False)
    print(out)
    print(f"shape={df.shape}, n_features={X.shape[1]}, n_classes={df['target'].nunique()}")


if __name__ == "__main__":
    main()

