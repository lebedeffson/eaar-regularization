#!/usr/bin/env python3
"""Run MLP classifier vanilla vs EAAR multi-seed with CE-deletion faithfulness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mlp_classifier import MLPClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--python", default="/home/lebedeffson/Code/venv_cuda/bin/python")
    p.add_argument("--tag-prefix", default="covtype_mlp_eaar3")
    p.add_argument("--k-list", default="1,2,3,4,5")
    p.add_argument("--mask", choices=["permute", "mean", "noise"], default="permute")
    p.add_argument("--random-trials", type=int, default=20)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _mean(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _class_map(per_class):
    out = {}
    for row in per_class or []:
        cls = int(row.get("class"))
        out[cls] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1", 0.0)),
            "support": int(row.get("support", 0)),
        }
    return out


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _apply_mask(X, cols, mode, rng):
    X2 = np.asarray(X, dtype=float).copy()
    for j in cols:
        if mode == "permute":
            idx = rng.permutation(X2.shape[0])
            X2[:, j] = X2[idx, j]
        elif mode == "mean":
            X2[:, j] = float(np.mean(X2[:, j]))
        else:
            std = float(np.std(X2[:, j]))
            X2[:, j] = X2[:, j] + rng.normal(0.0, 0.1 * std + 1e-8, size=X2.shape[0])
    return X2


def _load_importance(path: Path):
    df = pd.read_csv(path)
    col = "importance" if "importance" in df.columns else df.columns[-1]
    v = np.asarray(df[col], dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.maximum(v, 0.0)
    s = float(np.sum(v))
    if s <= 1e-12:
        return np.full_like(v, 1.0 / max(1, v.size))
    return v / s


def _load_logits(model_path, X_train, X_eval, input_dim, num_classes, dropout):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_eval_sc = scaler.transform(X_eval)
    model = MLPClassifier(input_dim, num_classes, dropout=dropout)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_eval_sc, dtype=torch.float32)).cpu().numpy()
    return logits


def _ce(y_true, logits):
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
    return float(log_loss(y_true, probs, labels=np.unique(y_true)))


def _deletion_ce_metrics(model_path, X_train, X_test, y_test, imp, input_dim, num_classes, dropout, k_list, mask, seed, random_trials):
    def predict_fn(Xm):
        return _load_logits(model_path, X_train, Xm, input_dim, num_classes, dropout)

    logits0 = predict_fn(X_test)
    ce0 = _ce(y_test, logits0)

    n_features = imp.size
    ks = sorted(set(k for k in k_list if 0 < k <= n_features))
    x, topv, botv, randv = [], [], [], []
    for k in ks:
        top = np.argsort(imp)[::-1][:k].tolist()
        bot = np.argsort(imp)[:k].tolist()
        rng = np.random.default_rng(seed + 111 * (k + 1))
        dt = _ce(y_test, predict_fn(_apply_mask(X_test, top, mask, rng))) - ce0
        db = _ce(y_test, predict_fn(_apply_mask(X_test, bot, mask, rng))) - ce0
        dr_trials = []
        for t in range(max(1, int(random_trials))):
            rng_t = np.random.default_rng(seed + 1009 * (k + 1) + 7919 * (t + 1))
            cols = rng_t.choice(n_features, size=k, replace=False).tolist()
            dr_trials.append(_ce(y_test, predict_fn(_apply_mask(X_test, cols, mask, rng_t))) - ce0)
        dr = float(np.mean(dr_trials))
        x.append(float(k) / float(max(ks)))
        topv.append(float(dt))
        botv.append(float(db))
        randv.append(float(dr))

    xa = np.asarray(x, dtype=float)
    topa = np.asarray(topv, dtype=float)
    bota = np.asarray(botv, dtype=float)
    randa = np.asarray(randv, dtype=float)
    auc_top = _trapz(topa, xa) if topa.size > 1 else float(topa[0])
    auc_bottom = _trapz(bota, xa) if bota.size > 1 else float(bota[0])
    auc_random = _trapz(randa, xa) if randa.size > 1 else float(randa[0])
    return {
        "auc_top": auc_top,
        "auc_bottom": auc_bottom,
        "auc_random": auc_random,
        "auc_gap": float(auc_top - auc_bottom),
        "top_random_ratio": float(auc_top / max(auc_random, 1e-12)),
    }


def _write_cfg(base_cfg, seed, enabled, tmp_path):
    cfg = deepcopy(base_cfg)
    cfg.setdefault("dataset", {})["random_state"] = int(seed)
    cfg.setdefault("model", {})["seed"] = int(seed)
    cfg.setdefault("eaar", {})["enabled"] = bool(enabled)
    tmp_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _run_train(python_exe, cfg_path, tag, mode):
    cmd = [python_exe, "scripts/train_mlp_classifier_eaar.py", "--config", str(cfg_path), "--tag", tag, "--mode", mode]
    res = subprocess.run(cmd, check=True, text=True, capture_output=True)
    lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
    summary = None
    for ln in lines:
        if "training_summary_" in ln and ln.endswith(".json"):
            summary = Path(ln)
            break
    if summary is None:
        raise RuntimeError(f"Cannot parse summary path from output:\n{res.stdout}\n{res.stderr}")
    return summary.resolve()


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    k_list = [int(s.strip()) for s in args.k_list.split(",") if s.strip()]
    out_path = Path(args.out) if args.out else Path("results") / f"mlp_classifier_eaar_multiseed_{cfg_path.stem}_{args.tag_prefix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / "_tmp_mlp_cls_multiseed"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ds = base_cfg["dataset"]
    mcfg = base_cfg.get("model", {})
    dropout = float(mcfg.get("dropout", 0.1))
    df = pd.read_csv(ds["train_data"])
    target_col = ds.get("target_column", "target")
    if "feature_columns" in ds and ds["feature_columns"]:
        feature_cols = list(ds["feature_columns"])
    else:
        feature_cols = [c for c in df.columns if c != target_col]
    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = np.asarray(df[target_col], dtype=np.int64)
    y_all = y_all - y_all.min()
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    max_samples = int(ds.get("max_samples", 0) or 0)
    if max_samples > 0 and X_all.shape[0] > max_samples:
        keep, _ = train_test_split(np.arange(X_all.shape[0]), train_size=max_samples, random_state=42, stratify=y_all)
        keep = np.asarray(keep, dtype=int)
        X_all, y_all = X_all[keep], y_all[keep]

    rows = []
    per_class_bucket = {}
    for seed in seeds:
        cfg_v = tmp_dir / f"{cfg_path.stem}_s{seed}_vanilla.yaml"
        cfg_e = tmp_dir / f"{cfg_path.stem}_s{seed}_ea.yaml"
        _write_cfg(base_cfg, seed, False, cfg_v)
        _write_cfg(base_cfg, seed, True, cfg_e)
        sum_v = _run_train(args.python, cfg_v, f"{args.tag_prefix}_s{seed}_vanilla", "vanilla")
        sum_e = _run_train(args.python, cfg_e, f"{args.tag_prefix}_s{seed}_ea", "ea")
        sv = json.loads(sum_v.read_text(encoding="utf-8"))
        se = json.loads(sum_e.read_text(encoding="utf-8"))

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=float(ds.get("test_size", 0.2)), random_state=seed, stratify=y_all
        )
        in_dim = X_train.shape[1]
        num_classes = int(np.max(y_train)) + 1

        out_dir_v = Path(sv["model_state_path"]).resolve().parent
        out_dir_e = Path(se["model_state_path"]).resolve().parent
        imp_v = _load_importance(out_dir_v / sv["saved_files"]["feature_importance"])
        imp_e = _load_importance(out_dir_e / se["saved_files"]["feature_importance"])

        dm_v = _deletion_ce_metrics(
            sv["model_state_path"], X_train, X_test, y_test, imp_v,
            in_dim, num_classes, dropout, k_list, args.mask, seed, args.random_trials
        )
        dm_e = _deletion_ce_metrics(
            se["model_state_path"], X_train, X_test, y_test, imp_e,
            in_dim, num_classes, dropout, k_list, args.mask, seed, args.random_trials
        )

        rows.append({
            "seed": seed,
            "vanilla_accuracy": float(sv["metrics"]["accuracy"]),
            "ea_accuracy": float(se["metrics"]["accuracy"]),
            "delta_accuracy": float(se["metrics"]["accuracy"] - sv["metrics"]["accuracy"]),
            "vanilla_macro_f1": float(sv["metrics"]["macro_f1"]),
            "ea_macro_f1": float(se["metrics"]["macro_f1"]),
            "delta_macro_f1": float(se["metrics"]["macro_f1"] - sv["metrics"]["macro_f1"]),
            "vanilla_ce": float(sv["metrics"]["ce"]),
            "ea_ce": float(se["metrics"]["ce"]),
            "delta_ce": float(se["metrics"]["ce"] - sv["metrics"]["ce"]),
            "vanilla_auc_gap_ce": float(dm_v["auc_gap"]),
            "ea_auc_gap_ce": float(dm_e["auc_gap"]),
            "vanilla_top_random_ce": float(dm_v["top_random_ratio"]),
            "ea_top_random_ce": float(dm_e["top_random_ratio"]),
            "vanilla_summary": str(sum_v),
            "ea_summary": str(sum_e),
        })

        v_cls = _class_map(sv["metrics"].get("per_class"))
        e_cls = _class_map(se["metrics"].get("per_class"))
        for cls in sorted(set(v_cls.keys()) | set(e_cls.keys())):
            vv = v_cls.get(cls, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
            ee = e_cls.get(cls, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
            bucket = per_class_bucket.setdefault(cls, {"delta_f1": [], "delta_precision": [], "delta_recall": [], "support": []})
            bucket["delta_f1"].append(float(ee["f1"] - vv["f1"]))
            bucket["delta_precision"].append(float(ee["precision"] - vv["precision"]))
            bucket["delta_recall"].append(float(ee["recall"] - vv["recall"]))
            bucket["support"].append(int(max(vv["support"], ee["support"])))

    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(cfg_path),
        "seeds": seeds,
        "k_list": k_list,
        "mask": args.mask,
        "random_trials": int(args.random_trials),
        "aggregate": {
            "n_runs": len(rows),
            "delta_accuracy_mean": _mean([r["delta_accuracy"] for r in rows]),
            "delta_accuracy_std": _std([r["delta_accuracy"] for r in rows]),
            "delta_macro_f1_mean": _mean([r["delta_macro_f1"] for r in rows]),
            "delta_macro_f1_std": _std([r["delta_macro_f1"] for r in rows]),
            "delta_ce_mean": _mean([r["delta_ce"] for r in rows]),
            "delta_ce_std": _std([r["delta_ce"] for r in rows]),
            "vanilla_auc_gap_ce_mean": _mean([r["vanilla_auc_gap_ce"] for r in rows]),
            "ea_auc_gap_ce_mean": _mean([r["ea_auc_gap_ce"] for r in rows]),
            "vanilla_top_random_ce_mean": _mean([r["vanilla_top_random_ce"] for r in rows]),
            "ea_top_random_ce_mean": _mean([r["ea_top_random_ce"] for r in rows]),
            "faithfulness_wins": int(sum(1 for r in rows if r["ea_auc_gap_ce"] > r["vanilla_auc_gap_ce"])),
            "faithfulness_losses": int(sum(1 for r in rows if r["ea_auc_gap_ce"] < r["vanilla_auc_gap_ce"])),
        },
        "per_class_delta": [
            {
                "class": int(cls),
                "delta_f1_mean": _mean(vals["delta_f1"]),
                "delta_precision_mean": _mean(vals["delta_precision"]),
                "delta_recall_mean": _mean(vals["delta_recall"]),
                "support_mean": _mean(vals["support"]),
            }
            for cls, vals in sorted(per_class_bucket.items())
        ],
        "runs": rows,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
