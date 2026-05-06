#!/usr/bin/env python3
"""Train MLP classifier with optional EAAR regularization."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mlp_classifier import MLPClassifier
from src.regularizers.error_aware_attribution import (
    backward_with_eaar_projection,
    compute_ea_regularizer,
    error_importance,
    gradient_importance,
)
from src.utils.runtime_metadata import collect_runtime_metadata, sha256_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--mode", choices=["vanilla", "ea"], default="")
    return p.parse_args()


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _sha(arrays):
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(np.asarray(arr))
        h.update(str(a.shape).encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def _metrics(y_true, logits):
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
    pred = np.argmax(probs, axis=1)
    labels = np.unique(y_true)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=labels, zero_division=0
    )
    per_class = []
    for i, cls in enumerate(labels.tolist()):
        per_class.append({
            "class": int(cls),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "ce": float(log_loss(y_true, probs, labels=np.unique(y_true))),
        "per_class": per_class,
    }


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    config_sha256 = hashlib.sha256(cfg_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    effective_config_sha256 = hashlib.sha256(
        json.dumps(_jsonable(cfg), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    ds = cfg["dataset"]
    mcfg = cfg.get("model", {})
    ecfg = cfg.get("eaar", cfg.get("shap_reg", {}))

    seed = int(mcfg.get("seed", ds.get("random_state", 42)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mode = args.mode or ("ea" if bool(ecfg.get("enabled", True)) else "vanilla")
    model_mode = "ea_raw" if mode == "ea" else "vanilla"
    metrics_source = "ea_raw" if mode == "ea" else "vanilla"

    df = pd.read_csv(ds["train_data"])
    target_col = ds.get("target_column", "target")
    if "feature_columns" in ds and ds["feature_columns"]:
        feature_cols = list(ds["feature_columns"])
    else:
        feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy()
    y = np.asarray(y, dtype=np.int64)
    y = y - y.min()

    max_samples = int(ds.get("max_samples", 0) or 0)
    if max_samples > 0 and X.shape[0] > max_samples:
        idx = np.arange(X.shape[0])
        keep, _ = train_test_split(idx, train_size=max_samples, random_state=seed, stratify=y)
        keep = np.asarray(keep, dtype=int)
        X = X[keep]
        y = y[keep]

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(ds.get("test_size", 0.2)),
        random_state=int(ds.get("random_state", 42)),
        stratify=y,
    )
    split_hash = _sha([X_test, y_test])

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(np.max(y_train)) + 1
    model = MLPClassifier(X_train_sc.shape[1], num_classes, dropout=float(mcfg.get("dropout", 0.1))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(mcfg.get("lr", 1e-3)))
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = int(mcfg.get("epochs", 80))
    batch_size = int(mcfg.get("batch_size", 256))

    train_ds = TensorDataset(
        torch.tensor(X_train_sc, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    gamma = float(ecfg.get("gamma", 0.05))
    align_mode = str(ecfg.get("alignment_loss", ecfg.get("ea_alignment_loss", "cosine_mse")))
    align_alpha = float(ecfg.get("alignment_alpha", ecfg.get("ea_alignment_alpha", 0.5)))
    masking_mode = str(ecfg.get("masking_mode", ecfg.get("error_importance_mode", "permute")))
    ema_beta = float(ecfg.get("ema_beta", ecfg.get("error_importance_ema_beta", 0.9)))
    reg_warmup = float(ecfg.get("warmup_fraction", ecfg.get("ea_warmup_fraction", 0.25)))
    score_type = str(ecfg.get("score_type", "logprob"))
    grad_mode = str(ecfg.get("grad_combine_mode", "projected"))

    t0 = time.perf_counter()
    ea_ema = None
    prev_q = None
    last_grad_diag = {}

    for ep in range(epochs):
        model.train()
        warm = 1.0
        if mode == "ea":
            w_ep = int(epochs * reg_warmup)
            if w_ep > 0:
                warm = 0.0 if ep < w_ep else min(1.0, (ep - w_ep + 1) / max(1, w_ep))

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            main_loss = loss_fn(logits, yb)
            if mode == "ea":
                baseline = torch.mean(xb, dim=0)
                ea = compute_ea_regularizer(
                    model,
                    xb,
                    yb,
                    loss_fn=loss_fn,
                    masking_mode=masking_mode,
                    baseline_values=baseline,
                    ema_state=ea_ema,
                    ema_beta=ema_beta,
                    prev_q=prev_q,
                    align_mode=align_mode,
                    align_alpha=align_alpha,
                    task_type="classification",
                    score_type=score_type,
                )
                ea_ema = ea.ema_state.detach()
                prev_q = ea.q_err.detach()
                last_grad_diag = backward_with_eaar_projection(
                    model,
                    main_loss,
                    ea.ea_loss,
                    gamma=gamma * warm,
                    mode=grad_mode,
                )
            else:
                main_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    t1 = time.perf_counter()

    model.eval()
    with torch.no_grad():
        logits_test = model(torch.tensor(X_test_sc, dtype=torch.float32, device=device)).cpu().numpy()
    m = _metrics(y_test, logits_test)

    x_imp = torch.tensor(X_train_sc[: min(512, len(X_train_sc))], dtype=torch.float32, device=device)
    y_imp = torch.tensor(y_train[: min(512, len(y_train))], dtype=torch.long, device=device)
    p = gradient_importance(
        model,
        x_imp,
        y_batch=y_imp,
        task_type="classification",
        score_type=score_type,
    ).detach().cpu().numpy()
    p = np.maximum(p, 0.0)
    p = p / (np.sum(p) + 1e-12)

    q_vals = None
    if mode == "ea":
        _, q_ema, _ = error_importance(
            model,
            x_imp,
            y_imp,
            loss_fn=loss_fn,
            masking_mode=masking_mode,
            baseline_values=torch.mean(x_imp, dim=0),
            ema_state=None,
            ema_beta=ema_beta,
            prev_q=None,
            task_type="classification",
        )
        q_vals = q_ema.detach().cpu().numpy()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.tag:
        run_id = f"{run_id}_{args.tag}"
    out_dir = Path(cfg.get("output", {}).get("results_dir", "results/covtype_mlp_eaar"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"mlp_cls_state_{run_id}.pt"
    torch.save(model.state_dict(), model_path)
    logits_path = out_dir / f"logits_{run_id}.npy"
    np.save(logits_path, logits_test)

    fi = pd.DataFrame({"importance": p}, index=feature_cols)
    fi_path = out_dir / f"feature_importance_{run_id}.csv"
    fi.to_csv(fi_path)

    shap_path = None
    if q_vals is not None:
        fi_q = pd.DataFrame({"importance": q_vals}, index=feature_cols)
        shap_path = out_dir / f"feature_importance_shap_{run_id}.csv"
        fi_q.to_csv(shap_path)

    summary = {
        "timestamp": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "config_path": str(cfg_path),
        "config_sha256": config_sha256,
        "effective_config_sha256": effective_config_sha256,
        "git_commit": _git_commit(),
        "seed": seed,
        "dataset": Path(ds["train_data"]).stem,
        "split_hash": split_hash,
        "data_hash": sha256_file(ds["train_data"]),
        "model_mode": model_mode,
        "fallback_used": False,
        "unstable_prediction_flag": False,
        "metrics_source": metrics_source,
        "metrics": m,
        "runtime": collect_runtime_metadata(device),
        "model_state_path": str(model_path.resolve()),
        "saved_files": {
            "logits": logits_path.name,
            "feature_importance": fi_path.name,
            "shap": {"feature_importance_shap": shap_path.name} if shap_path else {},
        },
        "effective_eaar_config": {
            "enabled": mode == "ea",
            "task_type": "classification",
            "score_type": score_type,
            "masking_mode": masking_mode,
            "ema_beta": ema_beta,
            "alignment_loss": align_mode,
            "alignment_alpha": align_alpha,
            "warmup_fraction": reg_warmup,
            "gamma": gamma,
            "grad_combine_mode": grad_mode,
        },
        "training_diag": _jsonable(last_grad_diag),
        "timing": {
            "train_sec": float(t1 - t0),
            "total_sec": float(t1 - t0),
        },
    }
    summary_path = out_dir / f"training_summary_{run_id}.json"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_path)
    print(json.dumps({"mode": mode, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "ce": m["ce"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
