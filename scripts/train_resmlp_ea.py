#!/usr/bin/env python3
"""Train ResMLP baseline with optional EA regularization (vanilla or ea_raw mode)."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.resmlp_regressor import ResMLPRegressor
from src.regularizers.error_aware_attribution import (
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


def _metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
    }


def _mask_columns(X, cols, mode, rng):
    X2 = np.asarray(X, dtype=float).copy()
    for j in cols:
        if mode == "permute":
            idx = rng.permutation(X2.shape[0])
            X2[:, j] = X2[idx, j]
        elif mode == "mean":
            X2[:, j] = float(np.mean(X2[:, j]))
        elif mode == "noise":
            std = float(np.std(X2[:, j]))
            X2[:, j] = X2[:, j] + rng.normal(0.0, 0.1 * std + 1e-8, size=X2.shape[0])
        else:
            raise ValueError(f"Unknown masking mode: {mode}")
    return X2


def _auc(y):
    y = np.asarray(y, dtype=float)
    if y.size == 1:
        return float(y[0])
    x = np.arange(1, y.size + 1, dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _deletion_gap_for_importance(model, X_sc, y, importance, mask_mode, k_list, random_trials, seed, device):
    model.eval()
    x_t = torch.tensor(X_sc, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred0 = model(x_t).detach().cpu().numpy()
    base_mse = float(np.mean((y - pred0) ** 2))

    imp = np.asarray(importance, dtype=float)
    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
    imp = np.maximum(imp, 0.0)
    if float(np.sum(imp)) <= 1e-12:
        imp = np.full_like(imp, 1.0 / max(1, imp.size))
    imp = imp / (float(np.sum(imp)) + 1e-12)
    order_desc = np.argsort(-imp)
    order_asc = np.argsort(imp)

    k_eff = [int(k) for k in k_list if 0 < int(k) <= X_sc.shape[1]]
    if not k_eff:
        k_eff = [1, min(2, X_sc.shape[1])]
    k_eff = sorted(set(k_eff))

    top_curve, rnd_curve, bot_curve = [], [], []
    for k in k_eff:
        rng = np.random.default_rng(seed + 97 * k)
        top_idx = order_desc[:k]
        bot_idx = order_asc[:k]
        X_top = _mask_columns(X_sc, top_idx, mask_mode, rng)
        X_bot = _mask_columns(X_sc, bot_idx, mask_mode, rng)
        with torch.no_grad():
            p_top = model(torch.tensor(X_top, dtype=torch.float32, device=device)).detach().cpu().numpy()
            p_bot = model(torch.tensor(X_bot, dtype=torch.float32, device=device)).detach().cpu().numpy()
        d_top = float(np.mean((y - p_top) ** 2) - base_mse)
        d_bot = float(np.mean((y - p_bot) ** 2) - base_mse)
        trials = []
        for t in range(max(1, int(random_trials))):
            rng_t = np.random.default_rng(seed + 1009 * (k + 1) + 7919 * (t + 1))
            cols = rng_t.choice(X_sc.shape[1], size=k, replace=False)
            X_r = _mask_columns(X_sc, cols, mask_mode, rng_t)
            with torch.no_grad():
                p_r = model(torch.tensor(X_r, dtype=torch.float32, device=device)).detach().cpu().numpy()
            trials.append(float(np.mean((y - p_r) ** 2) - base_mse))
        d_rnd = float(np.mean(trials))
        top_curve.append(d_top)
        bot_curve.append(d_bot)
        rnd_curve.append(d_rnd)

    auc_top = _auc(top_curve)
    auc_bot = _auc(bot_curve)
    auc_rnd = _auc(rnd_curve)
    return {
        "auc_top": float(auc_top),
        "auc_bottom": float(auc_bot),
        "auc_random": float(auc_rnd),
        "auc_gap": float(auc_top - auc_bot),
        "top_random_ratio": float(auc_top / max(auc_rnd, 1e-12)),
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
    ecfg = cfg.get("shap_reg", {})
    seed = int(mcfg.get("seed", ds.get("random_state", 42)))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mode = args.mode or ("ea" if bool(ecfg.get("enabled", True)) else "vanilla")
    model_mode = "ea_raw" if mode == "ea" else "vanilla"
    metrics_source = "shap" if mode == "ea" else "vanilla"

    df = pd.read_csv(ds["train_data"])
    X = df[ds["feature_columns"]].to_numpy(dtype=float)
    y = df[ds["target_columns"]].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=float(ds.get("test_size", 0.2)), random_state=int(ds.get("random_state", 42))
    )
    split_hash = _sha([X_test, y_test])

    sel_cfg = ecfg.get("faithfulness_selection", {}) if isinstance(ecfg, dict) else {}
    sel_enabled = bool(sel_cfg.get("enabled", False))
    val_frac = float(sel_cfg.get("val_fraction", 0.2))
    if sel_enabled and 0.0 < val_frac < 0.5 and len(X_train_all) > 32:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all,
            y_train_all,
            test_size=val_frac,
            random_state=seed + 17,
        )
    else:
        X_train, y_train = X_train_all, y_train_all
        X_val, y_val = None, None

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val) if X_val is not None else None
    X_test_sc = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResMLPRegressor(
        X_train_sc.shape[1],
        y_train.shape[1],
        hidden_dim=int(mcfg.get("hidden_dim", 128)),
        n_blocks=int(mcfg.get("n_blocks", 2)),
        dropout=float(mcfg.get("dropout", 0.1)),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(mcfg.get("lr", 1e-3)))
    loss_fn = torch.nn.MSELoss()
    epochs = int(mcfg.get("epochs", 160))
    batch_size = int(mcfg.get("batch_size", 128))

    train_ds = TensorDataset(
        torch.tensor(X_train_sc, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    t0 = time.perf_counter()
    ea_ema = None
    prev_q = None
    gamma = float(ecfg.get("gamma", 0.03))
    align_mode = str(ecfg.get("ea_alignment_loss", "cosine_mse"))
    align_alpha = float(ecfg.get("ea_alignment_alpha", 0.5))
    masking_mode = str(ecfg.get("error_importance_mode", "permute"))
    ema_beta = float(ecfg.get("error_importance_ema_beta", 0.9))
    reg_warmup = float(ecfg.get("ea_warmup_fraction", 0.25))
    importance_mode = str(ecfg.get("importance_mode", "task_loss"))

    best_state = None
    best_score = -1e18
    best_epoch = -1
    best_diag = {}

    eval_every = int(sel_cfg.get("eval_every_epochs", 10))
    sel_k_list = sel_cfg.get("k_list", [1, 2, 3, 4])
    if not isinstance(sel_k_list, (list, tuple)):
        sel_k_list = [1, 2, 3, 4]
    sel_trials = int(sel_cfg.get("random_trials", 10))
    sel_mask = str(sel_cfg.get("masking_mode", masking_mode))
    min_r2 = sel_cfg.get("min_r2", None)
    min_r2 = None if min_r2 is None else float(min_r2)
    alpha_top_random = float(sel_cfg.get("alpha_top_random", 0.0))
    lambda_r2_penalty = float(sel_cfg.get("lambda_r2_penalty", 1.0))

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
            opt.zero_grad()
            pred = model(xb)
            main_loss = loss_fn(pred, yb)
            loss = main_loss
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
                    importance_mode=importance_mode,
                )
                ea_ema = ea.ema_state.detach()
                prev_q = ea.q_err.detach()
                loss = main_loss + gamma * warm * ea.ea_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        if sel_enabled and X_val_sc is not None and (((ep + 1) % max(1, eval_every) == 0) or (ep + 1 == epochs)):
            model.eval()
            with torch.no_grad():
                p_val = model(torch.tensor(X_val_sc, dtype=torch.float32, device=device)).detach().cpu().numpy()
            r2_val = float(r2_score(y_val, p_val, multioutput="uniform_average"))
            x_imp = torch.tensor(X_val_sc[: min(256, len(X_val_sc))], dtype=torch.float32, device=device)
            y_imp = torch.tensor(y_val[: min(256, len(y_val))], dtype=torch.float32, device=device)
            p_imp = gradient_importance(
                model,
                x_imp,
                y_batch=y_imp,
                loss_fn=loss_fn,
                task_type="regression",
                importance_mode=importance_mode,
            ).detach().cpu().numpy()
            dm = _deletion_gap_for_importance(
                model=model,
                X_sc=np.asarray(X_val_sc, dtype=float),
                y=np.asarray(y_val, dtype=float),
                importance=p_imp,
                mask_mode=sel_mask,
                k_list=sel_k_list,
                random_trials=sel_trials,
                seed=seed + ep,
                device=device,
            )
            penalty = 0.0
            if min_r2 is not None and r2_val < min_r2:
                penalty = lambda_r2_penalty * (min_r2 - r2_val) ** 2
            score = dm["auc_gap"] + alpha_top_random * dm["top_random_ratio"] - penalty
            if score > best_score:
                best_score = float(score)
                best_epoch = int(ep + 1)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_diag = {
                    "score": float(score),
                    "auc_gap": float(dm["auc_gap"]),
                    "top_random_ratio": float(dm["top_random_ratio"]),
                    "r2_val": float(r2_val),
                    "penalty": float(penalty),
                }

    t1 = time.perf_counter()

    if sel_enabled and best_state is not None and bool(sel_cfg.get("restore_best", True)):
        model.load_state_dict(best_state, strict=True)

    model.eval()
    with torch.no_grad():
        pred_test = model(torch.tensor(X_test_sc, dtype=torch.float32, device=device)).cpu().numpy()
    m = _metrics(y_test, pred_test)

    x_imp = torch.tensor(X_train_sc[: min(512, len(X_train_sc))], dtype=torch.float32, device=device)
    y_imp_train = torch.tensor(y_train[: min(512, len(y_train))], dtype=torch.float32, device=device)
    p = gradient_importance(
        model,
        x_imp,
        y_batch=y_imp_train,
        loss_fn=loss_fn,
        task_type="regression",
        importance_mode=importance_mode,
    ).detach().cpu().numpy()
    p = np.maximum(p, 0.0)
    p = p / (np.sum(p) + 1e-12)

    q_vals = None
    if mode == "ea":
        y_imp = torch.tensor(y_train[: min(512, len(y_train))], dtype=torch.float32, device=device)
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
        )
        q_vals = q_ema.detach().cpu().numpy()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.tag:
        run_id = f"{run_id}_{args.tag}"
    out_dir = Path(cfg.get("output", {}).get("results_dir", "results/energy_resmlp_ea"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"resmlp_state_{run_id}.pt"
    torch.save(model.state_dict(), model_path)
    pred_path = out_dir / f"predictions_{run_id}.npy"
    np.save(pred_path, pred_test)

    fi = pd.DataFrame({"importance": p}, index=ds["feature_columns"])
    fi_path = out_dir / f"feature_importance_{run_id}.csv"
    fi.to_csv(fi_path)

    shap_path = None
    if q_vals is not None:
        fi_q = pd.DataFrame({"importance": q_vals}, index=ds["feature_columns"])
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
            "predictions": pred_path.name,
            "feature_importance": fi_path.name,
            "shap": {"feature_importance_shap": shap_path.name} if shap_path else {},
        },
        "effective_ea_config": {
            "enabled": mode == "ea",
            "error_importance_mode": masking_mode,
            "error_importance_ema_beta": ema_beta,
            "ea_alignment_loss": align_mode,
            "ea_alignment_alpha": align_alpha,
            "ea_warmup_fraction": reg_warmup,
            "importance_mode": importance_mode,
            "gamma": gamma,
        },
        "faithfulness_selection": {
            "enabled": sel_enabled,
            "best_epoch": best_epoch,
            "best_score": float(best_score) if np.isfinite(best_score) else None,
            "best_diag": best_diag,
            "eval_every_epochs": eval_every,
            "k_list": [int(k) for k in sel_k_list],
            "random_trials": sel_trials,
            "masking_mode": sel_mask,
            "min_r2": min_r2,
            "alpha_top_random": alpha_top_random,
            "lambda_r2_penalty": lambda_r2_penalty,
        },
        "model_config": {
            "hidden_dim": int(mcfg.get("hidden_dim", 128)),
            "n_blocks": int(mcfg.get("n_blocks", 2)),
            "dropout": float(mcfg.get("dropout", 0.1)),
        },
        "timing": {
            "vanilla_train_sec": float(t1 - t0),
            "ea_train_sec": float(t1 - t0) if mode == "ea" else 0.0,
            "total_sec": float(t1 - t0),
        },
    }
    summary_path = out_dir / f"training_summary_{run_id}.json"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_path)
    print(json.dumps({"mode": mode, "r2": m["r2"], "rmse": m["rmse"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
