#!/usr/bin/env python3
"""Generate reproducible experimental protocol/hyperparameter table (markdown)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--out", default="", help="Markdown output")
    p.add_argument("--stats-unit", default="seed-level paired comparison")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ds = cfg.get("dataset", {})
    model = cfg.get("model", {})
    shap = cfg.get("shap_reg", {})
    qp = shap.get("quality_policy", {})

    lines = [
        "# Experimental Protocol and Hyperparameters",
        "",
        f"Config: `{cfg_path}`",
        "",
        "## Data split and leakage control",
        "",
        "- Split in `train.py`: real data is split into train/val/test = `0.6 / 0.2 / 0.2` via `train_test_split` with fixed random seed.",
        "- Current default is random split. For SML2010 strong-journal submission, add a time-block split sensitivity run.",
        f"- dataset random_state: `{ds.get('random_state', 42)}`",
        f"- normalize_sum: `{bool(ds.get('normalize_sum', False))}`",
        "",
        "## Preprocessing",
        "",
        "- Missing/Inf handling: `np.nan_to_num` before training/eval.",
        "- Scaling: as defined by dataset files/config pipeline (no extra hidden transform in report scripts).",
        "",
        "## Model",
        "",
        f"- num_rules: `{model.get('num_rules')}`",
        f"- mf_class: `{model.get('mf_class')}`",
        f"- reg_lambda: `{model.get('reg_lambda')}`",
        f"- optimizer: `{model.get('optim')}`",
        f"- PSO epochs/pop: `{model.get('optim_params', {}).get('epoch')}` / `{model.get('optim_params', {}).get('pop_size')}`",
        "",
        "## EAAR settings",
        "",
        f"- gamma: `{shap.get('gamma')}`",
        f"- alignment loss D: `{shap.get('ea_alignment_loss', 'cosine_mse')}`",
        f"- alpha (mixed alignment): `{shap.get('ea_alignment_alpha', 0.5)}`",
        f"- error_importance_mode: `{shap.get('error_importance_mode', 'permute')}`",
        f"- error_importance_target: `{shap.get('error_importance_target', 'train')}`",
        f"- ema beta (q_err): `{shap.get('error_importance_ema_beta')}`",
        f"- ema beta (grad): `{shap.get('grad_importance_ema_beta')}`",
        f"- positive clipping in q_err: `{bool(shap.get('ea_positive_clipping', True))}`",
        f"- target ablation mode: `{shap.get('ea_target_ablation_mode', 'none')}`",
        "",
        "## Training schedule",
        "",
        f"- epochs: `{shap.get('epochs')}`",
        f"- batch_size: `{shap.get('batch_size')}`",
        f"- learning rate: `{shap.get('lr')}`",
        f"- early_stopping_patience: `{shap.get('early_stopping_patience')}`",
        f"- early_stopping_min_delta: `{shap.get('early_stopping_min_delta')}`",
        "",
        "## Final policy",
        "",
        f"- mode: `{qp.get('mode', 'quality_only')}`",
        f"- reject_unstable_predictions: `{qp.get('reject_unstable_predictions', True)}`",
        f"- r2 tolerance gate (acceptance_min_delta_r2): `{shap.get('acceptance_min_delta_r2')}`",
        f"- faithfulness margin (auc_gap_margin): `{qp.get('auc_gap_margin', 0.05)}`",
        f"- min_ea_auc_gap: `{qp.get('min_ea_auc_gap', 0.0)}`",
        f"- min_top_random_ratio: `{qp.get('min_top_random_ratio', 1.0)}`",
        "",
        "## Faithfulness evaluation",
        "",
        "- Main protocol: deletion top/random/bottom; AUC(top-bottom) gap.",
        "- Masking modes: `permute|mean|noise` (paper main: `permute`).",
        "",
        "## Statistical unit",
        "",
        f"- Unit of paired test: **{args.stats_unit}**.",
        "- In significance JSON, `n_pairs` corresponds to number of paired observations used in tests.",
        "",
    ]

    out = Path(args.out) if args.out else Path("results") / f"experimental_protocol_{cfg_path.stem}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

