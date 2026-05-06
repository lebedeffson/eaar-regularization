# Covertype (300k): MLP Vanilla vs MLP+EAAR (3 seeds)

Source:
- `results/mlp_classifier_eaar_multiseed_config_covtype_mlp_eaar_300k_covtype300k_cls_eaar3.json`
- `results/significance_covtype300k_mlp_eaar_vs_vanilla_auc_gap_ce.json`

## Aggregate

| Metric | Value |
|---|---:|
| n runs | 3 |
| ΔAccuracy mean (EAAR - vanilla) | -0.000211 |
| ΔMacro-F1 mean | -0.000717 |
| ΔCE mean | +0.002098 |
| Vanilla AUC gap (CE) mean | 2.7437 |
| EAAR AUC gap (CE) mean | 2.8378 |
| ΔAUC gap (CE) mean | +0.0941 |
| Faithfulness wins/losses | 3 / 0 |
| CI95(ΔAUC gap CE) | [0.0736, 0.1145] |

Interpretation:
- На более крупном подмножестве (300k) EAAR также стабильно улучшает CE-faithfulness.
- Accuracy/Macro-F1 практически на месте; CE слегка хуже в среднем.
- Это поддерживает переносимость эффекта на более крупный масштаб.

