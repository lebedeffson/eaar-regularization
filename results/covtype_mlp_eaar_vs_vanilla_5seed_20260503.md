# Covertype (100k): MLP Vanilla vs MLP+EAAR (5 seeds)

Source:
- `results/mlp_classifier_eaar_multiseed_config_covtype_mlp_eaar_covtype_cls_eaar5.json`
- `results/significance_covtype_mlp_eaar_vs_vanilla_auc_gap_ce_5seed.json`

## Aggregate

| Metric | Value |
|---|---:|
| n runs | 5 |
| ΔAccuracy mean (EAAR - vanilla) | +0.001270 |
| ΔMacro-F1 mean | -0.001682 |
| ΔCE mean | -0.001051 |
| Vanilla AUC gap (CE) mean | 2.3420 |
| EAAR AUC gap (CE) mean | 2.4138 |
| ΔAUC gap (CE) mean | +0.0718 |
| Faithfulness wins/losses | 5 / 0 |
| CI95(ΔAUC gap CE) | [0.0384, 0.1052] |

Interpretation:
- EAAR improves CE-based faithfulness consistently across all 5 seeds.
- Accuracy and CE are preserved (slightly better on average).
- Macro-F1 remains near-flat with small negative drift.

## Per-seed deltas

| Seed | ΔAccuracy | ΔMacro-F1 | Vanilla AUC gap (CE) | EAAR AUC gap (CE) | ΔAUC gap (CE) |
|---:|---:|---:|---:|---:|---:|
| 42 | +0.002600 | -0.003890 | 2.3701 | 2.5028 | +0.1328 |
| 43 | -0.000900 | -0.001829 | 2.3514 | 2.3821 | +0.0307 |
| 44 | +0.000250 | -0.009067 | 2.3168 | 2.3712 | +0.0544 |
| 45 | -0.000100 | -0.004648 | 2.3262 | 2.4032 | +0.0770 |
| 46 | +0.004500 | +0.011024 | 2.3454 | 2.4095 | +0.0642 |

