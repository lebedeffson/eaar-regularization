# SML2010: MLP Vanilla vs MLP+EAAR (10 seeds)

Source:
- `results/mlp_eaar_multiseed_config_sml2010_mlp_ea_sml_mlp_eaar10.json`
- `results/significance_sml2010_mlp_eaar_vs_vanilla_auc_gap_10seed.json`

## Aggregate

| Metric | Value |
|---|---:|
| n runs | 10 |
| ΔR² mean (EAAR - vanilla) | -0.000285 |
| ΔR² std | 0.003673 |
| R² wins/losses | 5 / 5 |
| Vanilla AUC gap mean | 17.3287 |
| EAAR AUC gap mean | 19.2265 |
| ΔAUC gap mean | +1.8977 |
| ΔAUC gap relative | +10.95% |
| Faithfulness wins/losses (AUC gap) | 6 / 4 |
| 95% CI ΔAUC gap | [-0.0609, 3.8563] |
| Wilcoxon p | 0.16015625 |
| t-test p | 0.09001990181314769 |
| Cohen d | 0.6005 |
| Cliff's delta | 0.3600 |

## Per-seed deltas

| Seed | ΔR² | Vanilla AUC gap | EAAR AUC gap | ΔAUC gap |
|---:|---:|---:|---:|---:|
| 42 | -0.005321 | 14.6623 | 18.1185 | +3.4562 |
| 43 | +0.000515 | 17.7631 | 19.0952 | +1.3322 |
| 44 | +0.003343 | 18.0061 | 21.6594 | +3.6533 |
| 45 | +0.000949 | 16.5267 | 16.3767 | -0.1500 |
| 46 | -0.000891 | 14.4187 | 15.4875 | +1.0688 |
| 47 | -0.000137 | 14.5719 | 13.8580 | -0.7139 |
| 48 | -0.000059 | 20.2223 | 18.9210 | -1.3012 |
| 49 | +0.003747 | 23.2853 | 21.6560 | -1.6292 |
| 50 | -0.007704 | 16.1175 | 24.3081 | +8.1906 |
| 51 | +0.002712 | 17.7134 | 22.7840 | +5.0706 |