# Energy: MLP Vanilla vs MLP+EAAR (5 seeds)

Source:
- `results/mlp_eaar_multiseed_config_energy_mlp_ea_energy_mlp_eaar5.json`
- `results/significance_energy_mlp_eaar_vs_vanilla_auc_gap_5seed.json`

## Aggregate

| Metric | Value |
|---|---:|
| n runs | 5 |
| ΔR² mean (EAAR - vanilla) | +0.000211 |
| ΔR² std | 0.004749 |
| R² wins/losses | 3 / 2 |
| Vanilla AUC gap mean | 192.7181 |
| EAAR AUC gap mean | 186.9414 |
| ΔAUC gap mean | -5.7767 |
| ΔAUC gap relative | -3.00% |
| Faithfulness wins/losses (AUC gap) | 3 / 2 |
| 95% CI ΔAUC gap | [-19.9073, 8.3540] |
| Wilcoxon p | 0.8125 |
| Cohen d | -0.3583 |

## Per-seed deltas

| Seed | ΔR² | Vanilla AUC gap | EAAR AUC gap | ΔAUC gap |
|---:|---:|---:|---:|---:|
| 42 | +0.001871 | 164.4769 | 157.5582 | -6.9187 |
| 43 | -0.007230 | 243.4653 | 210.2831 | -33.1823 |
| 44 | -0.000797 | 179.3637 | 182.4389 | +3.0752 |
| 45 | +0.001603 | 183.7919 | 190.5886 | +6.7967 |
| 46 | +0.005609 | 192.4924 | 193.8381 | +1.3457 |