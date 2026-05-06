# SML2010 Faithfulness Modes (time-block split, fast 5 seeds, save-model)

Multiseed:
- `results/multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock5_sm.json`
- unstable runs: `1` (seed `[44]`)

R² aggregation:

| Aggregation | n_runs | ΔR² mean | wins/losses | fallback_rate |
|---|---:|---:|---:|---:|
| all runs | 5 | -0.000244 | 0/5 | 0.00 |
| stable only | 4 | -0.000251 | 0/4 | 0.00 |

Faithfulness (permute deletion, random_trials=20):

| Mode | AUC top | AUC random | AUC bottom | AUC gap | Top/random |
|---|---:|---:|---:|---:|---:|
| Vanilla gradient | 0.0393 | 0.2076 | 0.4638 | -0.4245 | 0.1817 |
| Vanilla permutation | 0.6005 | 0.2076 | -0.0136 | +0.6141 | 2.9383 |
| EAAR internal (ea-only) | 0.4875 | 0.2076 | 0.0132 | +0.4744 | 2.3789 |
| EAAR final policy | 0.4154 | 0.2076 | 0.1051 | +0.3103 | 1.9706 |

Interpretation:
- Time-block split keeps the core pattern: vanilla gradient remains functionally wrong; EAAR internal remains correct.
- External permutation baseline is still stronger than EAAR internal in this protocol.
- Final policy is now weak-positive (above zero) but clearly below EAAR internal.