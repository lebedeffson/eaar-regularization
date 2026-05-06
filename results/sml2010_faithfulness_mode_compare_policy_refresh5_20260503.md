# SML2010 Faithfulness Modes (policy refresh, fast 5 seeds)

Multiseed:
- `results/multiseed_config_sml2010_ea_minimal_sml_policy_refresh5.json`
- unstable runs: `2` (seeds `[42, 43]`)

R² aggregation (from stability sensitivity):

| Aggregation | n_runs | ΔR² mean | wins/losses | fallback_rate |
|---|---:|---:|---:|---:|
| all runs | 5 | -0.006848 | 3/2 | 0.20 |
| stable only | 3 | +0.000003 | 2/1 | 0.33 |

Faithfulness (permute deletion, random_trials=20):

| Mode | AUC top | AUC random | AUC bottom | AUC gap | Top/random |
|---|---:|---:|---:|---:|---:|
| Vanilla gradient | 0.0842 | 0.2236 | 0.4836 | -0.3994 | 0.3747 |
| Vanilla permutation | 0.5611 | 0.2236 | 0.0064 | +0.5547 | 2.5139 |
| EAAR internal (ea-only) | 0.4862 | 0.2236 | 0.0193 | +0.4669 | 2.1299 |
| EAAR final policy | 0.2934 | 0.2236 | 0.2772 | +0.0161 | 1.1585 |

Interpretation:
- Faithfulness-aware policy removed the hard negative final gap (was negative before).
- `final policy` is now slightly positive, but still much weaker than `ea-only`.
- Main scientific signal remains `EAAR internal`; `final policy` is practical and stricter.

