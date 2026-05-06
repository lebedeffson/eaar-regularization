# Experimental Protocol and Hyperparameters

Config: `/home/lebedeffson/Code/bonner-shap-reconstruction/configs/config_sml2010_ea_minimal.yaml`

## Data split and leakage control

- Split in `train.py`: real data is split into train/val/test = `0.6 / 0.2 / 0.2` via `train_test_split` with fixed random seed.
- Current default is random split. For SML2010 strong-journal submission, add a time-block split sensitivity run.
- dataset random_state: `42`
- normalize_sum: `False`

## Preprocessing

- Missing/Inf handling: `np.nan_to_num` before training/eval.
- Scaling: as defined by dataset files/config pipeline (no extra hidden transform in report scripts).

## Model

- num_rules: `12`
- mf_class: `Gaussian`
- reg_lambda: `0.01`
- optimizer: `OriginalPSO`
- PSO epochs/pop: `80` / `60`

## EAAR settings

- gamma: `0.001`
- alignment loss D: `cosine_mse`
- alpha (mixed alignment): `0.5`
- error_importance_mode: `permute`
- error_importance_target: `train`
- ema beta (q_err): `0.9`
- ema beta (grad): `0.9`
- positive clipping in q_err: `True`
- target ablation mode: `none`

## Training schedule

- epochs: `30`
- batch_size: `64`
- learning rate: `8e-05`
- early_stopping_patience: `20`
- early_stopping_min_delta: `5e-05`

## Final policy

- mode: `quality_and_faithfulness`
- reject_unstable_predictions: `True`
- r2 tolerance gate (acceptance_min_delta_r2): `-0.001`
- faithfulness margin (auc_gap_margin): `0.02`
- min_ea_auc_gap: `0.0`
- min_top_random_ratio: `1.0`

## Faithfulness evaluation

- Main protocol: deletion top/random/bottom; AUC(top-bottom) gap.
- Masking modes: `permute|mean|noise` (paper main: `permute`).

## Statistical unit

- Unit of paired test: **seed-level paired comparison (default); ANFIS main significance in current artifact uses target-level paired units from 10 paired values**.
- In significance JSON, `n_pairs` corresponds to number of paired observations used in tests.

