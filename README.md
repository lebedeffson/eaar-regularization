# EAAR Regularization made in Russia

Error-Aware Attribution Regularization (EAAR) for tabular supervised learning.

## What This Repository Contains

- ANFIS + EAAR core pipeline
- MLP / ResMLP portability runs
- Faithfulness evaluation (deletion AUC gap, top/random ratio)
- Ablations (negative controls, gamma/divergence sweeps)
- ROAR-lite and SAGE validation blocks
- Reproducibility artifacts (configs, manifests, result packs)

## Method (Core Form)

\[
L = L_{task} + \gamma D(p_\theta, q_{err})
\]

- `p_theta`: internal model feature-importance distribution
- `q_err`: target distribution from loss increase under feature masking

Implementation detail: `q_err` is built as a detached target (`no_grad` / no backprop through q_err construction).

## Claim Boundary (Important)

EAAR is positioned as:
- internal attribution faithfulness repair
- not an accuracy SOTA booster
- not a replacement for external permutation importance

## Project Layout

- `src/` — models, regularizer, utilities
- `scripts/` — training/evaluation/report pipelines
- `configs/` — experiment configs
- `results/` — compact text/CSV/JSON result artifacts

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main ANFIS EAAR run:

```bash
python train.py --config configs/config_sml2010_ea_minimal.yaml --tag sml_eaar_main
```

MLP regression portability:

```bash
python scripts/run_mlp_eaar_multiseed.py \
  --config configs/config_sml2010_mlp_ea.yaml \
  --seeds 42,43,44,45,46
```

Classification portability:

```bash
python scripts/run_mlp_classifier_eaar_multiseed.py \
  --config configs/config_covtype_mlp_eaar.yaml \
  --seeds 42,43,44,45,46
```

## Key Result Packs

- `results/q1_master_summary_20260503.md`
- `results/article_onefile_q1_20260503.md`
- `results/eaar_tisu_final_results_pack_20260504.md`
- `results/q1q2_final_pack_20260504.md`
- `results/results_manifest.json`

## Reproducibility Notes

- seeds fixed in multiseed manifests
- configs are versioned in `configs/` and `results/ablation/configs/`
- heavy binary artifacts are intentionally excluded from git history

## How to Cite

Citation metadata is provided in `CITATION.cff`.

## Current Status

Repository is prepared for manuscript writing and reviewer-facing validation.
For strict international Q1/Q2 positioning, stronger ROAR/KAR and deeper mechanism isolation may still be required.
