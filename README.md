# EAAR Regularization made in Russia

Error-Aware Attribution Regularization (EAAR) for supervised tabular models.

This repository contains the experimental and reporting pipeline for aligning **internal model importance** with **error increase under feature masking**.

---

## 1) Motivation

Many models predict well but produce internal feature rankings that are functionally inconsistent: top-ranked features do not always cause the largest error increase when masked.

EAAR addresses this by adding an attribution-alignment regularizer during training.

---

## 2) Method

### Core objective

\[
\mathcal{L}(\theta) = \mathcal{L}_{task}(\theta) + \gamma \, D\!\left(p_\theta, q_{err}\right)
\]

Where:
- \(p_\theta\): internal feature-importance distribution produced by the model,
- \(q_{err}\): target importance distribution derived from per-feature masking error increase,
- \(D(\cdot,\cdot)\): divergence/alignment term (configurable),
- \(\gamma\): regularization strength.

### GitHub-safe plain-text form

```text
L_total = L_task + gamma * D(p_theta, q_err)
```

### Target construction

For feature \(j\):

\[
\eta_j = \left[\mathcal{L}_{task}\!\left(y, f_\theta(x^{(-j)})\right)-\mathcal{L}_{task}\!\left(y, f_\theta(x)\right)\right]_+, 
\quad
q_{err,j} = \frac{\eta_j}{\sum_k \eta_k + \varepsilon}
\]

Implementation note: in this repo `q_err` is constructed as a **detached target** (`no_grad` through target construction).

---

## 3) Claim Boundary

This repository supports the following claim scope:

1. EAAR improves internal attribution faithfulness in tested ANFIS/MLP settings.
2. EAAR is **not** presented as an accuracy-SOTA booster.
3. EAAR does **not** replace external post-hoc baselines (permutation, SAGE, ROAR/KAR-style checks).

---

## 4) Repository Structure

- `src/` — model and regularization internals
- `scripts/` — experiment orchestration and report builders
- `configs/` — run configurations
- `data/` — dataset files/prepared tables
- `results/` — generated summaries (`.md/.json/.csv`)

---

## 5) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 6) Typical Workflows

### 6.1 ANFIS + EAAR (main)

```bash
python train.py \
  --config configs/config_sml2010_ea_minimal.yaml \
  --tag sml2010_eaar_main
```

### 6.2 MLP regression portability

```bash
python scripts/run_mlp_eaar_multiseed.py \
  --config configs/config_sml2010_mlp_ea.yaml \
  --seeds 42,43,44,45,46
```

### 6.3 MLP classification portability

```bash
python scripts/run_mlp_classifier_eaar_multiseed.py \
  --config configs/config_covtype_mlp_eaar.yaml \
  --seeds 42,43,44,45,46
```

### 6.4 Explainability aggregation

```bash
python scripts/report_explainability_multiseed.py \
  --multiseed results/<multiseed_json>.json \
  --out results/<explainability_report>.json
```

### 6.5 Statistical report

```bash
python scripts/report_significance.py \
  --a results/<run_a>.json \
  --b results/<run_b>.json \
  --metric auc_gap
```

---

## 7) Key Result Packs

- `results/q1_master_summary_20260503.md`
- `results/article_onefile_q1_20260503.md`
- `results/eaar_tisu_final_results_pack_20260504.md`
- `results/q1q2_final_pack_20260504.md`
- `results/results_manifest.json`

---

## 8) Reproducibility

- fixed seeds in multiseed configs/manifests,
- versioned configs in `configs/`,
- explicit report scripts for significance, protocol, and stability.

Heavy artifacts are excluded from git (`.pt`, `.npy`, `.png`, `.pdf`, archives).

---

## 9) Limitations

Current repository outputs include explicit boundaries:

- final-policy may underperform internal-only mode on faithfulness,
- mechanism isolation (`q_err` vs sparsity/compactness) can remain partial in some ablations,
- stronger ROAR/KAR and broader SAGE coverage are still desirable for strict international Q1/Q2 claims.

---

## 10) Citation

Use `CITATION.cff` for citation metadata.

Repository URL:
- https://github.com/lebedeffson/eaar-regularization
