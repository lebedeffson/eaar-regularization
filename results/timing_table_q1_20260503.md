# Timing Table (Q1 package, 2026-05-03)

Средние wall-time по актуальным прогонам.

| Task | Model | Dataset | Vanilla train sec | EAAR train sec | Overhead (EA/Vanilla) |
|---|---|---|---:|---:|---:|
| Regression | ANFIS | SML2010 (`sml_policy_refresh5`, 5 seeds) | 2.7114 | 13.4471 | 4.9600x |
| Regression | MLP | SML2010 (`sml_mlp_eaar5`, 5 seeds) | 3.8423 | 23.8376 | 6.2041x |
| Classification | MLP | Covertype-100k (`covtype_cls_eaar5`, 5 seeds) | 18.2091 | 112.7511 | 6.1920x |

Complexity note:

\[
O(E \cdot B \cdot d \cdot C_f)
\]

where \(E\) is EAAR epochs, \(B\) batches per epoch, \(d\) features, \(C_f\) forward-pass cost.

Environment (for reproducibility):
- device: `NVIDIA GeForce RTX 4060 Laptop GPU` (CUDA 12.8), CPU: `AMD Ryzen 7 7840HS w/ Radeon 780M Graphics`, RAM: `15.9 GB`, PyTorch: `2.11.0+cu128`, Python: `3.14.4`.
