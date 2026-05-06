# Contributing

Thanks for contributing to EAAR.

## Workflow

1. Create a feature branch from `main`.
2. Keep changes focused and reproducible.
3. Run a minimal sanity check before commit.
4. Open a pull request with:
   - what changed,
   - why it changed,
   - which configs/results are affected.

## Code and Results Policy

- Prefer small, reviewable commits.
- Keep experiment configs in `configs/` (and ablation configs in `results/ablation/configs/`).
- Commit compact artifacts (`.md`, `.json`, `.csv`) needed for reproducibility.
- Do **not** commit heavy binaries (`.pt`, `.npy`, `.npz`, large raw dumps, temporary run folders).

## Style

- Follow existing project structure and naming.
- Preserve claim boundaries in docs: no over-claiming beyond reported evidence.
