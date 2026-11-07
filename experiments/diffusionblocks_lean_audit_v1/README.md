# DiffusionBlocks: reproducible Lean audit

This audit checks the algebra behind [DiffusionBlocks](https://arxiv.org/abs/2506.14202), distinguishes update-map equivalence from training-objective equivalence, and derives affine special-case closed forms.

**Status:** 20 named statements compiled with Lean 4.35.0, exit 0. The axiom audit contains only Lean's standard `propext`, `Classical.choice`, and `Quot.sound` (or no axioms). There are no `sorry` proofs, custom axioms, or native compiler proof shortcuts.

## Results

- Corrected diffusion and residual updates have an exact algebraic reparameterization.
- Their local and end-to-end training objectives are not generally equal; an exact rational counterexample is included.
- Time-homogeneous scalar affine dynamics have an all-depth geometric closed form, formally checked here.
- The paper's displayed positive-step-size Euler equations have a sign inconsistency. **The authors' public sampler already uses the correct sign.** Figure 3 also has inconsistent state indexing.
- A separately documented public ViT weighted-loss broadcasting concern is reproduced independently. It is not a finding about the unpublished language-model implementation.
- 100 time-varying affine matrix examples matched a product-sum formula to maximum absolute error `1.6653345369377348e-16`. These matrix checks are numerical, not formal proofs.

These results do **not** establish model quality, speed, convergence, floating-point implementation correctness, or a general closed-form training solution for nonlinear Transformers.

## Reproduce

Prerequisites: Lean **4.35.0**, Python **3.12**, and NumPy **2.3.5**. The Lean proof imports only `Init`; Mathlib is not needed. The `lean-toolchain` file pins the release for an existing elan installation. Alternatively, install the [official Lean release](https://github.com/leanprover/lean4/releases/tag/v4.35.0).

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
bash run_checks.sh
```

With a standalone Lean binary, use `LEAN=/absolute/path/to/lean bash run_checks.sh`. The runner does not install anything, download data, execute upstream project code, or alter recorded results.

## Files

- [Full mathematical report and limitations](REPORT.md)
- [Lean proof source](DiffusionEquivalence.lean)
- [Compiler and axiom audit](lean-check.log)
- [Verification provenance and source hash](verification.json)
- [Independent numerical checks](numerical_checks.py)
- [Recorded numerical results](numerical-results.json)
- [Reproduction runner](run_checks.sh)

## Sources

- [Paper, arXiv v4](https://arxiv.org/pdf/2506.14202v4), equations 1 and 3–6, Figure 3.
- [Public ViT implementation at pinned commit](https://github.com/SakanaAI/DiffusionBlocks/blob/ade0b08cf7a761b487948921fe2a14eb29039bd1/model.py).

The audit source and numerical checks were written independently. No model weights, dataset excerpts, upstream code copies, credentials, or personal data are included.
