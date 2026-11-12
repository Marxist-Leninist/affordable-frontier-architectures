# Catalog continuation and reproduction checks: 12 November 2026

## Findings

The pinned ASI-Arch export contains **105 records**, with 105 unique indices, names and exact program hashes. The earlier gallery screen has a result for each, across **146 runs** including controls and confirmation. `CATALOG_COVERAGE.json` records every program and the seeds available. The historical "103 architectures" count has not been reconciled to a canonical subset. Distinct source text is not proof of distinct architectural ideas.

This is coverage of an **adapted, approximately 350k-parameter byte-level wrapper trained for 120 CPU-seconds**, not a full-scale test of all original architectures. No new training was run for this update.

## Reproduction defects fixed

1. The static audit now creates the output directory, so the documented fresh-checkout sequence works without a manual `mkdir`.
2. README generation accepts the actual verifier schema without inventing missing gallery-wide counter counts or historical reconstruction metadata.
3. A newly generated report can show failed tolerance checks honestly. It no longer claims exact equality merely because a report was supplied, or confuses six reloaded checkpoints with all 146 historical runs.
4. The verifier creates its output directory and records the largest observed loss difference.

Four regression tests pass. Every Python file compiles; `git diff --check` passes.

## Fresh input and checkpoint check

- All five public input files match the pinned hashes.
- All 105 programs pass the static source audit.
- Rebuilding the data matches the original training-text SHA-256, all 183 WikiText test windows, all 99 FineWeb-Edu windows, and both evaluation-array hashes.
- All six published checkpoints load. Each matches the recorded state hash, parameter count and state keys.
- **Zero of six passes the original strict numerical reproduction threshold on this different machine/software.** Largest aggregate held-out loss shift: **0.000136117226 nats per byte**. Largest individual-window shift: **0.003269 nats**. The original mean tolerance is 1e-6 and window tolerance 2e-6. See `VERIFY_geth.json`.
- This check used Intel Xeon Skylake, Python 3.12, PyTorch 2.10.0+cu128 in CPU mode and einops 0.8.1. Original training used an AMX-capable CPU and PyTorch 2.6.0+cpu. Different arithmetic is a plausible source of the drift; this check does not isolate which software or hardware change caused it.

The original within-machine results are preserved, not replaced. This is a portability result, not evidence for a new quality winner.

## Commands

From the repository root:

```sh
E=experiments/gallery_cpu120_20261110
bash "$E/fetch_inputs.sh" inputs
python "$E/code/static_audit.py" inputs/asiarch/106.json work/STATIC_AUDIT.json
python "$E/code/data_prep.py" inputs/hf experiments/asi_arch_psfr_cache_v1/v44_inputs work/data
python "$E/code/test_reproduction.py" -v
python "$E/code/catalog_coverage.py" inputs/asiarch/106.json "$E" work/CATALOG_COVERAGE.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  unshare -rn -- python -I "$E/code/verify_weights.py" "$E/checkpoints" \
  inputs/asiarch/106.json work/STATIC_AUDIT.json work/data work/VERIFY.json
python "$E/code/make_readme.py" "$E/METRICS.json" work/VERIFY.json \
  "$E/SMOKE.json" "$E/STATIC_AUDIT.json" "$E/code/README_static.md" work/README.md
```

These checks do not rerun training. Use the same arithmetic as the original run for an exact tolerance check.

## Next empirical phase

The meaningful next step is stronger validation, rather than another tiny all-catalog screen. The current larger Fusion/DLGM experiments and their ablations remain with their existing owner; this update neither duplicates their GPU work nor changes live training or serving.

Before a new comparison starts, freeze and publish:
- The fidelity-tested implementations, clearly separating original programs, CPU adaptations, low-rank variants and head-normalized variants.
- A larger model/longer training budget that the available hardware can actually support; no new paid compute is assumed.
- Identical training data, tokenization, seed set, optimizer-tuning allowance and stopping rule for every candidate; at least three independent seeds.
- Fresh held-out documents with hashes and a predeclared selection rule. Reused selection windows must not also serve as confirmatory evidence.
- Both quality at matched token/update budgets and quality at matched measured training compute, plus actual serving latency, throughput and peak memory.
- A shortlist including the prior screen's #259, #151 and #437 alongside #458 DLGM, #879 Fusion and the Transformer control; expansion to the remaining catalog follows controlled promotion gates.
- Per-seed and document-paired uncertainty, failed attempts, numerical portability limits, raw result files and checkpoint hashes.

This protocol is **not a claim that the next runs have started**. There is no verified spare GPU lane while the existing owner runs the Fusion/DLGM queue.

## Project links

- [GitHub gallery screen and repairs](https://github.com/Marxist-Leninist/affordable-frontier-architectures/pull/1)
- [Hugging Face research repository](https://huggingface.co/MarxistLeninist/affordable-frontier-architectures)
- [Current Fusion/DLGM comparison](https://zqx.lat/asi-arch/dlgm-vs-fusion/)
- [Original ASI-Arch export](https://github.com/GAIR-NLP/ASI-Arch/blob/3113c51b04a1d4a7520a3b0238c6a08d3194d58f/data/106.json)

The comparison's composite score is a project-specific decision aid, not an all-scale architecture proof.
