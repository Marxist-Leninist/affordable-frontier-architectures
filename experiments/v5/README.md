# v5: actual tiny neural training, not another cost simulation

2026-11-03 follow-up to v4, repository baseline f2daa73351f5c9bfa6b8fd1f5061e4147f95acb0. This experiment trains neural next-character models from scratch on available local NumPy CPU compute. No models, packages or datasets were downloaded; no paid resources or production processes changed. Remote resource inspection failed before execution with an approval-review error, so this experiment uses the available local environment only.

## What actually ran

- Single-head causal self-attention with sinusoidal positions, residual projection and tanh feed-forward layer: **2848 parameters**, width16.
- Vanilla tanh recurrent neural LM: **2856 parameters**, width34. Parameter difference0.28%.
- Both train all parameters with manual backpropagation, Adam LR0.003, global gradient clipping1, float64,400updates ×8sequences ×32characters = **102400 training characters per seed**. Three paired seeds; identical sampled minibatch positions within each pair. No adaptive stopping or best-seed selection.
- Corpus:1500 programmatically generated English-template sentences.300 validation sentences are disjoint by exact sentence identity, with familiar grammar/vocabulary and unseen combinations. This is deliberately **synthetic, low-entropy text**, not natural-language benchmark capability. SHA256 hashes are in results.json; deterministic generator is included.
- Heldout scoring uses the same fixed8192character targets (32batches ×8 ×32), context resets every32characters, teacher-forced next-character cross-entropy. No long-context or reasoning benchmark.
- Timing measures actual one-thread NumPy execution. Decoding is batch1 greedy with full32-character sliding-window recomputation for BOTH models,192generated characters, first2timing samples omitted. It is not optimized incremental KV/state-cached decoding. p50/p95 are process timings, not service tail latency, and OS noise remains substantial.

## Results

Attention LM validation bpc by seed: **1.41865,1.52356,1.45289**.
Recurrent LM: **0.47065,0.45842,0.48118**. Lower is better.

Attention's measured decode p50 is about0.067–0.068ms/token, versus0.098–0.105ms for recurrent full-window recomputation. Its quality is markedly worse, so the lower latency is **not** a quality-preserving economics improvement. Generated examples are included in results.json and show attention repetition versus recurrent template fluency. Such fluent template completions are not general language skill.

Training took roughly0.27–0.34seconds per tiny run on this CPU environment; full source/results preserve exact timings. These numbers must not be extrapolated to GPU,3T parameters or user subscription economics. No hardware rental price or all-in cost saving is inferred from shared scratch CPU time.

## Fairness limits and correctness checks

This is a deliberately simplified Transformer-like attention block: no layer normalization, multi-head attention, gated FFN, modern positional optimization or tuning. The recurrent baseline is also a basic tanh cell, NOT Mamba. Equal parameter counts and token budgets do not imply equal FLOPs or equal tuning effort; one shared learning rate was not optimized per model. Width differs to match counts, which can affect capacity. The synthetic task favors a compact local state and cannot select a frontier architecture. Cache-free timing penalizes the recurrent model relative to its natural incremental execution, while also denying attention KV reuse.

Five tests pass: finite-difference gradients for every parameter group in both models (maximum errors <1e-7), causal masking, normalized output probabilities, disjoint sentence splits, approximate parameter match. Independent review checks the implementation and arithmetic. Finite differences sample tensor entries, not exhaustive formal verification. All parameters are float64, not binary/quantized AGILLM, and no AGILLM routing assumption enters this test.

## Decision and next experiment

Do not select the faster attention implementation as the cheaper equivalent model: it fails the quality gate here. Next useful bounded work is an optimized incremental-cache implementation with cached/uncached output parity and a stronger normalized causal-attention baseline, followed by a fixed natural-language dataset and a predeclared hyperparameter budget. Only then compare matched quality and actual end-to-end latency. Larger-model quality, hidden reasoning allowance and full-month workload are not reduced or assumed solved.

The v4 full-cost ledger remains unchanged: input/visible/hidden work, pooled versus dedicated bill, weights/scales/state, caching, concurrency/idle, bandwidth, payment/tax/support and training amortization are still required before a $500net-revenue margin claim. This toy result provides none of those frontier cost measurements and makes no £200/£20 or AGI claim.

Reproduce: Python3.10+, NumPy (tested version in results.json), `python -m unittest -v` then `python benchmark.py`. Thread environment is fixed before importing NumPy; run in a fresh process. No shell install scripts or Actions workflows.

Primary architecture reference: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), accessed2026-11-03. The published Transformer includes components deliberately absent from this minimal probe, so this result is not a replication of that architecture or its reported capability.
