# v6: incremental inference removes a misleading tiny-LM latency comparison

2026-11-03. Prior experiments v0–v5 reviewed; repository baseline 25d5d0ccf7d1bd60d010d30318f7a854ef1c0fa7. This is an actual local CPU neural-model execution experiment, not a price simulation. No rentals, package downloads, external inference, or production changes.

## Useful result

Caching preserves each model's probability distribution numerically while eliminating redundant work. Across three paired seeds, for 32-token sequences, recurrent cached execution takes about **0.0073 ms/token median** versus **0.0254 ms/token** for the attention model. Thus the recurrent model is faster with its natural state cache, whereas v5's cache-free benchmark made it look slower. This is not a controlled timing comparison against v5: v5 used sliding-32 windows, while v6 measures growing prefixes within explicit segments.

Within v6, both implementations process identical prefixes and tokens. Median whole-sequence speedups versus full-prefix recomputation at length 32 are **8.30–9.00× recurrent**, **1.78–2.31× attention**. These are implementation gains against an intentionally naive baseline, NOT novel architectural gains over normal cached serving, GPU results, or all-in cost reductions.

The same 8,192 held-out targets retain exactly the displayed bits-per-character values:

- Attention: 1.418654 / 1.523559 / 1.452886.
- Recurrent: 0.470650 / 0.458424 / 0.481183.
- Maximum per-probability cached/reference absolute difference: less than 2e-15.

No output, context, hidden reasoning, or work allowance was removed to obtain this within-model parity. Across-model quality is still unequal, and the synthetic template task cannot establish useful general-language equivalence.

## Reproduction and fixed protocol

Run in this directory with Python 3.10+ and NumPy already installed:

    python -m unittest -v
    python benchmark.py

`baseline.py` is the pinned v5 model/data/training code, with the training function returning trained weights rather than timing them. All six models are retrained deterministically with the same 400 steps, 8×32 batches, Adam settings, seed pairs, and 102,400 training characters per model. No hyperparameter search or adaptive stopping. `results.json` records source hashes, weight hashes, NumPy/platform, per-seed quality, timing and cache storage.

Quality evaluation uses fixed independent 32-character segments with explicit resets, exactly matching v5's held-out teacher-forced quality metric. Cached inference has the same positional indexing, causal context and reset semantics as the full-prefix reference. A capacity guard rejects accidental segment overflow; it does not silently truncate context.

Timing replays the identical first 32/128/512 held-out tokens, batch 1, rather than allowing different generated sequences to confound comparisons. Both modes compute the full vocabulary probabilities after every input token. Six trials alternate cached-first/recompute-first order, discarding the first trial; five trials supply p50/p95 per-token process timings and median whole-sequence timings. Cache allocation/reset occurs outside the timed region; per-token computation and cache writes are included. This is serial replay, not vectorized prefill or an optimized serving engine. No network, queueing or service SLO is measured. Percentiles pool prefix lengths and repetitions rather than measuring fixed-context tail latency. Shared CPU scheduling noise remains.

Longer 128/512 segments probe runtime scaling only. Models were trained at length 32, and no claim of useful long-context quality is made. The standard tests verify prefix parity, repeated reset, capacity rejection, state-storage arithmetic, and a short-window counterexample to treating truncation as equivalent. Independent audit evidence is included separately.

## Memory and decision

Resident NumPy cache buffers at length 512 are 131,200 bytes for attention, versus 272 bytes for recurrence. The attention count includes one unused 128-byte common state buffer; keys/values are 131,072 bytes. These counts exclude model weights, temporary activations, Python objects, allocator overhead and peak process RSS. They demonstrate linear KV versus fixed recurrent state in THESE toy implementations, not an apples-to-apples capability or production memory comparison.

Do not use v5's cache-free timing to choose an architecture. Preserve the paired semantic parity gate for subsequent implementations. Next useful research is a stronger normalized attention baseline and a fixed natural-language dataset with predeclared tuning budgets, then quality-matched comparison of genuinely optimized implementations. v6 intentionally isolates caching first; it does not complete those later gates.

The full intended 10M input + 2M visible + 2M hidden output monthly ledger, real latency/concurrency requirements, complete costs and positive-margin requirement below $500 remain unsolved. No measured frontier model, AGI claim, or business feasibility claim follows from these tiny synthetic results.
