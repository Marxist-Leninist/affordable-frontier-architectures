# v12: real-text gated recurrence with a non-neural control

2026-11-04. Bounded CPU research, not an AGI or frontier-quality claim.

## Results

All 12 neural runs completed (six pilots, six final). Three fresh final seeds:

| Model | Seed1 test bpb | Seed2 | Seed3 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| Modern Transformer | 2.728794 | 2.730003 | 2.718874 | **2.725890** | 0.006106 |
| Residual gated GRU | 2.491096 | 2.472278 | 2.486503 | **2.483292** | 0.009812 |
| Non-neural 5-gram | 2.335173 | deterministic | deterministic | **2.335173** | not applicable |

GRU mean cross-entropy is 8.90% below Transformer; the statistical model is another 5.96% below GRU. These are loss reductions, not accuracy or intelligence percentages. The 5-gram's serialized tables occupy 2,820,855 bytes versus roughly 0.53MB raw float32 neural weights. It wins prediction under this sample-limited protocol, not an equal-storage contest.

The n-gram fit took 2.60s. Neural mean training time per final run: Transformer 25.83s, GRU 50.93s. All pilot+final training totals: 156.33s versus 308.02s, excluding validation/test scoring, checkpoint I/O, audit, timing and development. The GRU quality advantage therefore does not establish equal-training-compute efficiency.

Median across the three seed-specific p50 timings:

| Prefix bytes | Transformer prefill / decode (ms) | GRU prefill / decode (ms) | Transformer KV / GRU state |
|---|---:|---:|---:|
| 128 | 1.421 / 0.506 | 3.976 / 0.280 | 131,072 / 512 bytes |
| 512 | 5.948 / 0.538 | 14.354 / 0.224 | 524,288 / 512 bytes |
| 2,048 | 41.093 / 0.733 | 54.856 / 0.221 | 2,097,152 / 512 bytes |

The GRU trades slower training/prefill for faster cached byte decoding and fixed small state. N-gram full-distribution prediction has p50~0.324ms with only four-byte context. This is **not a universal speed ranking**: kernels, cache storage, batch size, trained context, model quality and hardware differ from production. Raw sample p95 values remain in results.json.

## Why this experiment

v11 established that a properly trained Transformer solves synthetic multi-fact recall. v8's tiny real-text comparison used a one-block Transformer and vanilla tanh recurrence. This run asks a different question: does gated recurrence compete with a substantially stronger modern two-block Transformer on real text at the same small parameter and training-exposure budgets, and does a simple non-neural statistical model already explain the apparent gains?

## Models and fairness

- Transformer: 131,904 parameters, width 64, two blocks, four attention heads, rotary position embeddings, RMSNorm, residuals, SwiGLU FFNs, untied byte embedding/output matrix. Built-in scaled-dot-product causal attention; incremental KV caching implemented and parity-tested. FFN inner width 172.
- Gated recurrence: 132,160 parameters, width 64, two residual GRU blocks with RMSNorm/SwiGLU and the same embedding/output design. FFN inner width 128. Only 0.194% more parameters. This is a GRU experiment, not a Mamba/selective-SSM implementation.
- Non-neural control: suffix-frequency tables with recursive interpolation, alpha 10 fixed. Select maximum context 2/3/4 bytes on validation. Four context bytes is conventionally a 5-gram, not a 4-gram. Unigram smoothing adds one count to each possible byte. Missing contexts fall back to shorter suffixes.

Each neural run has 1,000 AdamW updates, batch 8, context 128 bytes: 1,024,000 training-target presentations. Each architecture gets three full-budget learning-rate pilots (.001/.003/.01, seed 101), then three fresh final initialization seeds (1/2/3). Same sampled windows across all runs. Validation chooses the final-step pilot rate before any neural test scoring. No early stopping or best-seed selection. Both neural pilots selected .01, the highest rate in the grid; neither optimum is established. Equal parameters/exposures/tuning budgets are **not equal FLOPs, memory, training maturity, or demonstrated hyperparameter optima**.

The n-gram uses the exact same sampled input/target windows and reset boundaries. Its three orders share a single maximum-order fit; deterministic count tables do not need repeated random-initialization runs. This is an equal-exposure comparison, not a parameter-memory comparison. Sparse count storage is materially larger than neural weights. Additional neural seeds and tuning runs cost real compute and are included in the logs, not hidden.

## Real data and quality metric

[Salesforce WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md), pinned revision b08601e04326c79dfdd32d625aee71d232d685c3. Dataset card licenses CC-BY-SA-3.0 and GFDL. Official train/validation/test splits; original rows joined by LF then UTF-8 encoded without filtering. The frozen source-Parquet and serialized-text hashes are in data_manifest.json. The source corpus is not redistributed.

Metrics are bits per byte, lower is better. They are not standard WikiText word perplexity or accuracy percentages. Evaluation scores all complete 128-byte windows: 1,147,904 validation and 1,291,904 test targets. The initial byte has no preceding target context; trailing 102/108 targets are excluded. State resets every 128 bytes for every model, including the n-gram. The sampled ledger covers 977,918 unique target positions, 8.9295% of training targets (the remaining presentations are overlaps). Training therefore exposes only a small fraction of the corpus, so this is sample-limited short-range language prediction, not language understanding, coding, reliable reasoning, or long-context retrieval.

## Latency and memory

Neural timing uses CPU float32, one PyTorch thread, batch 1, cached autoregressive forwarding of real test-text bytes. Prefill has 5 warmups and 10 measured iterations; decode 5 warmups and 50 measured tokens. Report p50/p95 for local observations, not service-level tail-latency guarantees. Contexts 128/512/2048 test implementation efficiency; **quality is only evaluated at 128, so long-context timing is not quality-equivalent serving evidence**. No sampling or tokenization cost included. Transformer KV appends use torch.cat rather than preallocated/paged storage, so growing-cache copy overhead is part of this reference implementation and may overstate its production disadvantage. Byte throughput is not interchangeable with provider subword-token throughput.

N-gram timings compute the full 256-byte next-token distribution from four bytes of context, over 500 samples. It does not need prompt prefill but cannot model farther context. Timings are observational CPU results, with possible neighboring process interference, not accelerator quotes or production cost measurements. GRU compressed state is fixed-size; Transformer KV grows with context. Count-table storage must also be counted.

## Economics stays separate

The user's target retail rates are $0.05/M uncached input, $0.001/M cached input, and $0.10/M output, treated as requested targets rather than independently verified market quotes. economics.py preserves a fixed **illustrative**, not user-measured, workload of 10M uncached input, 5M cached input, 2M visible output and 2M hidden output. The resulting target retail bill is $0.905. Hidden reasoning is not free.

A separate $500/month subscription scenario includes explicit assumed payment, support, training amortization, tools/storage/network and contingency allocations totaling $150. With zero tax in the scenario, 30% margin on net sales permits $200 serving spend. With 20% tax included in the gross price, it permits $141.67. These are hypothetical thresholds, not verified tax treatment or an available profitable service. The actual heavy-user workload, required quality, utilization, accelerator throughput, demand and all-in costs remain unmeasured. Positive margins cannot be established from these tiny models. No reduction in user allowance or required quality is credited.

## Reproduce

Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 2.3.5. PyArrow is only needed to extract the pinned dataset. Run from this directory:

    python prepare_data.py
    python -m unittest -v test_models test_auxiliary
    WIKITEXT_DATA=. python run.py
    WIKITEXT_DATA=. python ngram.py
    python economics.py

The runner normally reuses the adjacent v8 dataset cache but accepts WIKITEXT_DATA for standalone reproduction. It verifies exact split hashes, writes protocol.json before training, writes selection.json before final runs, saves every pilot and final checkpoint and exact-reload verifies each neural checkpoint. Frozen train_starts.npy can be regenerated deterministically by run.py. Complete code, protocols, numerical results, logs, tests and audit are published; binary checkpoint hashes are included and checkpoints are retained separately. Loading Python pickle is only appropriate for locally trusted n-gram artifacts.

## Architectural references

These establish the ingredients, not the quality of this small implementation:

- [RoFormer / rotary position embeddings](https://arxiv.org/abs/2104.09864)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078), gated recurrent units

## Interpretation and next gates

The non-neural control was added during neural pilot execution, before neural final-test results. Its candidate grid and smoothing were fixed in code before its validation and test scoring; it was not part of the initial neural source-hash list. The final artifact manifest covers its code. Treat comparisons as small exploratory research, not a preregistered cross-dataset confirmation.

A useful next gate would test whether statistical retrieval/interpolation adds value beyond a well-trained neural model, then evaluate exact factual recall and reasoning with external search or executable verifiers. A byte n-gram alone has no mechanism for open-ended planning or general reasoning. Neither its success nor any GRU advantage establishes an AGI architecture. More neural training, stronger recurrent/attention hybrids, multiple corpora and genuinely task-based quality gates are needed before scaling or spending decisions.

## Completed audit

All seven unit tests passed after the run. The independent review verified source/data/ledger hashes, gradient and streaming parity, counts, scoring boundaries and summary arithmetic. All six final neural checkpoints were independently reloaded and rescored on the complete test target set using a different batch size and accumulation method; maximum discrepancy was 1.014e-8 bits/byte. The statistical checkpoint reproduced its score exactly. See independent_audit.md and independent_audit_scores.json for details and limits. This is independent local checking, not external cross-corpus replication.
