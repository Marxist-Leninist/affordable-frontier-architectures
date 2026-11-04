# v13: do statistical and neural language models complement each other?

2026-11-04. Frozen-checkpoint hybrid prediction experiment. No paid compute, production changes, AGI claim or frontier-quality claim.

## Results

| Model | Mean test bits/byte | Seed sample SD |
|---|---:|---:|
| Transformer alone | 2.725890 | 0.006106 |
| GRU alone | 2.483292 | 0.009812 |
| Statistical 5-gram alone | 2.335173 | deterministic |
| Transformer + statistical mixture | **2.271157** | 0.000790 |
| Gru + statistical mixture | **2.242399** | 0.001524 |

Transformer hybrid lowers loss by 2.74% versus statistical-only; GRU hybrid by 3.97%. These are cross-entropy reductions, not intelligence or accuracy gains. Both improve on statistics alone in all three final seeds. The combined GRU is best here, but requires both systems. The neural signal remains useful even when its standalone loss is worse.

Transformer seed hybrid bpb: 2.270913, 2.272040, 2.270519. Median across seed-specific p50 component times: neural 0.537ms, ngram 0.374ms, hybrid 0.944ms.

Gru seed hybrid bpb: 2.243237, 2.240641, 2.243320. Median across seed-specific p50 component times: neural 0.334ms, ngram 0.464ms, hybrid 0.838ms.

The v13 script took 85.29 seconds wall time, including rebuilding counts, validation/test scoring and timing but excluding original v12 training, development and independent audit. No incremental paid compute was purchased; CPU time is still a real resource, and its dollar allocation is unknown.

## Question and locked protocol

v12's non-neural byte 5-gram beat both small neural baselines. v13 asks whether neural predictions nevertheless add information beyond it. This is a new complementarity test, not another retraining of v12 or evidence that a combined system wins at equal total compute/storage.

Use v12's exact frozen two-block modern Transformer (RoPE, RMSNorm, SwiGLU, 131,904 parameters), residual gated GRU (132,160 parameters), and statistical counts. The neural models each had 1,024,000 sampled training-target presentations, equal parameter budgets within 0.2%, equal learning-rate search grids, and identical windows. They remain sample-limited small models, not optimally trained frontier baselines. Both learning-rate optima were at v12's grid boundary. The statistical component uses the same sampled windows, not extra corpus data.

For each architecture, mix full probability distributions as `(1-w)*neural + w*ngram`. Test the same 11 weights 0,.1,...,1 on the complete validation split using only the original pilot initialization (101). Save selection.json before evaluating any final seed. Selected statistical weight: Transformer **0.9**, GRU **0.7**. Apply that single frozen weight to all three existing final initializations (1,2,3). No new neural fitting, test-set adaptation, count updates, seed selection or per-final-seed tuning. This protocol is exploratory and uses a dataset already inspected in v12; it is not an untouched-dataset confirmation.

## Dataset and scope

[Salesforce WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md), revision b08601e04326c79dfdd32d625aee71d232d685c3, card licenses CC-BY-SA-3.0 and GFDL. Original rows joined by LF then UTF-8 encoded; official splits; source and serialized hashes checked. Corpus is not redistributed. Full complete 128-byte windows: 1,147,904 validation /1,291,904 test targets, trailing 102/108 excluded. All state resets at every window, including the n-gram. The score is bits per byte, lower better, not standard word perplexity or an accuracy percentage. Same corpus and test positions across seeds; seed spread does not measure cross-corpus generalization.

## Costs and fairness

Both hybrids receive identical mixture-tuning budgets and unchanged component checkpoints, but additive components have additive costs. Original v12 pilot+final neural training totals were 156.33s Transformer and 308.02s GRU, plus a shared statistical fit of 2.60s. v13 records rebuild, scoring and total script time separately in results.json. Historical training is not free because it is reused. No equal-FLOP or equal-wall-time superiority is claimed.

Transformer+statistics serializes to 3,355,007 bytes; GRU+statistics 3,356,783, versus 2,820,855 statistical-only and ~0.53MB neural checkpoints. Serialized bytes are not live memory: Python count-table objects, allocator/interpreter overhead, temporary activations and workspaces also count. The audit measures **32,205,521 bytes (~30.7 MiB)** for the identity-deduplicated Python table object graph, before interpreter/allocator overhead and transient allocations, versus just 2.82MB serialized. See audit_memory.json. At 128 input bytes the Transformer KV tensors occupy 131,072 bytes, GRU recurrent state 512; the statistical suffix adds four bytes of logical history. This does not include Python container overhead. Both hybrids still keep all neural weights and statistical tables.

Latency is a single-thread CPU float32 cached next-byte distribution microbenchmark, fixed 128-byte prefix, 10 warmups and 50 samples, not production throughput. Hybrid samples sum separately timed neural forwarding/softmax, statistical full distribution and mixing. Prefill is excluded. They are not directly timed end-to-end service latency or service p95. One prefix only, component order fixed, nearby CPU activity may interfere. This asks the modest question whether better prediction also costs extra work; it does. No extrapolation to accelerator subword tokens, long-context quality or subscription profit is justified.

## Pricing and sustainable $500 goal

See [verified primary-source price references and caveats](pricing.md), and economics.py/economics.json. Official Muse Spark 1.3 Contributor rates validate the user's half-price targets: $0.05/M uncached input, $0.001/M cached input, $0.10/M output. Contributor's data-training permission and lack of max reasoning mean it is not automatically equivalent to private, maximum-reasoning service. Retail rates do not disclose inference cost.

The fixed illustrative workload is unchanged: 10M uncached input, 5M cached input, 2M visible output and 2M hidden output. Target bill $0.905. It is not an observed heavy-user workload. A separate $500 gross monthly subscription with $150 assumed other costs and 30% margin on net sales leaves $200 serving spend at zero tax, or $141.67 with illustrative 20% included tax. Training amortization, failed experiments, support, tools, idle capacity and other operating expenses cannot disappear. Nothing here establishes a sustainable frontier-quality $500 service; no quality or usage reduction is credited.

## Reproduction

Use Python3.12, PyTorch2.6.0+cpu, NumPy2.3.5; PyArrow19.0.1 for extraction. From a repository checkout, first reproduce v12 according to its README to create neural checkpoints and train_starts.npy. Binary neural checkpoints are not redistributed in this commit; exact input hashes are in v12/checkpoint_manifest.json. v13 verifies them before loading with `weights_only=True`. It rebuilds n-gram counts from the exact ledger instead of loading pickle.

From experiments/v13:

    python prepare_data.py
    V12_DIR=../v12 WIKITEXT_DATA=. python run.py
    V12_DIR=../v12 WIKITEXT_DATA=. python -m unittest -v test_audit
    python economics.py

Local run reused verified cached data and checkpoints. Dependency path needed in this container was PYTHONPATH=/tmp/frontier-v10-packages; that is an environment detail, not a portable dependency requirement. All code and numerical artifacts are included, not dataset text. Check hashes before reproduction. Source changes after results require rerunning affected checks.

## Next gate

The hypothesis to carry forward is neural/statistical complementarity, not that n-grams understand language. Validate on a second licensed corpus with a locked protocol and better-trained Transformer under matched total fitting/inference/storage budgets, then test factual retrieval, reasoning and executable verification. Tiny short-range byte prediction cannot establish AGI, long-context memory, task reliability or frontier economics.
