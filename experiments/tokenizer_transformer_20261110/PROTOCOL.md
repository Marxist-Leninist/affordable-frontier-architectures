# Actual tokenizer retraining comparison (2026-11-10)

Question: What happens when the tokenizer is changed in an otherwise identical small causal Transformer trained from scratch on English text? This is NOT a zero-shot swap of token IDs into pretrained weights.

## Frozen before main-run results
- GPT-2 tokenizer: openai-community/gpt2 @ 607a30d783dfa663caf39e06633721c8d4cfcd7e, 50,257 entries, MIT upstream.
- Qwen3 tokenizer: Qwen/Qwen3-235B-A22B @ 8efa61729e24bd65b1d152b5ab5409052aa80e65, 151,669 entries, Apache-2.0 upstream.
- Qwen3-0.6B tokenizer.json @ c1899de289a04d12100db370d81485cdf75e47ca is byte-identical to the 235B tokenizer (verified SHA-256). A tokenizer does not inherit its model's intelligence.
- Third control: GPT-2 vocabulary padded to 151,669 output/input entries. Extra IDs never occur as input/target. This equalizes parameter shapes and dense output dimensions with Qwen3, but deliberately charges GPT-2 for unused classes. Report separately, never as normal GPT-2 deployment.
- Same backbone: 2 pre-LayerNorm causal Transformer blocks, width64, 4 heads, FFN4x, learned positions, tied embedding/output, FP32, no dropout. Same trunk initialization per seed.
- AdamW lr0.001, betas(0.9,0.95), weight decay0.1, clipping1.0; minimize summed token NLL divided by raw UTF-8 bytes per batch. Same settings, not tuned individually.
- WikiText-2 raw official training/validation/test splits, revision b08601e04326c79dfdd32d625aee71d232d685c3. Card license CC-BY-SA-3.0 / GFDL; Wikipedia-derived.
- Segment each original row into nonoverlapping160-Unicode-character pieces. Keep pieces>=100 characters, tokenizable losslessly by both tokenizers into1–63 tokens. No text normalization.
- First10,000 eligible distinct training segments; first128 validation and128 test segments not exact duplicates of retained preceding splits. This is convenience/domain-limited sampling, not random Wikipedia sampling. Exact-segment duplicate removal does not establish absence of all near-duplicate contamination.
- Batch4, same per-seed shuffled raw samples. EOS is supplied as start context; all actual text tokens scored, no EOS target.
- Seeds1,2,3. Primary endpoint240 measured process CPU seconds of training per arm. Setup, tokenization, evaluation and checkpoint IO excluded from that training allowance and disclosed separately. CPU-seconds are a practical compute-budget proxy, NOT exact FLOP matching.
- Secondary endpoint after128 updates/512 identical raw text segments per seed: equal-data comparison, but different compute cost.
- Equal-data and equal-CPU cannot both hold when tokenizer lengths and vocabulary differ; report both rather than conceal the tradeoff.
- Metric: sum negative log probability of canonical tokenizations / ln(2) / UTF-8 bytes, on identical text. This is canonical-path bits/byte, not exact marginal string probability over all possible segmentations.
- Token perplexity and token accuracy are not directly comparable across tokenizers and are labelled accordingly.
- No benchmark questions from Artificial Analysis/LiveBench are used for this tiny language model: this test measures short-context held-out language modelling, not general intelligence.
- One CPU thread at nice19 on existing GETH, no GPU, new rentals or service changes. Disk start guard4GB; per-run600s and whole-run5400s wall caps. Nine serial training arms, total2160 budgeted training CPU seconds.
- Causal future-token perturbation assertion before each run; tokenizer exact round-trip assertions in preparation; full tokenizer and source data SHA-256 in manifest.
- Tiny undertrained models, a single corpus, three seeds and short contexts cannot establish frontier-scale benefit, no benefit, or an equivalence bound for large models. No universal claim of tokenizer negligibility will be made.
