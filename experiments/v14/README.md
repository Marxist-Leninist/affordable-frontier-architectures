# v14: training adequacy and a matched-quality cost gate

2026-11-04. Bounded local CPU research on pinned Hugging Face WikiText-2. No paid compute, production change, external data sharing, AGI claim, or demonstrated frontier-quality system.

## Results: stronger baseline, no matched-quality pass

| Model | Mean test bits/byte | Seed sample SD | Median local end-to-end probe |
|---|---:|---:|---:|
| Modern Transformer | 2.359696 | 0.004290 | 21.589 ms |
| Gated GRU | 2.238784 | 0.010037 | 14.251 ms |
| Statistical five-gram | 2.226092 | deterministic | 13.750 ms |

The Transformer improves substantially relative to v12's 2.725890 bpb, while the GRU improves from 2.483292. The neural gap shrinks from about .243 to .121 bpb, warning against treating a short-trained baseline as an architectural ceiling. Cross-version changes include the ledger and schedule, so this is not a clean causal estimate of training duration alone. Within each v14 final run, validation loss continues improving from 1k to 2k to 4k. Both selected learning rates are .01, now interior to the .003/.01/.03 grid. This still does not prove convergence.

The statistical model remains best on mean test loss, but is only .01269 bpb ahead of GRU; initialization spread is not a confidence interval for domain generalization. No neural/statistical mixture was newly fitted here.

**None of the three prespecified neural matched-quality gates passes.** At validation <=2.5, Transformer first crosses at 4k and GRU at 2k in all seeds, but GRU test loss is .03533–.05986 bpb worse, outside the .02 screen. At <=2.35, both cross at 4k, but GRU is .10827–.13494 bpb better, also not matched. Transformer never reaches <=2.25. These are inconclusive comparable-quality samples, not proof that comparable-quality operation is impossible. Do not call the observed lower GRU latency an equal-quality speedup.

The final checkpoint's median seed prefill p50 is 1.690 ms Transformer versus 4.167 ms GRU; cached-update mean-per-sequence p50 is .625 versus .315 ms. The mixed 128-byte-prefix plus 32-update probe takes 21.589 versus 14.251 ms, with statistical-only 13.750 ms. Training is the opposite tradeoff: final runs average 114.93 s Transformer versus 241.22 s GRU. All pilots and final runs total 684.05 s Transformer and 1,421.11 s GRU, plus 12.08 s shared statistical fitting. Full script wall time is 2540.24 s (42.34 minutes), excluding development and independent audits. Timing units and exclusions are detailed below.

The statistical table serializes to 4,506,831 bytes but its measured deduplicated Python object graph occupies **54,023,171 bytes (51.52 MiB)** before interpreter/allocator overhead. Neural checkpoints are about .535 MB each. Final 160-byte neural cache tensors occupy 163,840 bytes Transformer versus 512 bytes GRU; these are only state tensors, not total runtime RAM. No architecture clears a real dollar-cost or frontier-quality gate.


## Why this experiment

v12/v13 left an important confound: both tiny neural models' preferred learning rates were at the top of the tested grid, and the statistical model's advantage could reflect neural undertraining. v14 tests a new optimization/quality-cost question rather than repeating the frozen-mixture experiment. It increases per-run target presentations fourfold and widens the upper learning-rate bound, then asks whether fixed validation quality gates permit a comparable-quality latency comparison.

This improves the baseline training protocol, but is **not a certified strong or converged Transformer**. It remains a two-block 131,904-parameter RoPE/RMSNorm/SwiGLU byte Transformer, compared with a 132,160-parameter residual gated GRU and a non-neural interpolated byte n-gram. A well-trained large Transformer and frontier task reference remain missing. No architecture is predetermined to win.

## Locked protocol and data

Use [Salesforce WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md), revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The card lists CC-BY-SA-3.0 and GFDL. Original rows are joined with LF and UTF-8 encoded without normalization; original official splits and content hashes are verified. Dataset text is not redistributed. This is the **same corpus used in v12/v13, not a second-corpus replication**; previous test inspection means it is exploratory continuation.

Each neural run has 4,000 AdamW steps, batch 8, 128-byte windows, hence 4,096,000 sampled training-target presentations, not distinct bytes. Both families see exactly the same reproducible ledger, including overlaps/repeats. Learning rates .003/.01/.03 are compared using pilot seed 101 at final-step full-validation bits/byte. One LR per family is saved before any final test evaluation. Final seeds 1/2/3 are newly trained. Three checkpoints (1k/2k/4k) record validation curves. Fixed validation gates 2.5/2.35/2.25 bpb select the first recorded crossing, if any. The only test-scored checkpoints are these gate-selected checkpoints plus the final checkpoint. No test-based checkpoint, LR or seed selection occurs.

A common validation threshold does not force equal achieved loss. We report actual test gaps and require absolute gap <=.02 bpb in every paired seed for the prespecified **descriptive** matched-quality screen. This is not a statistical equivalence test or frontier task match. A failed screen is not called matched quality.

The statistical model counts exactly the same windows/target presentations, with fresh suffix context per sampled window. Orders 2/3/4 (up to four preceding bytes, a five-gram) share the maximum-order count table; order is chosen on full validation. Its storage, parameters, FLOPs and fitting work are not matched to neural models. The inherited `ngram.py` standalone docstring/entrypoint refers to the old v12 budget: use **v14 `run.py`**, which supplies the correct 4,096,000-target ledger.

Full validation/test scoring uses 1,147,904/1,291,904 targets, resetting state each 128-byte window. The first source byte and incomplete trailing 102/108 target bytes are excluded for every family. Bits/byte is next-byte cross-entropy, lower is better, not an accuracy percentage, standard word perplexity, or intelligence measure. Three seeds quantify initialization variation on one corpus, not cross-domain uncertainty.

## Timing and resource accounting

Timing happens after all fitting and scoring in the experiment process. Eight deterministic test prefixes, 128 bytes each, receive 32 observed cached updates; five repetitions per prefix discard the first two. Both neural and statistical paths compute **33 full next-byte distributions** per retained probe: one prefix distribution plus 32 update distributions. These are directly timed batch1 teacher-forced replay probes, not exactly 32 sampled generated outputs or service throughput.

There are 24 retained sequences. p50/p95 describe local sequences; the decode-step p95 is the p95 of **sequence-average** update times, not individual inter-token p95. Softmax/full 256-way probabilities are included; sampling, networking, queuing, tokenization, tools and request scheduling are not. Quality resets at 128 bytes while timing grows cache 128→160; no long-context quality conclusion follows. Neural prefill calculates all-position logits; a last-position output head could be more efficient. Statistical prefill legitimately needs only the final suffix when prefix likelihood is not requested. Fixed timing order and uncontrolled host scheduling remain limitations.

All pilot and final fitting seconds and total script wall time are retained. Training seconds exclude validation/checkpoint serialization and testing, so they are not total experiment cost. Gate-crossing time measures progress inside a run which still continues to 4k; it is **not realized early-stop savings**, because all pilot and final runs are actually executed. CPU dollar allocation is unknown, despite no incremental paid compute. State tensor bytes exclude weights, allocator/interpreter and transient activations. Serialized statistical tables are not live RAM; `memory_audit.py` measures the deduplicated Python object graph separately.

The new ledger, warmup, cosine horizon and selected LR differ from v12, so a cross-version quality change cannot isolate extra steps as its sole cause. Within-v14 validation curves are the direct evidence for continued learning. Parameter/data/selection-budget matching does not imply equal FLOPs, equal wall time, or equal total memory.

## Economic requirements remain separate

[Verified primary-source prices and full arithmetic](pricing_audit.md), `economics_audit.py` and `economics_audit.json` preserve target retail rates $0.05/M uncached input, $0.001/M cached input and $0.10/M output, half Meta Contributor. Contributor training permission and unavailable max reasoning prevent automatic private/full-reasoning equivalence.

At 30% margin, output-only hardware must deliver **3,968.25 generated tokens/s per $1/hour** at full utilization, or 6,613.76 at 60%, before separate costs. The fixed illustrative ledger 10M uncached + 5M cached + 2M visible + 2M hidden has $0.905 revenue if hidden is billed. At $1/h, 60% utilization, 10% cached-work sensitivity and $0.10 other cost per complete ledger, 8k prefill is infeasible before decode; 80k prefill requires 3,917.31 generated t/s, or 5,565.53 if hidden is unbilled. These are scenarios, not observed user usage or benchmarks.

**This CPU byte experiment does not pass or fail those commercial-token thresholds: the units, hardware cost and workloads are not comparable.** The serving-economic gate remains unmeasured. Never substitute bytes/s for provider-token/s, scoring throughput for generation, or a single-card hourly price for the whole replica. Training/failed experiments, tools, support, payments, storage/network, idle headroom and failover need explicit nonduplicated cost allocation. A $500 gross subscription with 30% margin and $150 assumed non-serving cost leaves $200 serving budget, or $141.67 with illustrative 20% included tax; no sustainable frontier service is established.

## Reproduction and audit

Python 3.12, PyTorch 2.6.0+cpu, NumPy 2.3.5 and PyArrow 19.0.1; see `requirements.txt`. From `experiments/v14`:

    python prepare_data.py
    WIKITEXT_DATA=. python run.py
    python summarize.py
    WIKITEXT_DATA=. python memory_audit.py
    python -m unittest -v test_models test_v14 test_thresholds
    python economics_audit.py

The local run reused hash-verified cached original data and CPU packages; it did not download a new dataset or use prior neural checkpoints. Neural checkpoints and per-window binary arrays are local reproducible intermediates, not redistributed; checkpoint hashes are provided. `train_starts.npy` is regenerated by `run.py`, and independently verified by tests. No pickle is loaded; statistical counts are rebuilt from the ledger. `ngram.py` imports pickle only to measure serialization. Do not run its legacy standalone main for v14.

See [independent audit](audit.md), test outputs, frozen protocol, selection and complete numerical results. An artifact manifest records the delivered text/source hashes. Hashes verify identity, not scientific validity or provenance attestation.

## Next gate

A second licensed corpus, sufficiently trained Transformer and other architecture baselines, matched **total** training/inference/storage budgets, untouched evaluation, executable reasoning/retrieval tests, and real tokenized service traces are still needed. Any next experiment should be selected from these gaps, not extend tiny-model curves into an AGI or frontier-cost claim.
