# Independent v13 audit

## Verdict

Pass for the bounded claim: a validation-selected probability mixture of these frozen small models improves held-out WikiText-2 byte cross-entropy over either component. This is not evidence of frontier task quality, lower serving cost, profitability, or a general architecture win.

Eight independent tests pass (`audit_tests.txt`). The auditor did not modify `run.py`, models, input data, checkpoints, or the training ledger. Test source is `test_audit.py`; memory accounting is reproducible with `audit_memory.py`.

## Target alignment and reset semantics

Both scorers predict byte i+1 from bytes through i. They cover exactly 1,291,904 test targets in complete 128-byte windows, resetting neural state and statistical context at every window. Source byte zero and 108 incomplete trailing target bytes are excluded consistently. The statistical context includes at most four observed bytes and never crosses the window boundary. Synthetic known-logit and spy-context tests check exact target order, final truncation, and the 128-byte reset boundary. Sample checkpoint scoring agrees with independently computed cross-entropy; different batch sizes agree. Cached final-position logits agree with full-window logits for both model families.

Ngram training counts the same 1,024,000 target presentations as the neural ledger, including repeated/overlapping sampled windows, and starts context fresh for every sampled window. Evaluation does not mutate counts. Normalization tests pass.

## Selection and frozen artifacts

Mixture weights are selected on full validation using only pilot seed 101, then written before test scoring. The 11-point grid is 0.0 through 1.0. Selected ngram weights are 0.9 for Transformer and 0.7 for GRU. Final seeds 1/2/3 use those frozen family-specific values. No gradient or count update occurs during evaluation. Prior v12 validation selected learning rate .01 for both models and ngram order 4, matching v13's hardcoded choices. Reusing the same validation corpus across v12 model and v13 mixture selection is additional validation selection, not independent validation evidence. Test data was already exposed by earlier experiments, so this is a held-out scoring set for the mixture selection procedure, not a newly untouched research benchmark.

SHA256 checks pass for all eight used neural checkpoints, checkpoint manifest, identical ledger, copied model/ngram sources, dataset manifest, and all three data splits. Protocol source hashes match the executed sources. Hashes establish local identity and reproducibility, not external provenance attestation.

## Score parity and results

Every v13 full-test neural score matches the earlier independent v12 full-corpus rescore within 4.0e-15 bits/byte (`audit_score_parity.json`). Ngram test bpb matches the v12 score within 1e-11. This is a comparison against the already available independent full rescore, not six fresh full rescores performed by this auditor. New synthetic/sample tests separately validate the v13 scoring implementation.

Mean test bpb: ngram 2.3351727883; Transformer hybrid 2.2711571057; GRU hybrid 2.2423992774. All six hybrids beat both of their constituent scores. GRU hybrid gains 0.0927735110 bpb versus ngram on average. Seed sample SD is descriptive training-initialization variation on one fixed corpus, not cross-domain uncertainty. Block improvement fractions are descriptive, with correlated neighboring text blocks.

## Fairness, costs and memory

The hybrid adds two components; it is not parameter-, memory-, FLOP-, latency-, or training-cost matched to either component alone. It inherits neural and ngram training and selection costs, despite no new neural training in v13. Any accounting must distinguish sunk component training from incremental mixture scoring and include both for a from-scratch comparison.

Actual ngram checkpoint size is 2,820,855 bytes. A deduplicated recursive `sys.getsizeof` traversal of the identically rebuilt Python ngram object graph measures **32,205,521 bytes** (about 30.7 MiB), involving 166,687 distinct object identities. Counter tables dominate. This excludes the Python interpreter, allocator fragmentation, transient allocations, neural runtime, and process RSS. Serialized bytes must not be labeled live RAM. The measured object graph is implementation-specific; a compact native representation could differ. Combined serialized sizes in results match actual checkpoint file sizes.

## Timing scope and remaining limitations

The latency probe prefills 127 bytes outside the timer, then measures one cached neural byte and a complete 256-way ngram distribution at a fixed observed context. Ten warmups precede fifty samples. Hybrid samples are the sum of separately timed neural, ngram, and arithmetic-mixture components. This is **component-summed fixed-context CPU microbenchmark timing**, not directly timed end-to-end serving, a representative workload, prefill-inclusive generation, or service p95. Python time boundaries and arrays/softmax differ from v12's logits-only cached decode, so latency figures should not be compared as identical measurements. Neural target-scoring throughput and target-only ngram scoring are also not generation throughput.

The audit's short CPU tests and independent ngram rebuild overlapped part of the main experiment. Together with uncontrolled host scheduling, this makes timing especially descriptive; isolate and repeat for any latency decision. No timing result supports production SLOs or monetary token-cost extrapolation. Byte units are not commercial subword-token units.

The result program assumes nonempty complete evaluation windows and positive finite model probabilities; it is adequate for the fixed checked corpus/checkpoints, not a generic hardened scoring library. The statistical model is rebuilt rather than archived by v13, with reproducibility supplied by frozen source/data/ledger and the prior v12 checkpoint. External quality benchmarks, new untouched corpus splits, matched resource budgets, actual service measurements, and fully specified user workloads remain necessary before broader decisions.

These are deliberately tiny, short-trained byte models, with no demonstration that any neural baseline is near its attainable quality. Statistical complementarity here could reflect undertraining, dataset-local frequency effects, or scale-specific behavior. No equal-compute claim is supported, and hybrid gains cannot be extrapolated to well-trained large models.
