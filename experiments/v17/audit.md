# Independent v17 prefill audit

2026-11-04. Source review, synthetic tests and fresh actual-window checks. No model fitting, production changes, paid compute or repository publication was performed by this auditor.

## Finding

No correctness blocker was found. The optimized monolithic and fixed-128-chunk paths preserve the tested local-attention predictions within the locked floating-point tolerances. This is **numerical equivalence**, not a claim of bit-identical arithmetic, stronger task capability or a production speedup. Timing was intentionally held until CPU audits finished.

## Seven synthetic tests

`test_prefill.py` and `audit_tests.txt` record seven passing tests, initially in 0.817 seconds:

- The requested suffix logits match an ordinary full-head model forward for the full Transformer, local Transformer and GRU. Tests include batch size two, 301-position inputs, 1/128/301-position tails, and monolithic or 37/128/500-position chunks. Thus requested tails that cross partial chunk boundaries are covered.
- Prefill states match ordinary model states, and a subsequent 33-position continuation agrees. Local absolute counters correctly advance from 333 to 366 and retained allocation is 130,064 bytes for batch one.
- A forward hook verifies exactly one vocabulary-head call on the requested tail. All state-dict tensors remain bit-for-bit unchanged. The final RMSNorm and output head are pointwise output operations; omitting their unused earlier outputs does not feed back into backbone states. Both benchmark competitors receive this optimization, avoiding an artificially weak full-head baseline.
- Instrumented calls intercept the actual attention masks. At 2,048 input positions, monolithic local attention supplies a 2,048×2,048 boolean mask in each layer. Fixed-128 chunking supplies at most 128×255; the initial chunk supplies 128×128. The final boolean-mask element counts are 4,194,304 versus 32,640, not process-peak memory measurements. Intermediate mask comparisons, activations, allocator behavior and attention-kernel workspaces also consume memory.
- Independent byte-index arithmetic checks the last-128 target alignment at lengths 512 and 2,048. Invalid tail/chunk/empty-input cases are rejected in the tested helper API.
- The response snapshot has exactly viewer rows 200–299, with no partial response or truncated cells, and its SHA256 and all protocol source hashes agree.

The helper is designed for these known model classes and inference use, not validated as a general training API. It does not itself force inference/evaluation mode; actual scoring and timing callers do so.

## Final numerical artifact audit

`audit_final.py` reproduces the check and saves `audit_final.json`. All checkpoint hashes and protocol/source identities agree. Every stored model/seed/length score list has the expected document/window IDs, target count and mean. All local-equivalence gates and maximum per-window loss differences match their stored values.

Fresh verification covers **72 actual-window/model comparisons**: three distributed windows for each of four model paths, three seeds and two input lengths. Ordinary full-head forwarding provides a separate execution path, and an independent float64 log-softmax/gather expression computes loss. Maximum disagreement with stored loss is **1.58e-7 bits/byte**. Optimized prefill versus ordinary suffix logits differs by at most **1.63e-5**, and the next observed-byte continuation differs by at most **5.73e-6**. These are below the locked tolerances. Local allocated state remains **130,064 bytes** for each sampled batch-one case. All six complete dataset/seed quality screens pass in the original full scoring artifacts.

This audit freshly rescores selected windows, not the entire corpus a second time. Synthetic state/weight tests and verified checkpoint/source identities complement that sample. Statistical-control source review confirms the same hash-checked WikiText ledger, fixed order four and exact tail target indices; this auditor did not independently rebuild or rescore all statistical predictions.

## Data and interpretation

The new FineWeb-Edu snapshot uses the same recorded revision as v15/v16, but a different viewer row interval. It contains 492,444 original UTF-8 text bytes, capped to 272,220 bytes. Sixty-one documents are long enough for the 2,080-byte timing inputs. These are ordered convenience-sample rows, not representative population evaluation or proof against all training contamination. Corpus text is not to be published with the research artifacts.

The 512 and 2,048 conditions score different sets of suffix positions and documents. A difference between their aggregate bpb values is not, by itself, a causal effect of context length. The key equivalent-quality comparison is **monolithic versus chunked on identical targets within each length**. The full-attention Transformer and GRU are additional controls, not quality-matched equivalents of the local model.

The implementation maintains v16's per-layer local attention, including its finite receptive-field and long-range-information limitations. Chunking changes execution, not that model's attention policy. It should not be described as newly learning long-context reasoning.

## Timing source review and scheduling

`benchmark.py` applies the same final-one-position output head to every neural prefill variant. Inputs are prepared before timing; the statistical tensor-to-bytes conversion is also before its timer. Every retained probe computes one prefix distribution plus 32 observed updates, for 33 distributions. The same eight documents with at least 2,080 bytes are used across all variants and prefix lengths. The number of head outputs and input preparation exclusions are therefore explicit and comparable.

Variants are interleaved using a fixed-seed random permutation per document/repetition. This is **randomized ordering**, not a guarantee that each variant occupies each temporal position equally often. Six repetitions discard two warmups, yielding 32 retained probes per model/seed/length. Three seeds share the same corpus cohort; they are not three independent workload samples. Fixed hardware, whole-host isolation, service latency, paid-hour utilization and commercial-token economics are not established by this script.

CPU audit work completed before the auditor's 17:26:30 UTC timing-clearance message. No additional model or latency execution is planned during the benchmark. This is scheduling evidence for these tasks, not proof that the host has no other workload. Timing results and any summary claims require a separate final artifact review after they exist.

## Reproduce

With the same local input snapshot, regenerated v14 checkpoints and v17 scoring intermediates:

    python -m unittest -v test_prefill
    V14_DIR=../v14 python audit_final.py

The fresh numerical audit took 5.31 seconds in its initial successful run. Raw corpus text and locally regenerated per-window intermediates are not necessary to redistribute; document their hashes and recreation requirements.

## Final report and timing artifact review (17:32 UTC)

The completed README, summary, statistical results, timing protocol, all 1,248 retained timing samples and final fourteen-test log were reviewed after the benchmark. No model inference, fitting or latency measurement was repeated during this review.

An independent standard-library calculation verified every timing group's seed count, exact eight-document/four-repetition cohort, absence of duplicate sample identities, seed-level p50 and interpolated p95, aggregate median-of-seed p50, and the identity `end_ms = prefill_ms + 32 * decode_mean_ms`. All 288 local pairs match exactly on seed, row and repetition. The statistical control has only one seed schedule, as disclosed. Its stored means also agree with its per-window records. Neural seed-mean quality summaries agree with the original scoring results.

Confirmed headline values:

- At prefix 2,048, median seed end-to-end p50 is 117.294679 ms monolithic and 55.708953 ms chunked. Median paired end-to-end ratio is 2.019885766, and median paired prefill ratio is 2.680326053. All 96 retained pairs favor chunking for both measures.
- At prefix 512, median paired end-to-end ratio is 0.976053085; only 35/96 pairs favor chunking. At 128, that ratio is 0.997451976 with 45/96 favorable pairs. These do not support an across-length speedup claim.
- Median seed prefill p50 at 2,048 is 92.250158 versus 33.465359 ms. These aggregate medians and paired-ratio summaries are different estimands; the README labels that distinction correctly.
- The largest full-scoring per-window quality error is 2.361822293e-7 bits/byte and maximum logit difference is 3.814697266e-5. All six frozen quality comparisons pass. The README rounds these and the reported model/control losses correctly.

The final fourteen-test log reports all tests passing. The reproduction sequence correctly regenerates omitted rows and window-score intermediates before their dependent audits and summary, sets the v14 checkpoint path, and points the statistical reconstruction at the WikiText files regenerated by v14. It depends on matching pinned snapshot hashes and regenerated checkpoints; it is not a claim that absent raw intermediates can be ignored. No reproduction command was rerun in this final review.

The benchmark was reported to start at 17:26:51 UTC after the auditor's 17:26:30 CPU clearance and to finish around 17:28:45, followed by final tests. The numerical artifacts do not themselves log wall-clock start/end, so those times are execution coordination evidence, not independently reconstructed timing metadata. The retained samples support the reported local CPU comparison, not service economics or statistical independence of 96 workloads. The README preserves the separate cohort, finite context, mask-versus-peak-memory, host-interference and commercial-token limitations. No blocking discrepancy was found.
