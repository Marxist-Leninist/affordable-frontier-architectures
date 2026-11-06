# Independent final read-only audit

No scientific or numerical blocker found. Reviewer independently recomputed all quality means, seed/document deltas,5000-resample intervals and KL aggregates with zero discrepancy. All four neural original-minus-donor intervals include zero. Copy benefit is0.306895 bpb,95% interval[0.235807,0.377445]. Conditional uncertainty, multiplicity, donor incoherence and non-equivalence caveats are retained.

All24 timing medians match31 raw samples each; throughputs use512 scored positions or4 one-step outputs per batch. Decode excludes cache construction. Revised underlying-storage accounting correctly deduplicates allocations: attention4,194,304/2,097,152 bytes at512/256 prefixes; signed4,196,352/2,099,200. Logical payload is smaller because V retains the larger QKV backing allocation. Signed's2048-byte cumulative state and parameter/position-table payloads reconcile.

All36 full-length FP64 model/condition checks are present: maximum optimized error3.42e-14, cache error2.93e-14, tail-head/future errors0. The independent copy distribution oracle has1152 passing checks. Pair/window/target hashes, all12 pinned v7 checkpoints and causal alignment were verified. Source matches v7.

Prediction counts,12.42-second evaluator boundary and3968.254 conditional hardware-only floor are correct. No all-in commercial/GPU/frontier claim is supported. Reviewer ran lightweight arithmetic/hash/source inspection only, not competing benchmarks. Final editorial updates clarify scored positions per second and distinguish retained storage from logical tensor payload.
