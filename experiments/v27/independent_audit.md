# Independent v27 bounded-memory audit

## Outcome

No scientific correctness blocker found. Exact sliding eviction is implemented, but none of the six bounded local/GRU hybrid settings meets the frozen quality gate. Even scope2048 loses 0.097607 bits/byte for local (paired95% interval [0.028833,0.203230]) and0.085195 for GRU ([0.025315,0.177186]) versus full copy memory. Upper bounds exceed0.01 substantially. These are conditional byte-language-model results, not frontier-quality or task equivalence.

## Independent checks

- All frozen source, data-manifest and six checkpoint hashes match. The ledger reconstructs50 windows,36 documents and6400 targets from pinned FineWeb rows1100–1199 with8193-byte caps. Each4096-input window scores only its final128 next-byte targets. No scores determine document/window selection or mixture weights.
- Independent substring-search oracle matches5622 complete256-byte distributions on constant, repeated, cycling and random byte streams, including exact eviction boundaries and scopes17/32/512/1024/2048/full. Maximum normalization error is8.89e-16. Positive event counts, legal starts and exact retained-event totals are checked at every update.
- On every evaluated history,25,600 independent oracle distributions reproduce all saved copy target losses and active masks. All hybrid losses reconstruct exactly with fixed0.3 local/0.2 GRU weights and neural fallback. All34 aggregate means agree.
- Building a bounded cache from only its last S prefix bytes matches feeding the full prefix and evicting, including32 continuation steps, across4800 checks. This validates timing's shortened cache initialization. Absolute event positions differ by a constant, but eviction and output distributions are translation-invariant.
- All six checkpoints pass native full-head versus optimized scoring on a real4096-input window. Maximum logit discrepancy is2.194e-5; saved target-loss discrepancy is5.866e-6, below inherited1e-4 tolerance. Changing future inputs leaves the first scored logit exactly unchanged.
- All14 reported paired bootstrap intervals, point estimates, document totals and quality-gate decisions independently reconstruct. Seeds are averaged first;5000 document-cluster resamples use target-weighted ratio-of-sums and seed27. Shared-document seeds are not independent corpus replications. Convenience sampling, only36 documents, repeated windows within documents, and shared web provenance constrain generalization. No multiple-comparison guarantee is implied.
- Exact128-byte scored target spans have no matches in WikiText training,1100 available prior evaluation-document records, or another current document. This does not exclude shorter, near-duplicate or semantic overlap.

## Storage interpretation

The bounded cache actually removes expired events, follower counts and empty contexts; it does not retain full history and merely filter queries. On high-cardinality streams, retained event/context counts stabilize at1508/3044/6116 for scopes512/1024/2048. Reachable storage remains approximately674KB/1.349MB/2.696MB from4096 through32768 observed bytes. The cache retains a growing integer position counter, so strictly speaking integer bit widths grow logarithmically with stream duration; this is bounded event/history storage rather than a mathematical fixed-bit infinite-stream implementation.

On the natural-text timing cohort, median final index storage is approximately583KB/1.161MB/2.295MB for scopes512/1024/2048 versus1.241MB for the compact full-history index. Different representations matter: the bounded dict/deque implementation has large per-event overhead. Scope2048 is physically larger than full at this finite length. The experiment supports actual eviction, not a claim that every finite scope reduces measured RAM. Reachable CPython-object sizes include retained containers and data, not peak RSS, allocator fragmentation, transient workspace or weights. Neural-state bytes and weights are separate.

## Timing and protocol limitations

All480 timing ledger rows have unique expected arm/seed/scope/document/repetition identities, correct weight sizes and state storage. Independent allocated tensor storage confirms local130064-byte and GRU512-byte state. The full path includes cache initialization, neural prefill, complete256-way normalization/mixing, and32 observed-byte updates yielding33 distributions. Timings are not sampled generation, production throughput or commercial-token throughput. Local neural prefill prunes structurally unreachable input;4096 supplied bytes do not mean4096 processed bytes.

Timing is on a shared host, and independent audit work overlapped part of it. Randomized job order, discarded warmup and two retained repetitions do not establish a clean causal speed comparison. Do not infer a hybrid speedup from noisy GRU medians. Two very short local-neural prefills report0.0 process CPU time with positive3.17/3.36ms wall time, consistent with coarse CPU accounting resolution; this is not zero computation. Wall timings are positive and CPU timings nonnegative. Durations were not independently remeasured.

The source/protocol freeze predates main scoring. The independent cache script happened to exist at freeze and is included unchanged in source hashes; independent results/statistics scripts were added afterward. Summarization is postfreeze presentation of the prespecified bootstrap, independently reconstructed here. No main scientific source was edited by the auditor.

No training was done. Inherited local/GRU histories are not a matched-compute architecture contest, and fixed mixture weights are not retuned. Copy-only, hybrid and neural baselines must remain visible. Successful small byte-prediction gains do not establish distant reasoning, an architecture winner, AGI, frontier quality, service economics or the requested pricing margin.

## Reproduction

With the original frozen inputs available, run independent_cache_audit.py, independent_results_audit.py and independent_statistics_audit.py using PYTHONPATH=/tmp/frontier-v10-packages:/tmp/frontier-v8-packages. Machine-readable findings are in the corresponding JSON files. Audit script CPU work and setup are outside the separately measured main/timing CPU figures.
