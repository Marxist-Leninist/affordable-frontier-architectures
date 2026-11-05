# v27: bounded copy memory is a real quality/storage tradeoff

Isolated CPU experiment, November5,2026. No paid compute, training, production changes or corpus/weights publication.

## Useful result

Physical sliding-cache eviction works correctly but does not preserve full-history quality. All six bounded configurations fail the prespecified practical gate: upper paired95% loss-increase interval must be <=0.01 bits/byte versus the same neural model with full copy history. GRU2048 loses0.08520bpb [0.02531,0.17719] versus full; local Transformer2048 loses0.09761 [0.02883,0.20323]. Smaller caches still improve over neural-only, but sacrifice useful older repeated strings.

Bounded history does not automatically mean lower finite-size RAM: the2048-byte Python deque/dict index occupies about2.295MB, versus1.241MB for the compact full4096-prefix index. A512-byte index uses0.583MB but loses0.17691/0.20537bpb for GRU/local. No equal-quality cost or speed win, distant reasoning, frontier quality, or profitability is established.

## Matched evaluated targets

Lower bits/byte is better; means over three seeds. Identical neural probabilities and targets are reused across scopes, isolating memory policy.

|Model|Neural|512 copy|1024 copy|2048 copy|Full copy|
|---|---:|---:|---:|---:|---:|
|local|3.082398|2.867440|2.813955|2.759676|2.662069|
|gru|3.006452|2.822613|2.777962|2.730899|2.645704|

All-target primary cohort:50 windows,6400 targets,36 contributing documents out of100 downloaded rows. No score-selected subgroup or parameter fitting. Full copy history grows from3969 to4096 observed bytes across each scored128-target tail; full is not an unbounded-document evaluation.

## Frozen design

Fresh HuggingFaceFW/fineweb-edu sample-10BT train rows1100–1199 at revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9. First8193 UTF8 bytes per document, byte vocabulary0–255, no joining/normalization, nonoverlapping4096-input windows score final128 next-byte targets. Viewer revision checked fail-closed; full/capped document hashes and response hash retained. This is deterministic convenience data and a minimum-length cohort, not representative sampling.

Frozen v19 local span128 Transformer and gated residual GRU, seeds1–3. Same target ledger at evaluation, and same512000-target v19 adaptation ledger, but inherited architecture training/search histories and total compute differ. No equal-FLOPs or converged architecture comparison. Models remain tiny byte predictors (~132k parameters), not frontier baselines. Local Transformer can only use255 input bytes via two128-span layers; its pruned path preserves absolute RoPE positions and is identical across variants. GRU receives full4096 neural input.

Longest available exact context order4/8/16; empirical observed-follower counts plus one uniform pseudocount across256 bytes. Fixed inherited mixture weights .3 local/.2 GRU; neural fallback when no context match. Full index uses prior CompactCache; bounded implementation physically expires all complete context/follower pairs whose start leaves the finalS observed bytes, deleting empty counters and events. Both are complete normalized language models, not target-only shortcuts. Dynamic scope includes the complete context and its already-observed follower. Current targets are never indexed before prediction.

Novelty versus v20/v26: actual bounded-storage eviction implementation, tested streaming invariants and finite memory accounting, on fresh4096-context holdout; not another donor perturbation or repeated claim that exact copying helps. Timing can build bounded memory from only the lastS prefill bytes because earlier complete pairs are necessarily expired. Independent checks establish exact equivalence through32 updates.

## Uncertainty and gate

Average three seed losses first, then5000 paired recipient-document cluster bootstrap draws, seed27; target-weighted ratio of sums. All windows from the same document move together. Seeds do not constitute independent corpus replications. Confidence intervals are conditional on this small deterministic cohort. All14 contrasts and per-document sums/counts are in summary.json, allowing bootstrap reconstruction without publishing text. The0.01bpb noninferiority tolerance was frozen before scores. None of the six scope-vs-full tests passes. No multiple-comparison winner selection was performed.

## Complete-path timing and storage

|Path|p50 / p95 ms|Final index bytes|Neural state bytes|
|---|---:|---:|---:|
|local_neural|25.320 / 30.284|0|130064|
|local_512|27.679 / 34.238|582840|130064|
|local_1024|35.742 / 39.576|1160850|130064|
|local_2048|33.672 / 42.858|2295213|130064|
|local_full|42.992 / 55.743|1240803|130064|
|gru_neural|125.633 / 161.500|0|512|
|gru_512|117.808 / 192.921|582840|512|
|gru_1024|124.741 / 166.265|1160850|512|
|gru_2048|137.751 / 172.963|2295177|512|
|gru_full|135.318 / 183.984|1240783|512|

One CPU thread, batch1, float32 neural weights with float64 normalized256-byte distributions; prefix4096 plus32 observed updates (33 predictions). First8 eligible documents>=4128 bytes,3seeds, one warmup plus2 measured repetitions, randomized job order seed270027;480 measured rows. Complete index construction/query/mixing and neural prefill/updates included. Weights are527616 bytes local/528640 GRU, excluded from table state/index columns.

Reachable CPython object sizes include deque elements and allocation, not peak RSS; tensor state omits transient allocations and Python framework overhead. Repeated timings are dependent and run on a shared host; independent audit computation overlapped part of timing. Some nominal hybrid timings are faster than neural-only due to noise, not an earned acceleration. No causal or matched-quality speedup is claimed. A bounded index is asymptotically fixed but higher-overhead at this context length; the independently tested32k synthetic stream tests storage invariants, not predictive quality.

Main scoring10.900 CPU seconds, timing74.465 CPU seconds, separate hard limits500/200. Preparation, startup, tests and independent audit are not included in those totals, so no exact inclusive CPU budget claim.

## Provenance, checks and reproduction

Training is inherited WikiText-2-raw-v1, Salesforce/wikitext revisionb08601e04326c79dfdd32d625aee71d232d685c3. Training serialization is36718 LF-joined UTF8 rows (10951562 bytes), hashc58d555a94ada0882d1e8c0b3dc7ba68e1c269da5ad9e7a057300245d130c4e8. training_data_manifest.json retains split hashes; checkpoint manifest verifies exact artifacts. No new model training or test-set tuning. FineWeb ODC-BY1.0/Common Crawl terms and underlying source rights apply. Fresh pinned WikiText card still has CC-BY-SA3.0/GFDL metadata and CC-BY-SA4.0 prose; preserve this inconsistency and attribution/share-alike obligations. [Current source review](pricing_sources.md).

Five unit tests passed. Independent cache oracle verifies5622 complete distributions, all eviction boundaries and storage invariants. Independent result audit reconstructs25600 scored copy distributions, all active masks/hybrid losses and34 aggregate losses;4800 suffix-initialization/continuation comparisons pass. All6 native model causal/parity checks pass (max logits2.194e-5, saved loss5.866e-6, future invariance zero). Frozen hashes pass. No exact128-byte scored target span occurs in WikiText train,1100 prior evaluation document records or another current document. Shorter and near-duplicate contamination remains unmeasured. See independent_audit.md and independent_*audit.json.

Prepare lawful inherited v19 checkpoints and manifests under sibling frontier_research_v19/checkpoints; prepare WikiText train under sibling frontier_research_v8. Install requirements in an isolated environment. Directory names are historical executed layout; repository experiments/v27 reproduction requires corresponding sibling aliases. Run prepare_data.py, unit tests, run.py --freeze for a new reproduction only, run.py, timing.py, summarize.py. Never overwrite/refreeze historical protocol. Post-score summaries and independent checks do not change frozen scientific source. Original source, protocol, hashes, full aggregate/bootstrap sufficient statistics, timing rows and audit scripts retained; losses.npz is regenerable and corpus/checkpoints excluded.

## Economics

Retained targets$0.05/M uncached input,$0.001/M cached input,$0.10/M output and30% margin on net revenue are unchanged. Conditional output-only floor3968.25 billable output tokens/s per$1/h at100% utilization is before all other costs. <=$500/month keeps the average/heavy workload unchanged; GBP200/GBP20 stay separate. Explicit complete ledger and fresh pricing are in pricing_sources.md. CPU bytes are not commercial tokens; none of these tests measures all-in GPU serving, frontier quality, training amortization or monthly profitability. The failed bounded-quality gate blocks an equal-quality cost claim even before those broader omissions.

Additional accounting caveat: two short prefills recorded0.0 process CPU seconds with positive wall time, consistent with coarse CPU timing resolution, not zero computation. Training text hash and current/prior exact full-document duplicate checks are recorded in provenance_checks.json. Copy-only losses are6.705460/6.290003/5.829483/5.182766bpb for512/1024/2048/full; these weak standalone predictors complement, rather than replace, neural prediction.
