# v26: distant-repeat histories expose copy-memory benefit, not distant reasoning

Bounded isolated CPU research, 2026-11-05. No training, paid compute, production changes, corpus or weights publication.

## Main finding

The stricter history-only repeated-string cohort contains870 targets (7.01%) across63 documents. The context-trained Transformer still has no clear original-over-donor benefit: +0.005716bpb donor-minus-original,95% interval[-0.022700,0.035734]. All-target effect is+0.000610[-0.006147,0.007276]. Its original context still loses to suffix512 by0.106594bpb overall and0.099103 in the subgroup.

By contrast, fixed complete copy hybrids show the expected content-specific positive control. All-target original-vs-donor benefits are0.206560/0.139924/0.115938bpb for full/local/GRU hybrids; subgroup benefits0.990397/0.726898/0.592782. This demonstrates usable repeated-string information in this cohort, not learned distant reasoning. GRU hybrid improves all-target loss2.947848→2.659809bpb; local2.694185 remains0.034376 worse, so this is not an equal-quality architecture speed contest.

## Fixed comparisons

Lower bits/byte is better. Each mean averages three seeds except deterministic standalone copy. All-target results are primary; the history-only distant-repeat subgroup is secondary.

### all
12416 targets (100.000% of all), 68 contributing documents, 97 windows.

|Model|Original2048|Donor2048|Suffix512|
|---|---:|---:|---:|
|context2048|3.446778|3.447388|3.340184|
|local|3.029557|3.029557|3.029556|
|gru|2.947848|2.947854|2.947661|
|context2048Hybrid|3.003269|3.209829|3.122590|
|localHybrid|2.694185|2.834109|2.842960|
|gruHybrid|2.659809|2.775747|2.785377|
|copy|5.748566|6.317810|6.738069|

Donor minus original is positive when original distant content helps relative to unrelated content. Original minus suffix is positive when longer original context hurts. Hybrid suffix512 changes both neural reset context and copy scope; suffixCopyOnly changes only copy scope with original neural outputs held fixed.

- context2048, donor2048 minus original2048: +0.000610 bpb; recipient-document bootstrap95% [-0.006147376517822232, 0.007275616841451063]; zero-denominator replicates 0.
- context2048, original2048 minus suffix512: +0.106594 bpb; recipient-document bootstrap95% [0.09221908450463473, 0.12093196168860922]; zero-denominator replicates 0.
- local, donor2048 minus original2048: +0.000000 bpb; recipient-document bootstrap95% [0.0, 0.0]; zero-denominator replicates 0.
- local, original2048 minus suffix512: +0.000000 bpb; recipient-document bootstrap95% [-1.7696618895362836e-07, 4.6025944305950905e-07]; zero-denominator replicates 0.
- gru, donor2048 minus original2048: +0.000006 bpb; recipient-document bootstrap95% [-0.0001397040846997066, 0.00014284622164369086]; zero-denominator replicates 0.
- gru, original2048 minus suffix512: +0.000187 bpb; recipient-document bootstrap95% [5.928055487901068e-05, 0.0003033589792607602]; zero-denominator replicates 0.
- context2048Hybrid, donor2048 minus original2048: +0.206560 bpb; recipient-document bootstrap95% [0.16436057534603593, 0.2508035487518875]; zero-denominator replicates 0.
- context2048Hybrid, original2048 minus suffix512: -0.119322 bpb; recipient-document bootstrap95% [-0.16733169010102564, -0.0719995415318871]; zero-denominator replicates 0.
- context2048Hybrid, suffixCopyOnly minus original2048: +0.207360 bpb; recipient-document bootstrap95% [0.16551571962768624, 0.2509231154098878]; zero-denominator replicates 0.
- localHybrid, donor2048 minus original2048: +0.139924 bpb; recipient-document bootstrap95% [0.1095163467775354, 0.17308509587943413]; zero-denominator replicates 0.
- localHybrid, original2048 minus suffix512: -0.148776 bpb; recipient-document bootstrap95% [-0.1821457656598035, -0.11656619174782455]; zero-denominator replicates 0.
- localHybrid, suffixCopyOnly minus original2048: +0.148776 bpb; recipient-document bootstrap95% [0.11656626296171689, 0.18214589306802215]; zero-denominator replicates 0.
- gruHybrid, donor2048 minus original2048: +0.115938 bpb; recipient-document bootstrap95% [0.08975984165762664, 0.14410001409868564]; zero-denominator replicates 0.
- gruHybrid, original2048 minus suffix512: -0.125568 bpb; recipient-document bootstrap95% [-0.15413830951244867, -0.0981820146593123]; zero-denominator replicates 0.
- gruHybrid, suffixCopyOnly minus original2048: +0.125726 bpb; recipient-document bootstrap95% [0.09833294943987125, 0.15433489073140125]; zero-denominator replicates 0.
- copy, donor2048 minus original2048: +0.569244 bpb; recipient-document bootstrap95% [0.4613187897826007, 0.6837120932593904]; zero-denominator replicates 0.
- copy, original2048 minus suffix512: -0.989503 bpb; recipient-document bootstrap95% [-1.1091735201053643, -0.8726415639097048]; zero-denominator replicates 0.
### distant_repeat
870 targets (7.007% of all), 63 contributing documents, 88 windows.

|Model|Original2048|Donor2048|Suffix512|
|---|---:|---:|---:|
|context2048|3.757091|3.762807|3.657988|
|local|3.361265|3.361265|3.361265|
|gru|3.185029|3.184776|3.184664|
|context2048Hybrid|2.354387|3.344784|3.333000|
|localHybrid|2.276711|3.003609|3.074737|
|gruHybrid|2.298426|2.891208|2.942749|
|copy|2.829019|6.087570|6.661513|

Donor minus original is positive when original distant content helps relative to unrelated content. Original minus suffix is positive when longer original context hurts. Hybrid suffix512 changes both neural reset context and copy scope; suffixCopyOnly changes only copy scope with original neural outputs held fixed.

- context2048, donor2048 minus original2048: +0.005716 bpb; recipient-document bootstrap95% [-0.02269994191665695, 0.035733659153855284]; zero-denominator replicates 0.
- context2048, original2048 minus suffix512: +0.099103 bpb; recipient-document bootstrap95% [0.06486981586743143, 0.1328282851640848]; zero-denominator replicates 0.
- local, donor2048 minus original2048: +0.000000 bpb; recipient-document bootstrap95% [0.0, 0.0]; zero-denominator replicates 0.
- local, original2048 minus suffix512: -0.000000 bpb; recipient-document bootstrap95% [-1.5150982888705602e-06, 9.438194626403044e-07]; zero-denominator replicates 0.
- gru, donor2048 minus original2048: -0.000253 bpb; recipient-document bootstrap95% [-0.0006660282572926976, 0.00010458280019529846]; zero-denominator replicates 0.
- gru, original2048 minus suffix512: +0.000365 bpb; recipient-document bootstrap95% [1.2355572887063778e-06, 0.0008580099370562684]; zero-denominator replicates 0.
- context2048Hybrid, donor2048 minus original2048: +0.990397 bpb; recipient-document bootstrap95% [0.7303466119046085, 1.2414441393206506]; zero-denominator replicates 0.
- context2048Hybrid, original2048 minus suffix512: -0.978613 bpb; recipient-document bootstrap95% [-1.2232676476086048, -0.7248444825845471]; zero-denominator replicates 0.
- context2048Hybrid, suffixCopyOnly minus original2048: +1.053536 bpb; recipient-document bootstrap95% [0.8040141064257531, 1.2945187692050644]; zero-denominator replicates 0.
- localHybrid, donor2048 minus original2048: +0.726898 bpb; recipient-document bootstrap95% [0.5044186766373974, 0.9380901056128181]; zero-denominator replicates 0.
- localHybrid, original2048 minus suffix512: -0.798026 bpb; recipient-document bootstrap95% [-0.9880880102004844, -0.5911113442236204]; zero-denominator replicates 0.
- localHybrid, suffixCopyOnly minus original2048: +0.798026 bpb; recipient-document bootstrap95% [0.5911113716534135, 0.9880871394738627]; zero-denominator replicates 0.
- gruHybrid, donor2048 minus original2048: +0.592782 bpb; recipient-document bootstrap95% [0.3891813984754344, 0.7917773845169737]; zero-denominator replicates 0.
- gruHybrid, original2048 minus suffix512: -0.644323 bpb; recipient-document bootstrap95% [-0.8316536841558536, -0.4544926047551409]; zero-denominator replicates 0.
- gruHybrid, suffixCopyOnly minus original2048: +0.644608 bpb; recipient-document bootstrap95% [0.4550702481681573, 0.8319557093312687]; zero-denominator replicates 0.
- copy, donor2048 minus original2048: +3.258551 bpb; recipient-document bootstrap95% [2.6531768402035896, 3.8328100068915636]; zero-denominator replicates 0.
- copy, original2048 minus suffix512: -3.832494 bpb; recipient-document bootstrap95% [-4.325379346095511, -3.276145388820139]; zero-denominator replicates 0.

## Design and limitations

Fresh pinned FineWeb-Edu sample-10BT rows1000–1099, revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9; first4097 UTF8 bytes, no normalization, joining or content selection. Vocabulary byte0–255. All nonoverlapping2048-input windows score their identical final128 target bytes. Frozen v23 context2048 Transformer and v19 local128 Transformer/GRU checkpoints, seeds1–3. No fitted parameters or evaluation-based tuning. Cross-architecture inherited training histories are unmatched.

The subgroup was frozen before any scores. At target index t (1921–2048), history is raw[:t], excluding current target. Its final8 bytes must have a prior complete8-context-plus-observed-follower occurrence whose follower j<1408, entirely in the replaced prefix; no complete9-byte occurrence can lie in the dynamic last512 observed-byte suffix. Selection never asks whether any historical follower equals the current target. Retained-middle matches may remain. Parent tightened initial strictly-before-suffix wording to the manipulated prefix before freeze; design_clarifications.json preserves this change. All targets remain primary, even those without qualifying history.

The donor replaces first1408 bytes with first1408 from the next eligible distinct-full-document-hash row in cyclic order, preserving641 bytes including all128 targets. Neural outputs are actually recomputed for original and donor. Equal lengths/positions remove a simple length confound, but the artificial boundary can disrupt prediction. Original beating donor does not alone establish long-context usefulness if suffix is better. Content-specific substring copying does not establish distant reasoning, AGI, frontier quality or general memory.

Copy is a complete normalized256-byte language model: longest available observed context order4/8/16, empirical next-byte counts plus one uniform pseudocount. This differs from fixed8 subgroup selection. Standalone copy falls back to uniform; hybrids fall back to neural when no match. Fixed v20 mixture coefficients are .6 full Transformer, .3 local, .2 GRU. The .6 is transferred unchanged to v23 context2048 and is not calibrated for that checkpoint. Scoring uses existing exact reference index; timing uses tested mathematically identical compact index. No test-set tuning.

Neural suffix512 is a reset on last512 fixed inputs, effective385–512 histories. Copy suffix512 is a dynamic last512 observed-byte restriction requiring the whole stored context/follower pair inside. This differs by one byte from v20's suffix511 and is explicitly frozen here. Full-vs-suffixCopyOnly isolates copy memory scope without changing neural outputs. Original-vs-donor hybrid changes both components as the complete system should.

Statistics average seeds then use5000 paired recipient-document bootstrap replicates, seed26; subgroup is ratio of sums over selected targets, not equal document weighting. Conditional deterministic convenience cohort, not population sampling; cyclic donor dependencies not fully modeled. Shared documents mean seeds are not independent corpus replications. Subgroup size and document concentration limit inference; no selection by outcome or formal equivalence claim.

## Complete original-path timing

|Model/path|p50/p95 total ms|Prefill state bytes|Final state bytes|Final copy-index bytes|
|---|---:|---:|---:|---:|
|context2048_neural|88.552/97.796|2097152|2129920|0|
|context2048_hybrid|95.781/100.005|2097152|2129920|626195|
|local_neural|30.530/31.620|130064|130064|0|
|local_hybrid|40.853/45.373|130064|130064|626183|
|gru_neural|90.897/96.910|512|512|0|
|gru_hybrid|95.427/102.625|512|512|626183|

Batch1 one-thread float32 weights; float64 full256 softmax/mixing. First8 eligible documents>=2080 bytes,3 seeds, randomized job order seed260026; one whole-path warmup then one measured repetition,24 samples per variant. All original2048 paths include neural prefill and32 observed updates,33 complete distributions. Hybrid includes prefix index construction, query, normalization and mixing. No sampled generation, service concurrency or commercial-token conversion. Local prefill processes only structurally reachable final255 bytes for its one output despite2048 supplied inputs; supplied-byte rate is not processed-byte throughput or equal distant-context capability. Shared host, no exclusivity claim. Index reachable CPython-object storage includes dictionary/array allocation but is not peak RSS; tensor state excludes weights, temporary allocations and Python overhead. Timing/state/weight/count rows retained in timing.json.
Main evaluation CPU 60.114s; timing CPU 23.202s. Separate hard main/timing limits650/120 seconds; overall allowance900. Preparation/tests/audit/report startup not all instrumented, so no exact inclusive CPU total claimed.

## Reproduce and audit

Prepare sibling frozen v19/v23 checkpoints according to their manifests, v8/train.txt and public pinned data independently. requirements.txt lists dependencies; local environment used PYTHONPATH=/tmp/frontier-v10-packages:/tmp/frontier-v8-packages. Run prepare_data.py; run.py --freeze; python -m unittest -v test_v26; run.py; summarize.py; check_data.py; timing.py; write_report.py. Frozen protocol hashes pre-score scientific source, data/input manifests and subgroup mask. write_report.py is post-score presentation only. Independent audit scripts may be added after freeze without changing frozen evaluation. Source, losses, masks and manifests may be archived; exclude rows.json, checkpoints and __pycache__. Do not silently refreeze historical runs. Six preflight tests passed, including independent history oracle, target-independence, compact/reference parity, causal and optimized neural parity, donor invariance and hashes. No evaluation failures or post-score protocol adaptations occurred. Fresh WikiText card retrieval timed out; this is retained in data_audit.json and the pinned v25 card provided inherited verification, as DATA_CARD.md explains.

## Economic boundary

No established frontier quality, architecture winner or30% margin. Requested targets remain US$0.05/M uncached input, US$0.001/M cached input, US$0.10/M output; conditional output-only hardware floor3968.25 billable output tokens/s per US$1/h at100% utilization before other costs. <=US$500/month retains same useful average/heavy full-month workload; GBP200/GBP20 separate. Byte-level CPU observations cannot establish commercial cost. All-in costs include prefill/decode, cache, hidden reasoning/tools, idle/bursts/reliability, storage/bandwidth, tax/payment/support and training amortization. Parent handles current primary-source pricing and private archive.

## Independent final audit

Independent checks reconstruct870 subgroup targets across88 windows and63 documents.31/870 selected targets also have a complete matching8-context/follower in the retained middle, so the perturbation need not remove all distant matching evidence. Largest contributing document supplies92 selected targets (10.57%); top5 supply25.86%, median per nonempty document12.

All66 stored aggregate losses, full copy probabilities and all hybrid mixtures reconstruct; compact/reference distributions agree. All34 bootstrap intervals reconstruct. All9 checkpoints pass native causal checks; maximum native-versus-optimized logit discrepancy2.241e-5 and native-versus-saved loss discrepancy2.835e-5, within the inherited float32 tolerance.144 timing rows have consistent state/weight accounting. Independent exact128-byte target-span searches find no match in WikiText training,1000 prior evaluation document records or other current documents. Shorter and near-duplicate overlap remains unmeasured. See independent_audit.md and independent_*checks.json for details.
