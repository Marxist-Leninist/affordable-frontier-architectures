# Independent v15 audit

2026-11-04. Source and provenance review; numerical checks are added below only after execution.

## Source provenance and scope

The cached Hugging Face viewer JSON contains exactly row indices 0–99, `partial=false`, and no truncated cells. Its raw response SHA256 is `a4eeea6d8ea201f9835167aed98ea8e0b849a9e97db30954f0e03c42cfc869c8`. The original 100 texts total 370,599 UTF-8 bytes before the experiment's document caps. The original fetch reported `x-revision: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`; that historical HTTP header was supplied by the experiment operator and is not independently re-attested by the cached body.

The [pinned primary dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md) was independently retrieved. Its SHA256 is `a0cc8998a20499432b28b6575f3046b714938eb8e11b8d59a1d25ddf3716061e`. It declares [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/) and says use is also subject to [Common Crawl's terms](https://commoncrawl.org/terms-of-use). Dataset attribution is to HuggingFaceFW/FineWeb-Edu. The raw corpus and full copied card should remain local, not be included in the research commit.

The card describes `sample-10BT` as sampled from a larger random subset. Taking the first 100 viewer rows is nevertheless an ordered convenience sample, not a new probability sample or benchmark split. Results cannot establish population confidence intervals, whole-corpus performance, or broad generalization. A new dataset name also does not establish absence of overlap with WikiText training: exact and near-duplicate contamination have not been comprehensively excluded. Model freezing prevents adaptation to this evaluation, but not historical source overlap.

The viewer API snapshot is identified by response/text hashes and revision evidence. Reproduction must fail closed if content or revision changes rather than silently use replacement rows. A moving viewer endpoint is not itself an immutable source URL. Pinned source reconstruction or retaining the local snapshot is required if the viewer ceases serving matching content.

## Frozen-code synthetic tests

Six independent checks pass in `audit_tests.txt` (0.082 seconds). They check per-document window boundaries, target alignment including a batch boundary, exact paired suffix targets, statistical reset/context semantics, immutable n-gram evaluation and probability normalization, frozen source/data hashes, and the distinction between equal-window and equal-document averaging. The 512-byte and fresh-128-byte conditions predict exactly the same last 128 targets.

For the four-byte statistical context, any long-condition advantage is confined to the first three targets of the scored suffix: the fresh short condition has not yet accumulated four observed bytes. From the fourth target onward the statistical contexts are identical. Such a gain does not demonstrate memory beyond four bytes. Model state never crosses a document or evaluation-window boundary.

Bootstrap intervals in `pair_summary` concern the equal-document mean, whereas the headline `long_minus_short_bpb` is equal-window weighted. These are distinct estimands. Resampling this convenience sample is descriptive, not a population confidence interval.

## Final numerical and long-context audit

`audit_final.py` and `audit_final.json` reproduce the final check. All six checkpoints and frozen protocol/source identities match. All stored neural score means, target counts and paired deltas match the numerical results. Thirty-six actual long windows (six distributed windows per checkpoint) were freshly rescored with an independent float64 log-softmax/gather loss expression. Corresponding fresh short suffixes were also rescored. Maximum loss discrepancy was below 3.0e-7 bits/byte.

For these actual 512-byte sequences, full-window logits were compared with four cached 128-byte chunks. Their last-128-target loss differs by at most 1.35e-7 bpb. Transformer float32 logits differ by at most 3.53e-5; an initial stricter 2e-5 max-logit tolerance failed on this small numerical discrepancy, so the final test uses 1e-4 logits plus a separate 2e-6 bpb loss tolerance. This is disclosed rather than suppressing the initial result. GRU cached/full logits are identical in these checks. Perturbing future inputs from position 400 produces exactly zero change in preceding logits for all tested checkpoints.

An additional independent reference explicitly computes interleaved RoPE, the scaled attention matrix, a causal mask, softmax and value aggregation at length 512. It matches the implementation within 1.79e-7 absolute output units. No target-alignment, RoPE formula, causal-mask or cache-offset error was found that explains the large observed Transformer degradation. The finding is supported for these tiny, frozen, 128-byte-trained checkpoints; it does not establish a general Transformer limitation, identify a unique causal mechanism, or compare properly long-context-trained alternatives.

The fresh numerical audit covers sampled neural windows, not a second full-corpus neural pass. Synthetic tests check statistical alignment, but this audit does not independently rebuild and fully rescore all statistical predictions. No model weights, original scoring sources, dataset selections, or timings were modified.

## Timing audit

Both original prefix-length timing cohorts happen to be the same eight documents, indices 0–7. Each probe computes 33 distributions, including the prefill distribution; local sequence-level p95 is not production inter-token p95. The synthetic tests completed at 11:15:56.172 UTC, before original results were finalized at 11:16:02.419 UTC, so original timing interference cannot be excluded. The operator was informed and planned an isolated timing rerun while preserving the original measurements. The numerical audit CPU work finished before that rerun was requested. Any newer timing artifact must identify itself separately; this audit did not repeat latency measurements.

## Reproduce

After reconstructing the exact input snapshot and v14 local checkpoint intermediates:

    python -m unittest -v test_v15
    V14_DIR=../v14 python audit_final.py

The final sampled audit took 2.12 seconds in the initial successful run. No training, paid compute, production edits or repository publication was performed by this auditor.

## Final documentation, retiming and exploratory-control review

The completed `retiming.json` is a separate artifact, preserving the original `results.json` measurements. It uses the same document IDs 0–7 and all six frozen checkpoints plus the statistical model, with no quality recomputation or new model selection. Its finalization time is 11:18:23.271 UTC, after this numerical audit (11:17:19.861) and the separate WikiText diagnostic (11:17:44.817). The final 13-unit-test pass completed afterward at 11:18:33.663 UTC, in 0.053 seconds. The operator reports the rerun did not overlap these tasks; their recorded ordering is consistent with that report. This remains process-level scheduling evidence, not a whole-host isolation guarantee. Every timing summary value was independently recomputed as the median of corresponding stored seed values and agrees exactly.

The README's numbers and limitations match the summary and retiming artifacts. Its claim that all paired Transformer windows worsen was checked directly against all stored paired window scores. No primary pair passes the fixed .02-bpb descriptive quality screen. The timing rerun does not change that failed quality gate.

The additional `wikitext_diagnostic.py/json` is explicitly post-hoc, triggered by the unexpected transfer result. Source review confirms that it reuses fixed v14 final checkpoints and scores identical tail targets on the first 100 contiguous 512-target windows of the official WikiText test serialization. It does not change primary scores, models, hyperparameters or selection gates. Stored mean short/long losses recompute to 2.333637/5.285505 for Transformer and 2.207163/2.176278 for GRU, matching the README. This supports occurrence of the failure on both corpora; it does not identify the unique mechanism, establish a prespecified confirmatory result, or remove the different document-boundary protocols. The diagnostic's numerical predictions were not freshly rerun by this auditor; its reused scoring path was independently checked above.

Reproduction commands correctly set `V14_DIR=../v14` and `WIKITEXT_DATA=../v14` when following the v14 README's local extraction workflow, while FineWeb viewer data and window scores live in v15. If a reproducer extracted WikiText elsewhere, `WIKITEXT_DATA` must point to that actual location. `audit_final.py` needs regenerated local `window_scores.json` and the v14 checkpoint files; these are correctly described as local intermediates. No further blocker was found.
