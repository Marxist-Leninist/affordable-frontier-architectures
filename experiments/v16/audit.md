# Independent v16 inference-intervention audit

2026-11-04. Nine bounded tests and a separate real-window numerical audit pass. No critical implementation or target-alignment issue was found. This is evidence for the specified CPU float32 experiment, not a general-purpose production implementation or frontier capability claim.

## Implementation checks

`test_local_attention.py` verifies:

1. Replacement preserves all parameter/state-dict names and values. Random-model outputs at lengths 1, 2, 31, 127 and 128 match the original Transformer within floating-point tolerance.
2. An independent reference explicitly computes interleaved RoPE, the scaled query/key matrix, causal **and local-window** mask, softmax, value aggregation and output projection. At window 8, two heads and 31 positions it agrees within 5e-7 output units.
3. Single-layer outputs obey both causality and the inclusive-current-token span. Perturbations outside the eight-position receptive field do not change the tested output; future perturbations do not change prior outputs.
4. Full and irregularly chunked forwarding through 769 positions agree. Chunk sizes cross the trimming boundary, include a one-token chunk, and exceed the entire window. Stored absolute offsets advance with all processed tokens, not retained cache length.
5. Cache K/V tensors retain 127 prior positions at window 128. Explicit clones have no tensor base, zero storage offset, and backing storage exactly equal to their own logical byte size. They do not retain a hidden allocation for the long original prefix. Both batch-2 synthetic and batch-1 actual-checkpoint cases are checked.
6. Window-one attention handles empty prior K/V and advances its position scalar correctly. Invalid nonpositive/noninteger windows are rejected in the exercised cases.
7. Paired targets and per-document window boundaries align exactly; incomplete tails are excluded consistently.
8. Statistical target/context calls and row-index labels are independently checked with a spy; no document-crossing context is introduced.
9. Frozen source and data hashes agree with the protocol. Viewer rows are exactly 100–199, with no truncated cells or partial response.

Nine tests completed in 0.126 seconds on the final run. Original eight-test pass was expanded to include the statistical context spy. No production code or scoring-source changes were made by this audit.

## Frozen checkpoints and actual predictions

`audit_final.py/json` verify all six v14 checkpoints against recorded hashes, source/protocol identities, stored score means, target counts, and the main quality-gate arithmetic. Localized Transformer parameters remain bitwise equal to the original model parameters.

Thirty-six actual 512-input windows were freshly checked, six distributed windows for each of the six checkpoints. Independent float64 log-softmax/gather losses for reset, full and localized conditions agree with reported samples to at most **2.08e-7 bits/byte**. Full versus irregularly cached forwarding agrees within **2.29e-5 logits** and **1.70e-7 bits/byte**. The primary 128-byte Transformer logits are exactly unchanged on the audited actual samples; the main run reports maximum per-window primary loss difference zero over the whole primary set.

The actual batch-1 localized cache has **130,064 bytes** of tensor elements and separately verified backing allocation: two layers, two K/V tensors per layer, four heads, 127 positions, head dimension 16, float32, plus one 8-byte absolute-position tensor per layer. Metadata is included. At 512 processed positions, stored offsets are 512 despite only 127 cached positions. GRU state remains 512 tensor bytes. These figures exclude model weights, Python/interpreter overhead, transient tensors, allocator fragmentation, attention workspaces and process RSS.

The primary quality gate is a two-sided descriptive difference relative to the reset-128 Transformer, not a statistical equivalence proof. Local minus reset mean differences are **−.0214491, −.0213120 and −.0182382 bpb**. Seeds 1 and 2 narrowly fail the fixed .02 criterion in the better direction; seed 3 passes. Therefore an all-seed comparable-quality pass is not supported, even though mean loss is lower for all three. No threshold relaxation or new checkpoint selection is justified after these results.

This audit recomputes all stored means but only freshly rescores sampled neural windows. It does not perform a second full-corpus neural pass or rebuild and fully rescore the statistical table. Statistical alignment is checked synthetically, and its stored aggregate is verified.

## Interpretation and resource limits

A per-layer attention span of 128 is **not** the same computation as resetting the whole two-layer model every 128 bytes. Through depth, this two-layer model can have an effective receptive field of up to 255 raw input positions. The reset comparator also has a cold-start history at the beginning of every scored suffix, whereas the local-512 condition already has preceding local context. Better loss does not establish retention of arbitrary 512-byte information, exact long-context recall, reasoning, or unlimited memory.

The intervention changes the attention graph while leaving weights fixed. It preserves absolute RoPE offsets rather than wrapping or resetting position numbering. The findings support the specified fixed-span intervention; they do not identify a unique cause of the original failure or establish how a properly long-context-trained Transformer would behave.

Stored decode cache is bounded, but full prefill still constructs a dense query/key attention problem with a local boolean mask. This implementation does **not** establish linear-memory or sparse-kernel prefill; temporary work can scale with full prefill length. The cache-allocation finding must not be presented as a bound on total process memory. The helper is exercised only on the fixed CPU float32 model; it is not audited for arbitrary device/dtype conversion or production request validation.

The second ordered FineWeb slice is new to this experiment, but remains a convenience sample, not a random held-out benchmark or contamination-free guarantee. Model and window choices were frozen before scoring this slice; the intervention itself was motivated by earlier v15 test observations. Treat this as targeted follow-on evidence, not proof of general architecture superiority.

## Timing coordination

This audit deliberately completed CPU work before the separate timing stage. The final sampled audit took 1.72 seconds; the expanded unit pass ended before the parent was notified at approximately 14:17:18 UTC that timing could begin. No latency benchmarks were executed by this auditor. The parent owns any subsequent timing artifact and must retain its definitions, cohort IDs, local p50/p95 limitations and unknown dollar allocation. Process-level ordering is not a guarantee of exclusive whole-host use.

## Reproduce

After regenerating the v16 snapshot, scores and local v14 checkpoints:

    python -m unittest -v test_local_attention
    V14_DIR=../v14 python audit_final.py

No paid compute, model training, external communications, production changes or repository publication occurred in this audit.

## Final report and timing-artifact review

The completed README, `summary.json`, `timings.json`, reproduction commands and postprocessing code were reviewed after timing. All 48 timing-summary cells were independently recomputed from the stored seed measurements and agree exactly. Quality means and sample standard deviations also match the stored results. Every stored paired Transformer window improves under local attention versus full attention, as the README states. The fixed reset-reference quality-gate outcomes remain unchanged.

At the 544-observed-byte end of the 512-prefix probe, compact local state is 130,064 bytes versus 557,056 baseline bytes, a **76.6515% retained-cache reduction**. The ratio of the reported median-of-seed-p50 end-to-end probes is **7.0583% slower** for local attention. The README correctly reports these as smaller persistent cache but no measured speedup or inference-dollar success. They are not peak-memory, whole-process, production-SLO or equal-quality cost claims.

All timing cohorts are the same eight rows, 100–107, across models and lengths. The parent reports timing began at 14:17:29 UTC, after the audit's CPU-completion notification. The timing artifact was finalized around 14:18:00 UTC; the 16-test final unit pass was finalized afterward, around 14:18:07. This ordering is consistent with no overlap between these audit/unit-test tasks and the separate benchmark. It does not establish exclusive use of the entire host. No latency measurements were repeated during this review.

The data counts in the README match the manifest: 407,643 original bytes and 255,009 capped bytes. Full-document SHA256 sets have no overlap with the prior v15 slice; this narrow duplicate check does not establish absence of partial/near duplicates or training contamination. The report retains that limitation.

Reproduction commands correctly point `V14_DIR` and `WIKITEXT_DATA` to the sibling v14 extraction/checkpoint directory when following its documented workflow. The audit requires regenerated local rows, window scores and v14 checkpoints; these are appropriately described as unredistributed intermediates. No further numerical or documentation blocker was found. Regenerate the artifact manifest after this audit update.
