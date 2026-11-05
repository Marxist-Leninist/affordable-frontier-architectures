# Independent v25 audit

## Scope and novelty

Source/data audit passes. Unlike v20 copy-predictor-only intervention, v25 feeds the changed distant prefix through every trained neural layer, recomputing its hidden states and final128 target distributions. Frozen checkpoints are inherited from v23 (context2048/reset512), v24 (mixed), and v19 (local/GRU). v25 has no optimizer, training, early stopping, or test-based model selection. Different inherited training budgets prevent treating all five arms as an equal-training architecture contest.

## Data and causal intervention

`python independent_check.py` independently verifies all frozen source/checkpoint/manifest hashes, exact fresh FineWeb rows900–999 and all original full/capped document hashes. There are100 documents,282587 capped bytes,68 eligible documents,107 windows,13696 targets per condition/seed.

For every window, deterministic cyclic next-eligible donor has a different full-document hash; its first1408 bytes actually change the input. Original/donor are equally long and use identical absolute positions. Bytes1408–2048 remain unchanged:513 real prior bytes before earliest target1921, followed by all128 target bytes. Scoring output indices1920–2047 against target indices1921–2048 is correctly shifted. Attention has a causal mask, recurrent processing is forward-only, and no target is exposed before its prediction. Suffix512 uses input indices1536–2047 and identical targets, hence385–512 prior bytes at the scored positions. Local pruning preserves absolute RoPE offsets and its two-layer finite receptive field.

Exact128-byte target spans have zero occurrences in the full WikiText training serialization,900 prior evaluation-document records from available v15–v24 snapshots, or other current documents; there are no duplicate scored target spans. These checks do not exclude shorter overlap, near duplicates, shared upstream sources, or semantic contamination. Snapshot byte hashes verify the downloaded artifact; the dataset revision header is recorded by the original fetch, not independently re-fetched here.

## Protocol integrity

Original protocol and initial failed test are retained. Initial/final protocols differ only in the test file hash, which matches retained original test text. Failure was optimized versus native FP32 logit difference1.2636184692382812e-5 against1e-5. The documented pre-score amendment changed that threshold to inherited1e-4; causal1e-5 and donor-local loss1e-10 stayed fixed. Four preflight tests pass. Artifact chronology is documented by the producer; local mtimes alone are not independent proof of chronology.

## Statistical and timing interpretation

Document-cluster bootstrap resamples recipient documents and averages seeds before estimating paired effects. It retains unequal window counts in the window-weighted mean. Its intervals are conditional on these fixed checkpoints, convenience documents, and donor mapping, not a random-corpus/model-seed population guarantee. Donors come from the same cohort, so this also does not estimate donor-selection uncertainty or all dependence induced by donor reuse. Artificial donor boundaries can cause disruption. Donor-minus-original sensitivity and original-minus-suffix usefulness must be reported separately: sensitivity alone is not evidence that distant context improves prediction.

Timing source measures original inputs only, batch1,8 documents,3 seeds,2 contexts,5 arms, one complete warmup and one measured run. It includes full256-way softmax after prefill and each of32 observed-byte updates (33 distributions). These are teacher-forced updates, not generated-token throughput. Assertions inside decode add validation overhead. Shared CPU-host wall times are descriptive, not a serving-capacity/cost measurement. State-byte counts describe live returned tensors, not process peak RSS or allocator memory. No training or timing is rerun by this audit.

## Completed numerical and artifact checks

After producer confirmed timing completion, `PYTHONPATH=/tmp/frontier-v10-packages python independent_tensor_check.py` passed. Default Python initially lacked torch; that environment failure is retained separately. Across all15 checkpoints on one real document each: weights are FP32, causal first-target errors are zero, native versus optimized maximum logit error is1.52587890625e-5 (below amended1e-4), and independent full-head versus stored loss maximum error is8.588081996663277e-6. All45 saved means and10 paired mean effects reconstruct. `independent_statistics_check.py` reproduces all10 bootstrap intervals within1e-12. All240 unique timing records have expected actual weight/state bytes, including FP32 full-attention growth, fixed local cache with int64 position counters, and fixed GRU state. This checks logged byte fields against architecture and real tensors; it does not independently reproduce timing durations.

Local donor invariance is exactly zero for every saved seed/window/target. Local original-versus-suffix is only approximately invariant (mean -1.7473e-7 bpb; maximum target difference0.00071010), consistent with changed FP32 RoPE-coordinate arithmetic. GRU is not exact invariant (maximum donor target difference0.15678 bpb).

For context2048, donor-minus-original is+0.00011670 bpb (conditional95% interval[-0.00670082,+0.00643668]); original-minus-suffix is+0.08815528[+0.07696853,+0.09937417]. Mixed and reset512 donor effects likewise have intervals spanning zero, while their suffix advantages are0.41096677 and1.34440982 bpb. Individual full-Transformer target changes reach9.65–13.10 bpb: near-zero average donor effect is not content invariance or proof of length-only behavior. These results provide no clear average original-prefix benefit over these unrelated donors, and full original context is worse than suffix512 on this sample. They do not establish equivalence, generalized irrelevance of distant text, useful long-context reasoning, frontier parity, or viable production economics.

The stale inherited data card was corrected by the producer during reporting. No remaining correctness blocker found in this bounded audit. Raw text/checkpoints remain scratch-only, excluded from external deliverables.
