# Independent audit of v23

Status: complete, 2026-11-05 11:19 UTC. Qualified pass for the bounded context-training finding. No training or performance benchmark was rerun by this reviewer.

## Design and implementation

The principal comparison is a valid bounded context/reset intervention within the same Transformer, not an architecture comparison. Both arms reload the same v19 `transformer_seedN_adapt500.pt`, create fresh identically configured AdamW optimizers, consume the same frozen 250-entry ledger, perform 250 updates, and present 512,000 next-byte targets per seed. Full-context inputs/targets are independently reshaped into four 512-length rows in the reset arm. Flattened target order is preserved, including targets across the original segment boundaries. This isolates accessible context and sequence reset, not FLOPs, elapsed time, convergence or a uniquely causal position-only effect.

v19 continued at512 versus reset128. v23's2048 versus reset512 continuation is a substantive new test of the long-context distribution mismatch left open by v19. The model has two causal RoPE-attention/SwiGLU/RMSNorm blocks. Full attention uses causal SDPA; optimized tail scoring preserves the same Transformer computation and only omits unused vocabulary projections. Loss alignment is correct: the last128 input logits predict the last128 bytes of each2049-byte raw window. Nested512 inputs retain those exact targets, giving effective target contexts385–512 versus1921–2048. Reset512 training starts independent rows at position zero, as intended. Local reference pruning retains the required128+2×127=382 inputs and absolute RoPE offsets.

Frozen local and GRU references have no additional training, so their comparisons with newly trained Transformer arms are descriptive, not matched-budget architecture rankings. Fixed training schedule and all seeds/arms are reported; source shows no loss-driven early stopping, tuning or checkpoint selection. The predeclared0.02bpb screen is not an equivalence proof.

## Independently checked data and artifacts

`independent_checks.json` records fresh regeneration of the exact NumPy ledger, source-hash agreement, the WikiText training hash, and reconstruction of every evaluation item. Real pinned WikiText training and fresh pinned FineWeb rows700–799 are used. There are100 sampled documents,66 eligible documents,105 windows and13,440 scored byte targets per condition/seed. No capped-document, full-window or128-byte target-span duplicates were found. No full-document hash matched prior available v15–v22 manifests, and no complete128-byte scored target span occurred in the training serialization. These checks do not establish absence of shorter passage overlap, paraphrases or near duplicates. The convenience/length-filtered cohort is not representative of broader language ability.

The card states the WikiText license inconsistency explicitly: metadata names CC-BY-SA3.0 while prose names4.0. FineWeb dataset-card licensing does not extinguish underlying page rights. Corpus text and checkpoints are scratch-only and must not enter the report/archive. Viewer downloads check x-revision, with snapshot and per-document hashes retained.

Existing six unit tests passed, including ledger identity, initialization, causal masking and optimized-logit agreement. Their nested-target test alone is weak (it largely compares equivalent slices); independent raw-window/source review supplies the substantive alignment check. No source edits were made by this reviewer.

## Accounting and limits

The independent data/hash script used0.233 CPU seconds measured after its imports, not an inclusive OS process total. Main `process_time` begins after imports and distinguishes training from total training/evaluation time; it does not measure GPU/service dollars. Per-arm training times exclude evaluation and serialization. The main hard CPU limit820/825 seconds includes process initialization, while the internal770-second safety check only runs during training. Inclusive all-process accounting must therefore be supplied separately or explicitly qualified; historical v14/v19 training cost is additional, not erased by continuation.

The bootstrap resamples whole documents and retains their windows, averages seed effects first, and targets the window-weighted metric. It is conditional on this fixed corpus and does not capture general training-seed uncertainty. Shared seeds/corpus are not independent dataset replications. No new inference speed, commercial-token conversion, full-system serving costs, frontier-quality parity or AGI result is established. Prior economics remain conditional gates, not outcomes of this experiment.


## Final results independently reconstructed

All30 loss rows, six checkpoint hashes and six complete250-update/512,000-target training records pass. All nine reported comparison means and document-bootstrap intervals were independently reconstructed from the per-target arrays, with maximum discrepancy4.44e-16bpb. See `independent_final_checks.json` (0.104 additional measured CPU seconds after imports). The supplied final-model audit reports zero causal future-suffix and optimized/full-head discrepancy at2048 inputs.

The context2048 arm improves full2048 loss from4.357838 to3.563760bpb and beats reset512 (4.572721bpb) by1.008961bpb; paired document interval for the signed difference is[-1.043654,-0.974919]. Every seed has the same effect direction. However, its own nested512 loss is3.475771bpb, so longer context still costs0.087989bpb (interval[0.075031,0.101720]). At nested512, the reset-trained arm wins by0.245557bpb. This supports context-distribution adaptation and a short-context tradeoff, not demonstrated beneficial long-range information use. Frozen local/GRU remain numerically better on this cohort but cannot support matched-budget superiority claims.

Measured main CPU is249.019 seconds, of which183.336 seconds is training. Independently OS-accounted summary/audit/report subprocesses add1.856 seconds including their imports. Earlier preparation/unit-test/import overhead and main startup were not fully instrumented, so no exact inclusive total is certified; reported measured work is comfortably below the900-second allowance. The hard main-process limit is a separate safeguard, not a measured total. A failed `/usr/bin/time` auxiliary launcher occurred before research actions and is retained; it is not a failed training arm.

Conclusion: the fixed target ledger, causal implementation, aligned held-out targets and reconstructed aggregates support the narrow empirical finding. Publication is appropriate with the stated unequal-compute, conditional-bootstrap, contamination, licensing and accounting qualifications. No blocker remains for reporting this bounded experiment. No new inference/cost or frontier claim should be attached.
