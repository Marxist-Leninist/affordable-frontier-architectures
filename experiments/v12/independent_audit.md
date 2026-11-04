# Independent implementation and results audit

2026-11-04. Read-only review of experimental code and data, with separate audit artifacts. No experiment source, protocol, checkpoint or recorded result was modified. This audit is local implementation validation, not external replication or a frontier-quality certification.

## Findings

No substantive implementation or arithmetic error found. The observed ranking is robust across the three tested neural initializations under this particular short-context, low-exposure protocol: statistical suffix model, then GRU, then Transformer (lower bits per byte is better).

- Transformer: mean test 2.725890302 BPB, sample SD 0.006106213.
- GRU: mean test 2.483292125 BPB, sample SD 0.009811533.
- Non-neural four-context-byte model (conventional 5-gram): 2.335172788 BPB.
- Paired Transformer minus GRU differences: 0.237697845, 0.257725324, 0.232371362 BPB.
- GRU cross-entropy reduction versus Transformer is 8.8998%; count model reduction versus GRU is 5.9646%. Neither is an accuracy or intelligence percentage.

## Checks completed

1. Frozen neural source hashes match protocol.json. Actual train/validation/test text byte lengths and SHA-256 hashes match the pinned manifest. Preparation preserves UTF-8 bytes of LF-joined rows. No fitting on validation/test data is present in reviewed code. This does not establish absence of naturally duplicated text across original dataset splits.
2. All neural initializations and the count model receive identical 1,024,000 target presentations, sampled from training only. The window ledger covers 977,918 distinct target positions, or 8.929485% of training targets. Repeated seeds reuse this ledger and therefore measure initialization variability, not dataset or training-sample variability.
3. Parameter counts are 131,904 versus 132,160 (0.1941% difference). Architectural FFN widths differ to match total parameters. Three learning-rate candidates and fresh final seeds are used symmetrically. Test scores do not choose neural rates. Both rates select the highest candidate, so optimal tuning or training maturity is not established.
4. Causal attention, rotary offsets, incremental masks and GRU state passing inspected. Independent full-sequence versus irregular-chunk forward/backward test passed for both models: maximum logit discrepancy 5.96e-7 and gradient discrepancy 2.24e-8. This supplements provided causality, finite/nonzero-gradient, batch-parity and reload tests.
5. Exact scoring boundaries agree across models: 1,147,904 validation and 1,291,904 test targets, resets every 128 bytes, with 102/108 trailing targets excluded. Metrics are custom serialization byte BPB, not standard WikiText word perplexity.
6. Count-model training uses the same windows and targets; recursive interpolation is normalized and fallback behavior is sound. Four-context-byte candidate selected only on validation. Shared fitting of three candidate orders is disclosed. Its 2,820,855-byte serialized artifact is larger than neural checkpoints (534,152 and 535,928 bytes). These sizes are not total runtime memory.
7. Summary means and paired differences independently recalculated. All six neural checkpoints reproduced their full-test scores within 1.01e-08 BPB; the reloaded statistical checkpoint reproduced its test score exactly and contains exactly 1,024,000 target counts. All seven checkpoint hashes matched. Full-test checkpoint reload and rescoring results are recorded separately in independent_audit_scores.json. Neural rescore uses batch 16 and float64 accumulation of individual float32 cross-entropy losses rather than the original batch 32 summed reductions. It uses the same model implementation, so is not a separately implemented model replication.
8. Cache storage matches expected shapes: batch-one GRU state is 512 bytes; Transformer state is 1,024 times context length bytes. This excludes model weights, allocator overhead and workspaces.
9. Economics ledger arithmetic checked: illustrative target bill $0.905; stated subscription scenarios are assumptions, not measured service costs or financial advice. Byte-model CPU rates cannot validate provider subword-token prices.

## Limits on conclusions

- This is a tiny, sample-limited CPU experiment. It does not test reasoning, tool use, coding, language understanding, or general intelligence. Statistical-model success is a warning against attributing gains to advanced reasoning.
- Three seeds on one fixed data ledger are not broad statistical evidence. No cross-corpus replication or evidence of fully trained performance is supplied.
- Parameters and exposures are matched, not training FLOPs, compute time, memory or attained quality. GRU training averages 50.93 seconds versus 25.83 seconds for Transformer; GRU prefill is slower. Faster cached decode is not a blanket speed or cost win.
- Contexts 512/2048 measure timing only; quality is assessed only with 128-byte reset windows. Constant-size recurrence discards information and cannot be called quality-equivalent long-context memory.
- Transformer cache appends copy with torch.cat. This is not an optimized paged/preallocated serving implementation. Single-process CPU local timing, no sampling, and small samples cannot establish production p95 latency, GPU economics or service SLOs.
- Statistical model uses greater storage, specialized short-context counts and shared candidate fitting. Its timing is a Python full-distribution implementation observation, not an asymptotic or production comparison.
- The non-neural control was added during neural pilots, not frozen in the original neural hash list. Treat this as exploratory research, not a fully preregistered confirmatory experiment.

The strongest defensible conclusion is that under equal small training exposure and near-equal neural parameter counts, this GRU predicts held-out bytes better than this Transformer, while a larger-storage short-context statistical model predicts them better still. No replacement architecture, frontier-quality equivalence, AGI mechanism or profitable serving plan has been established.
