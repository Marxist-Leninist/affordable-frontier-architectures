# Independent WikiText-2 byte-model audit

## Status

Implementation, data materialization, evaluation, and completed-result consistency audit passed. All recorded model/driver/protocol source hashes and train/validation/test text hashes independently match the final files. The saved selection and tuning records match results.json, and each selected rate is the minimum validation-loss candidate. All six final runs and six tuning runs have the declared seeds and training budget.

Mean final test bits per byte across seeds 1/2/3 is **3.4826730716 for the Transformer** and **3.3350588900 for the recurrent model**. Sample standard deviations across training seeds are 0.0392800463 and 0.0936635709, respectively. The recurrent loss is lower in each paired seed. The mean difference is 0.1476141816 bits per byte, or 4.2385% lower cross-entropy, corresponding to 9.7258% lower geometric-mean byte perplexity. These are not accuracy, intelligence, or cost improvements.

Both selected learning rates are 0.01, the upper endpoint of the tested grid. Therefore this experiment does not establish either architecture's tuned optimum. The recurrent run has visibly higher seed variation, and three seeds are too few for strong generalization claims. No test-driven model or seed selection was found in the driver.

## Data and provenance

The preparation script declares Salesforce/wikitext, configuration wikitext-2-raw-v1, pinned revision b08601e04326c79dfdd32d625aee71d232d685c3. Each manifest URL uses that immutable revision and official train/validation/test Parquet filename. I independently read all three downloaded Parquet files and verified that LF-joining their complete original text columns and UTF-8 encoding yields the local split files exactly. No row is filtered, normalized, or truncated. The downloaded file hashes and serialized text hashes match every manifest entry. This establishes consistency with the declared downloads; it is not a separate redownload or publisher-signature verification.

The fixed 256-token vocabulary represents every possible byte, with no OOV substitution or learned vocabulary. Int64 token arrays round-trip to the exact serialized bytes. The inserted LF row separators are part of the declared serialization. Consequently this is a byte-level evaluation on WikiText-2 raw data, not a canonical word-perplexity number or directly comparable to v7 bits per character.

- Train: 36,718 rows, 10,951,562 serialized bytes.
- Validation: 3,760 rows, 1,148,007 serialized bytes.
- Test: 4,358 rows, 1,292,013 serialized bytes.

## Model and numerical checks

The model implementation is byte-identical to v7. I reran sampled central finite differences for every parameter group, causal future-token and prefix-truncation isolation, batch probability/loss/gradient invariance, probability normalization, and a small gradient descent step. All passed. Largest sampled gradient absolute discrepancy was approximately 5.38e-11. These are strong bounded sanity checks, not a proof for all possible inputs.

Transformer: one pre-normalized causal four-head block, width 32, 4x tanh feed-forward layer, learned affine norms, final norm, fixed sinusoidal positions. Recurrent: vanilla tanh recurrence, width 51. Parameter counts are 29,024 and 28,764, respectively, so the recurrent model has approximately 0.896% fewer parameters. Its width is chosen arithmetically before training as the closest total count in the declared search range.

## Fairness and separation

Each run consumes 400 steps × 8 examples × 64 next-byte targets = 204,800 training-target presentations. The sampling RNG is independent of architecture initialization, so matching seeds receive matching windows. These are sampled presentations, not 204,800 distinct bytes. Every run draws from the official training split only. The start-index bound covers every valid 64-target window.

Both architectures receive three learning-rate candidates (0.001, 0.003, 0.01), one tuning seed 101 per candidate, full validation scoring, and then three separately initialized final runs with seeds 1/2/3. This is 614,400 tuning plus 614,400 final training presentations per architecture, or 1,228,800 total. No early stopping or best-seed selection occurs. The selected learning rates are written before final test scoring. Test bytes are loaded at startup, so the accurate description is selection before test scoring, not before any test-file access. Final validation scores are also computed but do not change the selected rates.

Equal presentations, approximate parameter counts, and equal trial counts do not establish equal FLOPs, equal wall time, optimal architecture-specific tuning, or commercially meaningful cost.

## Complete-segment evaluation and weighting

I independently probed evaluate() with synthetic target-dependent losses and verified exact agreement with the direct token-weighted mean over all intended positions, including the smaller final batch. Validation's last batch contains 64 targets; test's last batch contains 192. Target positions are contiguous and unique from 1 through the scored count. Context resets every 64 bytes for both models.

- Validation scores 1,147,968 targets; excludes 38 trailing targets and the initial byte, which has no preceding input.
- Test scores 1,291,968 targets; excludes 44 trailing targets and the initial byte.

Thus nearly all held-out next-byte positions are scored, rather than a small selected prefix. It is still not literally every byte, and it is not a sliding-window evaluation with the maximum 64-byte context available at every position. See independent_data_audit.json for arithmetic.

## Interpretation limits

These are tiny models given a small, fixed optimization budget on one dataset. Short reset context, initialization, positional encoding, optimizer defaults, and a narrow learning-rate grid can influence the ranking. Official dataset splits are respected, but this audit does not prove the absence of repeated natural-language passages across splits. The three training seeds measure optimization/initialization variability, not uncertainty over future datasets, tasks, or capabilities.

A lower test bits-per-byte score supports only the observed language-model cross-entropy comparison under this protocol. It does not establish intelligence superiority, reasoning quality, long-context performance, frontier equivalence, inference-price superiority, or a viable business case. CPU timings describe this NumPy implementation, not production GPU economics. No live service, paid computation, model driver, or training settings were changed by this audit.
