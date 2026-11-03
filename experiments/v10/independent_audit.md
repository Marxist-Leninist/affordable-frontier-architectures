# Independent multi-fact associative recall audit

## Outcome

Implementation, data, budget, selection, and checkpoint consistency checks passed. The scientific outcome is inconclusive for relative memory capacity: neither baseline reliably learned this harder task under the fixed optimization budget. Transformer accuracy is exactly 6.25% in every reported seed/cell; recurrent cell means range from 4.5573% to 7.6823%. Random 16-way guessing has 6.25% expected accuracy. These near-chance observations do not establish that either model lacks the capacity, that either family is superior, or that the earlier three-bit result extends to multi-fact recall. See independent_verified_summary.json for all cells and paired seeds.

## Task and leakage checks

Inputs contain 4, 8, or 16 distinct keys, each adjacent to an independently sampled 16-way value. Keys, values, fillers, and the query marker use disjoint token classes. The query key is revealed at the final position and occurs once among the earlier fact keys. This requires retrieving one queried four-bit value from a mapping whose values jointly contain 16, 32, or 64 random bits. It does not require generating every stored fact, and it is not natural-language understanding.

Random pair order and nonoverlapping random placements prevent a fixed-position query shortcut. Compressed placement coordinates correctly keep every complete key/value pair before the final marker/key query suffix. Separate structural and value RNG streams prevent target values from determining keys, placements, or filler identities. Target values can repeat across distinct keys, as independent sampling requires.

I independently checked the oracle, exact key/value pair counts, key uniqueness, query-payload indexing, and balanced query/target marginals on separate bounded samples. I also regenerated all nine actual test cells and verified every dictionary-oracle answer, all dataset hashes, and exact marginal balance: 16 occurrences of each query key and each target class among 256 cases. Query/target joint counts are random, not exactly balanced. A constant prediction is exactly 6.25% on these balanced cells; an arbitrary query-only predictor is only 6.25% in expectation over generation. No deterministic label leakage or ordering shortcut was found. This is not a proof against every possible statistical shortcut.

## Architecture and numerical checks

The two-block Transformer has an explicit task-specific inductive bias: its first causal attention block sees the current and previous token, enabling adjacent key/value binding. Its second causal attention block accesses all contextualized tokens. No absolute positional embedding is used, removing the previous experiment's unseen sinusoidal-position confound. This is a tailored comparator rather than a generic production Transformer.

Fast computation is exact for this architecture. The first block explicitly implements the same current/previous window as the dense causal reference. Only the second block's final query is needed because no later layer mixes token positions. It sees all earlier keys and values, including contextualized value tokens. Residual connections, normalization, feed-forward layers, and classification are tokenwise. This is not a general justification for dropping intermediate queries in deeper dense Transformers.

Independent float64 checks at lengths 5, 13, and 35 found maximum fast/full final-logit discrepancy 5.56e-16 and maximum all-parameter gradient discrepancy 4.89e-15. The run's tests additionally cover causal prefix isolation and directional finite-difference gradients, with an extra RNN gradient check. Training and saved model tensors are float32, not the float64 used for numerical checks. See independent_initial_checks.json, tests.json, and audit_extra.json.

Parameter counts are Transformer 18,992 and recurrent 18,928 (hidden width 110), a recurrent deficit of approximately 0.337%. The recurrent baseline is vanilla tanh, not a modern gated or state-space model. Transformer inference retains sequence-length-dependent representations; recurrent hidden state is fixed width. Matching parameters does not match inference-memory budgets.

## Training and selection

The frozen protocol contains 600 updates of batch size 8: 4,800 supervised queries and exactly 1,394,688 input-token presentations per run. I independently regenerated and verified all 600 minibatch digests and the full fixed schedule. Every model, learning rate, and seed sees the same batches; final seeds therefore measure initialization variability rather than fresh training-data variability.

Each architecture receives three equally budgeted learning-rate candidates and three final initialization seeds. Each architecture consumes 8,368,128 training-input presentations and 28,800 supervised queries including tuning. Across both architectures the totals are 16,736,256 input presentations and 57,600 queries. Equal presentations and approximately equal parameter counts do not establish equal FLOPs, equal wall time, or optimal architecture-specific training.

Validation uses separate seeded data across all nine trained length/fact-count cells. With equal cases per cell, the mean of cell cross-entropies equals pooled query cross-entropy. The selected rate is the lowest validation CE, with the declared lower-rate tie-break. I verified both selected minima and that final checkpoint configurations use the selected rates. Both rates are frozen before any test scoring in the driver. No test-driven retuning appears in this experiment. Short fixed training and narrow tuning are major limitations given the failure to learn.

## Checkpoints and reproducibility

All source hashes in the frozen protocol match current models.py, data.py, and run.py. All twelve tuning/final checkpoint files exist, their file SHA256s match results, and their loaded tensor hashes independently match both result records and checkpoint metadata. Checkpoints contain the complete model state and architecture configuration. The writer reloads each into a fresh model and asserts exact probe-logit equality. They do not preserve optimizer/RNG state for exact training continuation.

I verified all three final seeds, test dataset hashes, query counts, and accuracy values directly from saved predictions and independently regenerated labels. The saved test cells are shared across seeds, so their repetition must not be treated as independent new datasets. No original experiment training or full test-model evaluation was repeated during this audit, and no experiment source or configuration was modified. The separate bounded training diagnostic below is not part of the original results.

## Separate fixed-batch optimization diagnostic

After the frozen experiment completed, I ran a separately requested sanity check with fresh model seed 202, one generated length-64/four-fact batch of eight examples (data seed 202), 100 Adam updates at learning rate 0.003, and gradient norm clipping at 1. This uses no validation or test data and does not change any original selected rate, checkpoint, or result.

Transformer CE fell from 2.5684323 to 0.0065742; recurrent CE fell from 2.8132524 to 0.0006807. Both reached 100% accuracy on those same eight training examples. This demonstrates a working loss/gradient/update/label pathway and ability to overfit this tiny batch. It does not establish general associative recall: memorizing eight examples can exploit their incidental features or query mappings. Numerical correctness and fixed-batch overfitting do not guarantee successful learning on fresh examples. Full details and reproducible code are in independent_training_sanity.json and independent_training_sanity.py.
