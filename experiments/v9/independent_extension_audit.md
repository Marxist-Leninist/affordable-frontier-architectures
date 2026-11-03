# Independent 2k/4k synthetic recall extension audit

## Implementation and data

The final-query helper is mathematically equivalent to the original one-block Transformer forward pass at the final token. The first normalization and Q/K/V projections are tokenwise. All keys are causally visible to the last query, so omitting other queries and the final-query causal mask changes no result. The attention output, residual, second normalization, feed-forward transformation, and final normalization are also tokenwise. Absolute sinusoidal positions remain exactly 0 through length minus one. This reduction does not automatically apply to arbitrary multi-layer Transformers.

The recurrent helper executes the identical zero-initialized tanh recurrence and output projection, retaining only current state rather than all time-step activations. Model parameters and intermediate computation remain float64. These are inference-only storage reductions, not training modifications.

Independent checks use a different random parameter seed (913), arbitrary vocabulary inputs, and lengths 1, 2, 7, 31, 65, and 129. Final probabilities agree with the original full forward pass to maximum absolute error 2.50e-16 for the Transformer and 1.39e-17 for the recurrent model. See independent_extension_parity.json. The extension also requires full-reference probability and masked-query CE parity at lengths 8, 64, 128, and 256 for random models and every reconstructed final model.

I independently generated the two evaluation sets, without running full-length inference, and verified 1,024 examples at each length, exactly 128 of each target class 0 through 7, the target at the first input position, query token 16 at the final position, distractors only in classes 8 through 15, and all nonfinal labels masked. The deterministic input hashes are saved in independent_extension_data.json. Every model must use these same arrays. The 2,048 and 4,096 lengths count synthetic vocabulary symbols, not bytes or text tokens from WikiText.

## No retuning and integrity

The extension reads all six original v9 final runs, preserves seeds 1/2/3 and their selected learning rates, and deterministically reconstructs each model. It asserts exact concatenated-weight SHA256 equality before evaluation. Reconstruction training is necessary because original checkpoints were not saved; it is not a new tuning trial and cannot replace a failed hash match. I independently verified that the original v9 recorded source hashes still match their files.

## Metric and interpretation

Accuracy uses full 17-way argmax; CE uses stable full 17-way log-sum-exp and averages one supervised query per example. The 12.5% reference is chance when choosing among the eight possible target classes, not uniform choice across all 17 output classes. The payload is one eight-valued symbol, only three bits of target information. Successful retention across many distractors is not evidence of remembering thousands of arbitrary facts or text tokens, broad intelligence superiority, or production cost superiority.

Lengths greatly exceed training's maximum 64 symbols. The Transformer must extrapolate both context length and previously unseen absolute sinusoidal positions; failure therefore does not isolate an inherent memory-capacity limit of Transformers. Three seeds and one synthetic task do not support broad model-family conclusions.

Reported cache figures name selected arrays and deliberately exclude temporary normalization/BLAS allocations, model weights, Python overhead, and actual process RSS. The recurrent model's state cache is independent of sequence length, but input/label arrays still scale with length. These figures are not measured peak memory and should not be used as a complete memory or speed comparison. CPU wall times do not measure deployment economics.

## Completed-result verification

Completed results verified independently: all six original weight hashes match; model seeds and learning rates match v9; source/protocol/original-result hashes match current files; all 12 evaluations match independently regenerated input hashes and balanced class counts; accuracy arithmetic and aggregate means are correct. The saved parity report includes all six trained models.

- transformer, 2048 symbols: mean accuracy 0.1250000000; mean CE 7.7455635272 nats.
- transformer, 4096 symbols: mean accuracy 0.1250000000; mean CE 8.1508577569 nats.
- recurrent, 2048 symbols: mean accuracy 1.0000000000; mean CE 0.0018700870 nats.
- recurrent, 4096 symbols: mean accuracy 1.0000000000; mean CE 0.0029713538 nats.

No training or full 2k/4k inference was duplicated by this audit.
