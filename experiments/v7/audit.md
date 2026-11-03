# Independent implementation and experiment audit

## Verified corrected result

The corrected run reports mean held-out bits per character of **3.3768911012 for the Transformer** and **3.1767359018 for the vanilla recurrent model**, across seeds 1, 2, and 3. Sample standard deviations across training seeds are 0.0167977423 and 0.0116780085, respectively. These describe initialization/training variation, not confidence intervals for new corpora.

The recurrent model is lower in every paired seed. Mean difference is 0.2001551994 bits per character, a 5.9272% reduction in cross-entropy, corresponding to a 12.9543% reduction in geometric-mean perplexity. These are not accuracy or intelligence percentages.

This supports only the narrow result that this small vanilla recurrent baseline outperformed this particular normalized Transformer under this corpus, training, and evaluation protocol. It establishes neither general recurrent superiority nor frontier-level intelligence.

## Corpus materialization correction

The first materialized input had one extra trailing newline: 1,115,395 bytes and SHA256 4b47fa2c48873c9585da2030c733f9935b7c971c640d5722ed75d175d9939a23. Its Git blob did not match the predeclared source blob. The audit caught this provenance failure. That preliminary run was archived under superseded_materialization and superseded, not selected or combined with corrected results.

Removing the accidental extra newline restored the exact source bytes, and the complete experiment was rerun with the unchanged predeclared protocol. Verified final input: 1,115,394 bytes; SHA256 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed; Git blob SHA1 7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9. Both the local file and results.json agree. The source hashes recorded for models.py, run.py, and protocol.json match their current contents. All three recorded split hashes independently match corrected corpus slices.

## Implementation checks

Transformer: one normalized causal four-head block, width 32, learned embeddings/output, affine pre-LayerNorm attention and 4x tanh FFN, residual connections, final affine LayerNorm, fixed sinusoidal positions. Recurrent: vanilla tanh recurrence, width 80, learned embeddings/output and recurrent bias. Analytical gradients were checked with sampled central finite differences for every parameter group. Maximum absolute discrepancy was below 5.38e-11. Both models passed causal prefix changes and truncation, batch probability/loss/gradient invariance, and single-step descent. See model_tests.json and test_models.py.

## Fairness and arithmetic

- Parameters: Transformer 16,800 versus recurrent 16,880, a 0.4762% recurrent surplus. Width was chosen arithmetically before training.
- Each run uses 400 steps x batch 8 x sequence 64 = 204,800 target-token presentations, not necessarily unique tokens. Both architectures receive identical sampled start positions for each seed.
- Each architecture has three equally budgeted learning-rate trials with tuning seed 101. Validation selects the rate. Selection is written before final-seed test scoring; test losses do not select learning rates or seeds. Test text is loaded and checked for vocabulary coverage at startup, but its losses are not used for selection.
- Each architecture then has three final runs, seeds 1/2/3, with all results retained. Both selected rates are 0.01.
- Corrected split lengths are 892,315 training, 111,475 validation, and 111,476 test characters, with 64-character boundary gaps discarded.
- Evaluation scores 32 equal-size batches of 512 disjoint target positions: exactly 16,384 targets per split. Mean batch negative log likelihood divided by log(2) correctly yields bits per character. Teacher-forced target indexing and causal reset every 64-character segment are consistent.
- Among the 256 evaluated 64-character contexts in each held-out split, none is an exact 64-character substring of training. This narrow duplicate check is not proof against shorter repeated phrases or literary similarity.

## Limitations and minor findings

Both chosen rates hit the upper endpoint of the three-rate grid; an optimum is not established. Four hundred steps are a small optimization budget. Architecture-specific optimization choices, initialization, positional representations, and depth were not exhaustively tuned. Equal tokens and parameters do not equal FLOPs or wall time; CPU timings are descriptive of this implementation only.

Only the first 16,384 targets, about 14.7% of each held-out split, are evaluated. Contiguous splits can differ by work, scene, speaker, and style. Context is reset at 64 characters; this is not a long-context or generation-quality test. The single corpus and three training seeds provide no broad architecture ranking, intelligence metric, business case, or frontier equivalence claim.

The training start-index bound excludes the last two otherwise valid start positions. This affects both architectures identically and is negligible for this comparison, but can be corrected in a separately declared future protocol. No running experiment or source was mutated during this audit.
