# Delayed first-symbol recall: preregistered CPU experiment

Prepared and tested before training; all12 protocol runs are now complete. See `results.json` and `audit.md`.
Protocol: `protocol.json`. Preparation timestamp and hashes: `prepared_manifest.json`.

## Results

Across three final seeds, both models score100% at every trained sequence length8/16/32/64. At unseen length128, recurrent scores100% for all seeds and Transformer averages95.61%. At unseen256, recurrent scores100% for all seeds and Transformer averages58.82% (37.50/62.60/76.37%). Selected learning rates are Transformer.01 and recurrent.003; all tuning and individual seed/length results are preserved.

This is positive evidence for this recurrent model's extrapolation on one deliberately simple task, not proof it retains long text better. The stored answer is only3bits (one of8 symbols), regardless of distractor count. All distractors come from a separate symbol class, making the target easy to identify. The Transformer sees unfamiliar positions beyond64, so positional/context extrapolation is confounded with retrieval. There is no advantage at the trained lengths, and neither model represents typical frontier systems.

## Requested 2k/4k extension

At2048 and4096 synthetic vocabulary symbols, every recurrent seed answers all1,024 balanced test queries correctly; every Transformer seed answers128/1,024 (12.5%, chance among the8 target classes). These are vocabulary symbols, not bytes, words or natural-language tokens. All original model weight hashes match exactly after deterministic reconstruction; no model was retuned or trained on longer sequences.

`extrapolation_2k4k_results.json` contains every seed and length, exact input/weight hashes, query counts, CE, timings and named-array memory estimates. `extrapolation_2k4k_protocol.json` was written before the extension. The one-block Transformer computes only the last query with all keys/values, mathematically matching the original final output without quadratic full-sequence outputs. State-only recurrence is likewise equivalent. Probability/CE parity against full reference passed for random and all trained models at8/64/128/256, max probability error3.33e-16; a separate reviewer checked additional arbitrary lengths and all results.

This is a strong pass on **this three-bit retention task**, not a demonstration of retaining2048/4096-token documents, many facts, arbitrary retrieval, or superiority to typical Transformers. Sinusoidal position and context extrapolation beyond the training maximum64 remain confounds. Full details and independent verification are in `extrapolation_2k4k_audit.md` and `independent_extension_audit.md`.

The target is the sequence's first token (one of eight classes). All intervening tokens are independently drawn distractors from disjoint classes; the final token is the query. This deliberately easy test measures whether a model can preserve/retrieve one symbol, not broad long-context ability, intelligence, practical language modeling, or all recurrent/Transformer architectures.

Models are copies of v8's normalized four-head Transformer and vanilla tanh RNN. The only model-code change is masked loss: targets -1 receive exactly zero direct loss/logit gradient, normalization is over supervised queries, and an all-masked batch raises an error. Gradients through earlier hidden states remain intact, enabling credit assignment for final recall.

Transformer width32 has13,728 parameters; nearest integer RNN width101 has13,736, a difference of8 (0.0583%). Input and supervised-target budgets are exactly paired, not wall-clock or FLOPs. Each model independently chooses among three LRs with seed101 and400steps. Final paired seeds1/2/3 each use400steps. Initialization distributions are inherited from v8, with identical seed numbers but necessarily different parameter shapes. There are no architecture-specific extra trials.

Training sequence lengths include the initial symbol and final query. Each block of four steps includes one batch at each of8/16/32/64 in randomly permuted order. Each batch has16 independent targets uniform over0..7. Each run therefore sees192,000 input tokens and6,400 supervised queries. Full experiment:12 runs,2,304,000 input tokens,76,800 queries.

Validation and held-out test cases each have128 instances per target at each length. Trained lengths8/16/32/64 are distinct from extrapolation lengths128/256. Report CE over the full17-way output and argmax accuracy over all17 outputs;12.5% chance refers to uniform guessing over the eight possible target classes, not uniform17-way output.

Run tests (no training):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python test_models.py

Run the full fixed training protocol from this directory:

    python run.py > run.log 2>&1

The test artifact `model_tests.json` covers finite-difference gradients in every parameter tensor under query-only masking, exact zero direct nonquery gradients, correct query CE, batch invariance, all-masked rejection, baseline causal tests, deterministic paired data, equal exact token counts, and balanced evaluation targets.
