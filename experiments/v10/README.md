# v10: harder multi-fact recall exposes a training-adequacy limit

2026-11-03. This experiment follows v9's easy three-bit retention result with a harder key–value retrieval task and a stronger, task-specific Transformer comparator. **Neither model learned robust retrieval under the initial fixed budget. No architectural memory winner is established.**

## Task and stronger control

Each sequence contains4,8 or16 distinct key–value pairs, randomly ordered and positioned among distractors. Keys range0..15; independently random values range0..15 and are represented by tokens16..31. Fillers32..39 contain no answers. Only the final query marker40 and key reveal which fact must be returned. The potential map has16,32 or64bits of random value information, compared with only3bits in v9. Query keys and target values are balanced in each evaluation cell. Chance among16 answer classes is6.25%.

The Transformer has two pre-normalized residual blocks: local causal attention over the previous/current token for binding adjacent key/value pairs, followed by global content attention. Width32, four heads, GELU FFN width64; no absolute positional embeddings. This avoids the earlier absolute-position extrapolation issue and gives a task-tailored local/global attention control, not a frontier or typical production Transformer. For final-query-only supervision, calculating only the final query in the global block is exactly equivalent to full causal evaluation. Earlier local outputs still cover every input token.

The comparator is a vanilla tanh RNN with hidden110 and embedding32. Parameter counts18,992 Transformer versus18,928 RNN (0.337% fewer). This is not a test of all recurrent systems, GRUs, LSTMs, Mamba or the user's large language models.

## Frozen initial protocol

-600updates, batch8, shared fixed schedule of lengths128/256/512 and fact counts4/8/16. Exactly1,394,688 input tokens per run. Identical600 minibatches across architectures, learning rates and initialization seeds; this controls data variation but means final seeds measure initialization variation, not independent data streams.
-Adam defaults, clipping1, float32, deterministic PyTorch2.6.0+cpu, one thread; no dropout or weight decay.
-Three tuning LRs .001/.003/.01, seed101. Lowest pooled independent validation CE selects .01 Transformer and .001 RNN. Both selections frozen before any final test scoring.
-Final seeds1/2/3. Each test cell has256 balanced cases, for4/8/16 facts at lengths512,2048,4096. Length512 is within training;2048/4096 are extrapolation. These are synthetic vocabulary symbols, not natural-language words or document tokens.

## Outcome

Transformer accuracy is exactly6.25% at every final seed/test cell (constant-class collapse). RNN mean cell accuracies range4.56–7.68%, also around chance. `SUMMARY.md` gives every cell and seed ranges; `results.json` preserves all tuning and final metrics and predictions. Because neither model succeeds even at trained length512, these results cannot rank long-context memory capacity. Failure to learn under this optimization budget is not proof an architecture cannot solve the task.

## Checks and checkpoint preservation

Dictionary oracle answers100% in every checked cell. Constant-class guessing is exactly6.25% on balanced targets. Query-fact positions span all chronological ranks. Independent checks verify distinct facts, valid nonoverlapping pairs, no label in fillers, and independent structural/value RNG streams.

Efficient/full-reference Transformer logits and parameter gradients agree at float64 numerical precision. Causal-prefix invariance and directional finite differences pass; an independent RNN directional check also passes. A separate fresh-model training sanity check overfits eight fixed L64/K4 examples to100% for both models. That rules out some gross gradient/label-flow failures but is memorization, not evidence of retrieval generalization.

All12 tuning/final checkpoints were saved before process exit, with file/tensor hashes and exact fresh-model reload prediction parity. `checkpoint_manifest.json` records them; binary checkpoints are separate artifacts rather than embedded in this source repository. Independent audit verifies every checkpoint, frozen source hash,600 input-batch hashes, token count, test dataset hash and prediction-derived accuracy.

## Reproduce and next step

Install ordinary CPU PyTorch2.6.0 and NumPy in an isolated environment, then run from this directory:

    python run.py

This reruns the entire frozen experiment and writes local checkpoints. No GPU, paid service, production process or API inference was used. CPU runtime is not a deployment-cost estimate.

A separately declared validation-only continuation tests whether more updates resolve the training failure. Any curriculum or further model selection must remain separate and use validation only; preserve this failed initial experiment rather than replacing it with later successes. Long-context comparisons require demonstrable trained-context mastery before a broad retention claim. No frontier capability, general intelligence, API-price or $500 subscription feasibility claim follows from v10.
