# GPT-2 versus Qwen3 tokenizer: controlled small-Transformer pilot

## What was tested
Actual Transformer training from scratch with two released tokenizers, not incompatible token IDs swapped into a pretrained model. All runs use the same 103,680-parameter Transformer trunk and the same training text source. Three seeds and two budget views are reported. A third arm pads GPT-2's vocabulary to Qwen3's output size as an artificial parameter/cost control.

**This is a narrow, undertrained English language-modelling pilot. It does not measure general intelligence or establish what a 1B or frontier model would do.**

## Main results
Lower canonical-path bits per UTF-8 byte is better. Means ± sample standard deviation across three training seeds. These are not confidence intervals for broad language performance.

| Tokenizer arm | Total parameters | Equal 512 raw segments | ~240 training CPU seconds |
|---|---:|---:|---:|
| gpt2 | 3,320,128 | 2.536075 ± 0.008480 | 2.421633 ± 0.005355 |
| qwen3 | 9,810,496 | 2.662909 ± 0.011037 | 2.636355 ± 0.017591 |
| gpt2_padded | 9,810,496 | 2.621860 ± 0.009347 | 2.592206 ± 0.011910 |

## Interpretation
In this specific pilot, the Qwen3 tokenizer did not improve held-out text likelihood over GPT-2. Native vocabulary size changes output compute and total parameter count; the padded control reduces that disparity but introduces unused output classes. It must not be represented as a normal deployment or a pure causal isolation of tokenization.

Do not generalize this result to all tokenizers, all languages, or well-trained large models. Shared optimizer settings were not separately tuned. Newer or more multilingual tokenizers need not compress this particular English subset better. General intelligence was not evaluated.

## Resource accounting
- gpt2: training CPU seconds [240.20595327700005, 240.08824674000024, 240.0163248349997]; steps [841, 833, 814]; raw bytes seen [529130, 525027, 511154]; tokens seen [115902, 115328, 112045].
- qwen3: training CPU seconds [240.43593542299985, 241.07283851500046, 240.10821365900006]; steps [194, 190, 199]; raw bytes seen [122208, 120095, 124971]; tokens seen [28563, 27959, 28820].
- gpt2_padded: training CPU seconds [240.73554393900017, 240.09866736200001, 241.01542536499986]; steps [204, 203, 211]; raw bytes seen [128456, 128344, 132609]; tokens seen [28350, 28221, 28922].

Training CPU time is measured process CPU time, not exact FLOPs. Endpoints overshoot the nominal 240 seconds by their final optimizer step. Setup, tokenization, validation and file IO are outside that allowance; whole-process CPU consumption was not recorded. One FP32 CPU thread, low scheduling priority, existing Intel Xeon Skylake host; no GPU, rentals, purchases or live service changes.

## Broader held-out sampling check (exploratory)
Added after the first two main seeds because the initial first-N evaluation can cluster by article. No training or hyperparameter changes were made. This evaluation draws 256 different eligible fragments using seed 1234 across the full official test split and excludes all retained main train/validation/test exact fragments. It is supplementary, not preregistered.

- gpt2: mean 2.425835, standard deviation 0.010897; seeds [2.414724205910031, 2.426273794774819, 2.4365056718231464].
- qwen3: mean 2.676030, standard deviation 0.005097; seeds [2.6784986908020674, 2.670168744845263, 2.679423520010751].
- gpt2_padded: mean 2.610475, standard deviation 0.006484; seeds [2.6175247693654047, 2.609132745702816, 2.6047678326118606].

## Full-corpus unigram reference
These add-one smoothed unigram models see all 10,000 retained training fragments, unlike the compute-limited Transformer endpoints. They are contextual references, not a matched superiority claim.

- gpt2: test canonical bits/byte 2.434700.
- qwen3: test canonical bits/byte 2.552725.
- gpt2_padded: test canonical bits/byte 2.508107.

## Tokenizers and provenance
- [GPT-2](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e): 50,257 entries, upstream MIT.
- [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/tree/8efa61729e24bd65b1d152b5ab5409052aa80e65): 151,669 entries, upstream Apache-2.0.
- Qwen3-0.6B and 235B tokenizer.json **and tokenizer_config.json are byte-identical** at the recorded revisions. Thus the same tokenizer is used by very different model sizes; model capability cannot be assigned to the tokenizer alone.
- [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3), Wikipedia-derived, upstream card CC-BY-SA-3.0/GFDL. Raw source data and tokenizer files are not redistributed in this report; preparation fetches exact revisions.

## Verification and limits
- Nine checkpoint save/reload likelihood checks passed, with weight hashes recorded.
- Causal future-token invariance assertion passed before every run.
- Recalculated per-example losses match all aggregate bits/byte scores.
- Same raw bytes and 128 updates at the equal-data endpoint, all three arms and seeds.
- Exact retained-segment overlap removed across splits; this is not a complete near-duplicate contamination audit.
- Same 160-character segments and 63-token maximum under both tokenizers; this selection favors short mutually tokenizable English fragments.
- EOS start context is supplied; EOS target is not scored. Canonical token likelihood is not marginal probability summed over every segmentation.
- Three training seeds do not establish statistical equivalence. This study neither proves negligible tokenizer effects nor broad modern-tokenizer gains.

## Reproduction
Use Python 3.12.3 and requirements.txt in a fresh environment. Run bash reproduce.sh. CPU-time endpoints depend on hardware/load; the 512-fragment endpoint is the deterministic fixed-data reference. See PROTOCOL.md, REVIEW.md, runtime_notes.json and manifest.json. Main local weights were preserved; this report package publishes code and results, not production weights.
