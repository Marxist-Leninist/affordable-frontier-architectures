# Pretrained Pythia to DiffusionBlocks: small conversion pilot

**Result: executable conversion, but not quality-preserving at this budget.** This is a real fine-tuning experiment starting from published pretrained weights, not a model trained from scratch. It does not establish that converting a large language model works, and it does not refute the paper's proposed future direction.

## Sources and licenses

- Paper: [DiffusionBlocks, arXiv:2506.14202v4](https://arxiv.org/abs/2506.14202), particularly Sections 3 and 5.4 and Appendices B, C, E.4. [PDF](https://arxiv.org/pdf/2506.14202).
- Authors' implementation: [SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks). The inspected public repository supplies a vision implementation, not a pretrained language-model conversion. No upstream experiment script was executed here.
- Model: [EleutherAI/pythia-14m](https://huggingface.co/EleutherAI/pythia-14m/tree/cf967c0a9a04383db6f7b1108d86b2962634b4ac), Apache-2.0, pinned revision `cf967c0a9a04383db6f7b1108d86b2962634b4ac`. Approximately 14.1M parameters, six layers, width 128, four attention heads. SafeTensor weights only; `trust_remote_code=False`. The current model card notes that this URL was replaced on 27 February 2026; the revision and SHA-256 manifest prevent ambiguity with the former deduplicated-Pile model.
- Data: [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext), WikiText-2 raw, revision `b08601e04326c79dfdd32d625aee71d232d685c3`, CC BY-SA 3.0 / GFDL. Original train, validation, and test splits are preserved. Source text, parquet and token hashes are recorded. Text is not redistributed in this research package.

No repository visibility, access permissions or license was changed. Original and adapted model tensors are retained locally, not uploaded. The public model and dataset sources can be downloaded and verified by `prepare.py`.

## What was converted

The original six pretrained layers are divided into three independent two-layer groups. Each group receives clean past-token embeddings and corrupted normalized target-token embeddings, plus a new noise-conditioning MLP initialized with zero output. Each group gets its own copy of the original final layer norm.

The original input embedding and output matrix are frozen in **both** the converted model and the ordinary fine-tuning control. This provides a relatively small and controlled comparison: 1,189,888 trainable parameters in the control and 1,246,464 in the conversion. It is not full-parameter fine-tuning.

The attention mask allows a noisy target at position i+1 to see clean tokens through position i and itself, but no other noisy target. Clean tokens use ordinary causal attention and cannot see noisy tokens. Each update trains only one two-layer group. Frozen lexical maps are shared; trainable parameters are not shared between groups.

Noise intervals contain equal probability mass under a truncated log-normal distribution. Blocks cycle in order. Training uses correctly shaped per-example weighted cross-entropy, not a batch-broadcast weight. Inference uses descending-noise Euler updates with **negative** delta sigma, posterior mean embedding estimates, two updates per block and a final low-noise readout: seven denoiser calls, or fourteen layer applications.

This changes the function represented by the pretrained model before fine-tuning. In particular, later groups receive lexical/noise codes rather than the intermediate hidden states on which they were pretrained. The experiment tests direct weight reuse, not a function-preserving conversion or teacher-distilled adapter. This limitation is a plausible explanation, not a causally established diagnosis.

## Frozen protocol

`protocol.json` was written before the first outcome and pins the implementation source hashes. Nothing in the training or evaluation code was changed after freezing.

- Three paired seeds: 1, 2, 3.
- Batch size 2, context length 32, 96 updates: 6,144 training-token exposures per equal-data run. Uniform random windows from the full training split, with replacement.
- AdamW, learning rate 0.0003, twelve-step warmup, gradient clipping 1.0. No hyperparameter search or best-checkpoint selection.
- An additional converted checkpoint at the first update reaching the ordinary fine-tuning run's measured training CPU budget. These had 121, 131 and 128 updates, with a 0.62–0.71% one-update budget overshoot. The equal-data and equal-CPU results are separate comparisons.
- Validation: 32 disjoint windows, 512 scored targets. Test: 64 disjoint windows, 1,024 scored targets. Only the last sixteen predictions of each window are scored.
- Identical data batches within each paired seed. Test data is never used to tune the training procedure. Later decoder diagnostics use validation only.
- One CPU thread in isolated scratch. No GPU rentals or changes to live training or serving. Training budgets exclude setup, model loading, two no-update warmup passes, evaluation, diagnostics and integrity checks.

The dataset is held out from **this adaptation**. Exposure during the model's original Pile pretraining is unknown. An initial whole-file BPE process exited with status 137; the exact kill cause was not diagnosed. The successful frozen preprocessing tokenizes newline-preserved lines independently. It is not claimed identical to whole-file tokenization.

## Held-out results

Lower negative log score is better; units are natural-log nats per scored token.

| Method | Training comparison | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---|---:|---:|---:|---:|
| Untouched pretrained Pythia | No adaptation | 5.903 | same model | same model | 5.903 |
| Ordinary fine-tuning | 96 updates | 6.374 | 6.378 | 6.387 | 6.379 |
| Pretrained diffusion conversion | Same 96 updates/data | 14.018 | 11.352 | 10.864 | 12.078 |
| Pretrained diffusion conversion | Matched training CPU | 13.080 | 12.102 | 11.939 | 12.374 |

For the ordinary models, this is exact next-token log loss at the stated short context. For the converted model, this is the log score of an explicitly specified **four-particle mixture of final softmax probabilities** from target-free noise. It is **not** the exact likelihood, perplexity or ELBO of an implicit diffusion model. Finite-particle scoring can penalize sharp samples that miss the target token. The separate validation diagnostic tests this sensitivity.

Relative to ordinary fine-tuning, the converted finite decoder loses 5.698 nats/token at equal data and 5.994 at matched training CPU. A paired window bootstrap gives intervals [5.444, 5.977] and [5.764, 6.234], conditional on these three seeds and this 64-window test sample. These are not confidence intervals over new models, datasets, or all possible training seeds.

The ordinary fine-tuning control also got worse than the untouched model. This tiny, fixed fine-tuning budget is therefore not a tuned adaptation recipe. There is no evidence here of a quality gain from either adaptation.

## Decoder sensitivity and resource accounting

A post-hoc validation-only diagnostic used the seed-1 CPU-matched checkpoint on 128 targets. Its iterative decoder scored 13.088 nats with four particles and 10.289 with thirty-two particles. The high-noise first-block head, without iterative decoding, scored 8.648 and 8.585 respectively. This confirms substantial decoder/particle sensitivity, but does not restore the original model's quality. These small validation diagnostics did not replace the primary test outcome or select a checkpoint.

On a warmed single CPU thread, the standard model's forward/backward pass took 0.0722 seconds versus 0.0457 for the converted block. The saved non-parameter tensor storages were 16,065,540 versus 15,182,980 bytes: only about **5.5% less** in this vocabulary-head-dominated tiny experiment, not threefold less. This metric is not peak GPU memory, process RSS, optimizer memory or total model storage.

A one-particle inference pass took 0.0284 seconds for the ordinary model and 0.2177 for the converted seven-call decoder, approximately **7.7 times slower**. The primary evaluation additionally uses four particles. Training and inference therefore have different tradeoffs; a faster local training update is not a cheaper or quality-matched language model.

## Interpretation and next experiment

The plumbing works: pretrained weights load, independent groups update, causal masking prevents target leakage, and the finite decoder runs. The intended quality-preserving conversion did **not** succeed in this pilot.

Reasonable next experiments, not completed claims:

1. Add an independently trained context adapter or distillation objective that preserves useful pretrained hidden-state/logit information before replacing the original computation.
2. Run a substantially longer, fixed training budget with learning-rate selection confined to validation and an untouched final test set.
3. Add random-initialization and unweighted-loss ablations to identify whether pretrained weights and the chosen weighting actually help.
4. Evaluate decoded text with a separate teacher and distributional generation metrics, with sufficient samples, rather than treating low-particle log scores as exact diffusion perplexity.
5. Only then test a larger model and measure actual GPU peak memory, training throughput and matched-quality inference cost.

## Reproduction and checks

Use Python 3.12 and `requirements.txt`; for CPU install Torch 2.6.0 from the official PyTorch CPU wheel index. Run `bash reproduce.sh`. The script verifies pinned downloads and preserves the frozen protocol. Equal-data weights and scores are deterministic in the recorded environment. CPU-budget endpoints can land on a different update on another machine; that timing dependence is explicit.

`checks.py` tests original-forward equivalence, future-clean and other-noisy-token exclusion in every block, gradient and optimizer isolation, frozen lexical maps, finite gradients/inference, noise interval mass and sampling, and the Euler sign using a constant oracle. `validate.py` checks all nine adapted checkpoints, their tensor names/finiteness, unchanged original files and frozen maps, and replays both seed-1 equal-data held-out scores.

`summary.json` contains aggregates and bootstrap results. Per-run JSON files include all window scores and measured costs. `diagnostic.json`, `accounting.json` and `samples.json` contain supplementary validation diagnostics, clearly scoped memory/timing accounting, and fixed illustrative generations. No claim of a threefold total-memory saving is made merely because one third of the layers receive gradients.
