# Reversible Transformer pilot: prospective protocol

Status: completed. This is the prospective scientific protocol with deployment/scheduling details omitted; see execution_receipt.json and RESULTS_STATUS.md for actual outcomes.

## Questions
1. Which Transformer operations are invertible, and what information must be retained?
2. Does an additive-coupling hidden-state Transformer actually reconstruct its activations with bounded numerical error?
3. Does reconstructing rather than saving activations reproduce ordinary backpropagation gradients and learning?
4. Does the reversible architecture help held-out next-byte prediction against a conventional residual Transformer under matched parameters, data, and a separately reported training-time budget?

## Models and controls
- Ordinary: h <- h + Attention(LN(h)); h <- h + MLP(LN(h)).
- Reversible graph: y1 = x1 + Attention(LN(x2)); y2 = x2 + MLP(LN(y1)), trained by ordinary autograd.
- Reversible inverse: identical architecture, initialization, minibatches, loss, optimizer, and parameters, but reconstruct x2 = y2 - MLP(LN(y1)); x1 = y1 - Attention(LN(x2)) during backpropagation.

The last two isolate the execution algorithm. Both still compute ordinary reverse-mode gradients. The inverse is exact over real arithmetic, approximate in floating point. This is a known reversible-residual principle, not a new invention.

All arms: byte vocabulary256, width48, depth4, attention heads4, MLP96, context64, batch8, tied embedding/readout, trainable positions, no dropout. They have exactly the same named parameters, tensor shapes, and trainable count. Reversible input duplicates the embedding into two48-wide streams; output averages streams. Thus reversible internal state has96 coordinates vs ordinary48. This is an architectural difference despite equal parameter count. Averaging the two streams also changes the residual-path scaling relative to ordinary residual blocks; an apparent architecture effect cannot be attributed to reversibility alone. Embedding duplication and mean readout are not bijections of the complete model. There is no inverse from generated tokens to a unique prompt.

Every block executes the same attention and MLP dimensions once in a forward pass. Inverse backprop adds recomputation, so equal parameters or updates do not mean equal training FLOPs. Report separately: equal updates/data, and equal measured training CPU time. Do not claim simultaneous equality of parameters, exposures, FLOPs, and runtime.

## Training plan, fixed before test results
Seeds11,22,33;500 updates per arm and seed. AdamW lr0.002; linear warmup30; betas(.9,.95); weight decay0.01; gradient norm cap1. Each arm starts from an identical per-seed state dictionary and receives identical minibatches. Training loss is causal next-byte cross entropy at all64 positions.500 updates expose256000 target bytes per arm/seed. Equal-data checkpoints remain immutable. For each seed, continue each faster arm toward the slowest arm's500-update measured training CPU budget, at most2000 updates. If the global cap prevents continuation, mark it incomplete rather than silently removing it. Timed update phases include sampling, zero_grad, forward, backward, clipping, and optimizer; exclude validation, test, initialization and file writes. Record CPU and wall times and continuation overshoot. Fixed order rotates across seeds to reduce ordering bias. No test-based tuning or selection.

## Real data and evaluation
Read-only reuse of pinned Hugging Face Salesforce/wikitext wikitext-2-raw-v1 official train/validation/test, with source revision and exact byte hashes. Source dataset license verified via Hugging Face card (CC-BY-SA3.0 / GFDL tags); no corpus is redistributed. Article-contained training windows. Fixed content-independent validation128 and test256 windows of65 bytes, scoring only the final32 next-byte targets. Report byte NLL in nats and bits-per-byte (BPB), not word/token perplexity. Equal-data and equal-time endpoints both report validation and held-out test. An optional frozen100-document FineWeb-edu out-of-domain set has ODC-BY dataset terms with underlying text rights; no claim that this entire family has never been used in prior research. Exact selected-window/train overlap and duplicates checked; no broad semantic decontamination claim. Training never uses validation, test or FineWeb. Evaluate untrained initialization as a learning sanity check. Report all seeds, paired differences, mean and sample standard deviation, without claiming general intelligence or broad reasoning from a byte-language proxy.

## Correctness gate before training
At depths1,4,12 test inverse(forward(x)) and forward(inverse(y)), float32 and float64; report max absolute and relative L2 errors, not merely a pass flag. Compare all input and parameter gradients from inverse-based and stored-graph backprop at identical states. Test logits and gradients at initialization and after a few paired optimizer steps, including tied embedding gradient. Check causality via future-input perturbations and finite-gradient checks. Smoke tests are synthetic correctness tests, not learning evidence.

## Memory and compute
Report retained non-parameter tensor storage saved by autograd immediately after forward at depths1,4,12, using unique underlying storage IDs. It is not allocator/GPU peak or total memory. Separate fresh-process peak RSS includes Python/Torch and optimizer, not only activations; add baseline RSS and incremental difference with this caveat. Measure median warmed whole forward/backward update CPU and wall times on all arms, plus total training CPU. State parameter/optimizer storage and partial forward matrix MAC counts with omitted operations listed. Inverse execution recomputes a block's nonlinear functions and attention and should be expected to trade runtime for activation memory.

## Cost and execution
Existing CPU-only PyTorch runtime, one Torch/BLAS thread and one core. Global caps600 child-process CPU seconds,900 wall seconds,950MiB owned-process RSS. No new paid compute or production-model changes. The completed run consumed109.915747 child-process CPU seconds and110.857168 wall seconds. Original hashes and observed runtime are preserved in protocol_frozen.json; public-file hashes and sanitization differences are recorded separately.
