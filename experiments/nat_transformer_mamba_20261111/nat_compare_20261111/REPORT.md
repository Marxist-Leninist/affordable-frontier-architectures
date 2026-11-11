# Single-pass NAT Transformer versus bidirectional Mamba-1
Measured November 11, 2026. Small isolated CPU experiment, not a 600B deployment trial.

## Decision
This Transformer trained **9.17× faster** than this unfused Mamba CPU reference at equal updates. Held-out capability differences depend on the task; there is no defensible single “intelligence percentage.” The reference scan is not an optimized fused Mamba kernel, so this result does not establish a general CPU or GPU architecture ranking.

**No live 600B change was made.** Neither tiny-model results nor a drop-in replacement of an attention block establishes that a trained 600B checkpoint would improve. A Mamba architecture swap requires compatible training, not merely replacing inference code.

## Training and single-pass generation
Each model receives 32 prefix byte positions and 16 MASK placeholders. It emits all 16 output distributions in one forward pass, followed by parallel argmax. No generated token is fed back, no gold suffix is supplied, and there are no diffusion/refinement steps. Output length is fixed, so this does not solve length prediction or open-ended generation.

Both models: two layers, width 48, byte vocabulary plus input-only MASK/PAD, independent 256-byte output projection, FP32, one Torch thread, 16 examples/update. Transformer: four-head bidirectional SDPA, RoPE, RMSNorm, squared-ReLU FFN width 266. Mamba: separate forward/reverse Mamba-1 branches, expansion 2, state width 8, convolution width 4, RMSNorm and associative scan. The shared sinusoidal input position scheme is identical. Parameter matching is achieved with the Transformer FFN width, not by adding unused parameters.

This is AGILLM-inspired, not exact AGILLM: the actual historical AGILLM NAT suffix-mask/head path and later packed128b core were inspected. Production quantization, routed experts, tokenizer, AR/SAT objectives, scale and full training recipe are not reproduced. Mamba-1 is used, not Mamba-2.

| Quantity | Transformer | Mamba reference |
|---|---:|---:|
| Dense parameters | 94,416 | 94,320 |
| Mean time, 800 updates | 9.886 s | 90.638 s |
| Mean supervised byte targets/s | 20,723 | 2,260 |
| Inference batch 1, p50/p95 | 0.805/1.006 ms | 2.651/3.123 ms |
| Inference batch 16, p50/p95 | 3.823/4.096 ms | 22.021/22.988 ms |

Inference is prefix processing plus all 16 output argmaxes, not autoregressive token/s. Timing uses randomized paired family order, warmups, 60 trials per seed/batch, all three seeds. The CPU is a shared AMD EPYC 9V74; unrelated host load is uncontrolled. Training time includes forward, loss, backward, clipping and AdamW update but excludes batch preparation, initialization, evaluation and saving.

Each family gets two learning rates (0.001 and 0.003), 300 updates at tuning seed 101; both select 0.003 on validation. Final seeds 1, 2, 3 receive 800 updates each, identical paired minibatches. Each final run presents 204,800 byte targets, half WikiText and half fresh digit tasks; equal data/parameters do not imply equal FLOPs. Learning-rate optima beyond the upper grid endpoint and training to convergence are not established.

## Untouched-test quality after equal training data
Means of all three seeds. The text column is conditional suffix cross-entropy in bits/byte, not autoregressive perplexity. Accuracy is individual bytes; exact match is the complete 16-byte output.

| Task and metric | Transformer | Mamba reference |
|---|---:|---:|
| Text bits/byte, lower is better | 4.58790 | 4.53755 |
| Text byte accuracy | 18.83% | 19.38% |
| Copy byte accuracy | 99.61% | 29.92% |
| Copy complete-output exact match | 94.14% | 0.00% |
| Reverse byte accuracy | 49.62% | 30.79% |
| Reverse complete-output exact match | 0.00% | 0.00% |
| Sort byte accuracy | 75.37% | 85.16% |
| Sort complete-output exact match | 1.43% | 9.11% |

Text seed SD: Transformer 0.00625; Mamba 0.00588. Three seeds are too few for a broad architecture conclusion. The text samples are weak small-model completions, not evidence of useful general conversation. Copy/reverse/sort are narrow learned operations, not general reasoning or knowledge.

## Equal measured training-time snapshots
The first update crossing 5 seconds was saved for every final run. Transformer steps: [409, 382, 416]; Mamba steps: [47, 46, 46]. Actual times: Transformer [5.0084, 5.0068, 5.0038], Mamba [5.0841, 5.108, 5.0702]. This is a near-equal budget, with at most one update of overshoot, and excludes evaluation/preparation time. It is a very early learning comparison, not time to convergence.

| Test metric at about 5 training seconds | Transformer | Mamba reference |
|---|---:|---:|
| Text bits/byte | 4.60015 | 4.63430 |
| Copy byte accuracy | 99.54% | 11.43% |
| Reverse byte accuracy | 29.44% | 11.22% |
| Sort byte accuracy | 66.82% | 34.33% |

Full validation learning curves every 100 updates, CPU/wall training times, raw inference trials and all checkpoint hashes are retained.

## Data, held-out separation and coverage
WikiText-2 raw official splits, Salesforce/wikitext revision b08601e04326c79dfdd32d625aee71d232d685c3. Input is UTF-8 bytes, no learned tokenizer. Training uses the first 2,000,000 serialized training bytes; each batch draws new windows. The primary text test has 512 windows and 8,192 target bytes. No full 48-byte primary evaluation window matches the used training prefix, and no full primary window or 16-byte target overlaps between validation and test. Shorter training phrase overlap exists: 33/512 test target strings and 28/512 validation target strings appear in training. No corpus is redistributed.

The original selector accepts the first 512 of 1,024 evenly spaced candidates and therefore covers approximately the first half of the held-out file. This was identified before opening test quality. A clearly secondary coverage extension then evaluated all admissible candidates across the full test file with no retraining or selection:

| Full-file coverage extension | Transformer | Mamba reference |
|---|---:|---:|
| Text bpb, 800 updates | 4.57726 | 4.52368 |
| Text bpb, about 5 s | 4.59188 | 4.62839 |

This extension has 1024 windows and 16,384 target bytes; 63 target strings occur in the used training prefix. It is a convenience sample spanning the file, not a representative population sample. The primary cohort is retained, not replaced after seeing scores.

Synthetic tasks use unseen 16-digit inputs. The audit found all 23,136 input strings unique across tuning/final-train/validation/test sets. Each task test has 256 examples. Evaluation target tokens never enter the model.

## Compute and implementation limitations
Forward arithmetic per example at 48 positions: Transformer approximately 3,975,168 matrix MACs; Mamba approximately 3,330,048 matrix/convolution MACs plus 2,073,600 scalar scan multiplies/adds. These omit nonlinearities, normalization, exponentials/discretization, backward and optimizer. Both models are dense, so there is no sparse active-parameter discount.

The CPU Mamba scan has O(T log T) work and extra intermediate allocations; production fused Mamba kernels use a much more efficient implementation. Comparing it with optimized PyTorch SDPA measures these implementations, not a universal theoretical advantage. Supplementary inference lengths 128/512 are timing-only, untrained-context tests. No GPU timing was run: both existing GPUs were busy with live workloads, and those workloads were left untouched.

## Verification and sources
Four independent unit tests passed: affine scan outputs/gradients equal sequential recurrence (float64 max error below 2.7e-15); full Mamba input/parameter gradients agree; directionality and bidirectionality work; blank-only suffix and one-pass execution verified. Raw data hashes, paired minibatch hashes, duplicate screens and source provenance are included.

Mamba-1 equations and initialization: [state-spaces/mamba, pinned source](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa), Tri Dao and Albert Gu, Apache-2.0. The CPU implementation is independent, not the official CUDA package. [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext): metadata reports CC-BY-SA-3.0/GFDL; licensing prose has also linked CC-BY-SA-4.0, so preserve source rights and that uncertainty. Private user-owned AGILLM source was inspected but is not redistributed.

## Reproduce
The bundle contains this report, full scripts, protocols, audits, metrics and small checkpoints. Use Python 3.12, NumPy and PyTorch 2.6.0 CPU. PyArrow 19.0.1 is needed only to prepare WikiText.

1. Extract the archive, preserving nat_compare_20261111 and sibling frontier_research_v8 directories.
2. In frontier_research_v8, run python prepare_data.py to retrieve the pinned official files and verify their hashes.
3. In nat_compare_20261111, run python audit_tests.py, then python run.py.
4. Run python evaluate_whole_split.py, python benchmark_lengths.py, python analyze.py, and python make_report.py. Run python data_audit.py for the independent data checks.

Timing values and the update at which 5 seconds is crossed will vary across hardware and host load. Do not run overlapping timing jobs. No live services, paid resources, permissions or production weights need to change.
