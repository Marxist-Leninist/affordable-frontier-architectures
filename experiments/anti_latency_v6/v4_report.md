# Anti-attention optimization and longer-context training, v4

## Findings, training first

Six frozen-checkpoint continuation runs used the same data order and schedule for ordinary attention and signed suppression (U − A). Each started from a selected v2 WikiText checkpoint, then trained 200 optimizer updates, batch 4, context 512: **409,600 additional target bytes per run**, 2,457,600 across six runs. Seeds 1, 2 and 3 are paired. AdamW is restarted, with learning rate 0.001 cosine-decayed to 0.0001, weight decay 0.01, gradient clipping at 1. No hyperparameter search or checkpoint selection used these test losses.

Mean held-out bits per byte (lower is better):

| Dataset/context | Attention before | Attention after | Signed before | Signed after |
|---|---:|---:|---:|---:|
| WikiText 128 | 2.814110 | 2.822252 | 2.788551 | 2.777715 |
| WikiText 512 | 4.653147 | 2.866416 | 4.639102 | 2.771184 |
| FineWeb subset 128 | 3.378523 | 3.387685 | 3.356362 | 3.351137 |
| FineWeb subset 512 | 4.955495 | 3.423462 | 4.994045 | 3.339175 |

Signed has lower final loss than attention in all three seeds on all four metrics. It does **not** improve every before/after case: FineWeb at 128 for signed seed 1 worsens by 0.003026 bpb. Attention's mean 128-context losses also regress slightly. This is evidence that dismissing the short-context-trained model from its first 512-context extrapolation was premature. It is not evidence that added context alone caused the improvements: extra training, optimizer restart and context length are confounded. A matched 128-context continuation would isolate that question. These are tiny 131,904-parameter byte models and small fixed evaluation samples, not broad capability results.

### Training speed

Randomized interleaved full training-step profiling includes forward, cross-entropy, backward, clipping and AdamW. The exact scalar/positional-table implementation changes did not materially accelerate signed training. At context 512 in the initial two-thread run, ordinary attention was 67.37 ms/step and signed 70.78 ms/step.

A separate equal thread-count search (1, 2, 4 for both architectures), with three randomized-order rounds and 45 measurements per setting, found a practical CPU improvement:

| Context | Attention, 2 threads | Attention, 4 threads | Signed, 2 threads | Signed, 4 threads |
|---|---:|---:|---:|---:|
| 128 | 16.154 ms | 15.323 ms | 16.825 ms | 15.922 ms |
| 512 | 63.701 ms | 48.247 ms | 65.594 ms | 50.438 ms |

At 512, signed improves about 1.30× from thread tuning, but remains about **4.5% slower** than equally tuned attention. This is an observed local CPU scheduling improvement, not a new architecture speed advantage. The four-thread setting was selected on these timings; it was not used to rerun all six continuation jobs. A two-versus-four-thread full forward/gradient/one-AdamW-step check was bitwise identical on the tested batch for both architectures.

The actual continuation runs used two threads and took 36.50 seconds of optimizer-step computation for attention and 36.34 seconds for signed, summed over three seeds each. Those runs were sequential, so this near tie does not override the interleaved profile. The successful continuation process took 87.00 seconds including evaluations and checkpoint saves, excluding the separately completed profiling process. Batch assembly is excluded from optimizer-step times. No paid compute, GPU or live server was used.

## Secondary result: cached inference

Exact implementation changes: replace the single-token signed divide/subtract with `torch.add(A, sumV, alpha=-1/n)` and a sign-reversed output projection; reuse positional sin/cos tables for both architectures. Models still share the same single SDPA operation. The optimization is algebraically equivalent, not bitwise equivalent in FP32.

Batch 1, CPU two threads, prefill plus 32 fixed-token cached decoding steps, 31 randomized interleaved repetitions. Reported totals are the median of **each measured paired total**, not a sum of component medians.

| Prefix | Attention, old | Signed, old | Attention, optimized | Signed, optimized | Signed overhead after tuning |
|---|---:|---:|---:|---:|---:|
| 128 | 17.545 ms | 18.112 ms | 14.406 ms | 14.734 ms | 2.3% |
| 512 | 22.798 ms | 23.903 ms | 19.597 ms | 19.741 ms | 0.7% |
| 2048 | 54.727 ms | 54.326 ms | 46.676 ms | 47.035 ms | 0.8% |
| 4096 | 112.976 ms | 118.662 ms | 109.332 ms | 110.451 ms | 1.0% |

These are one seed's steady-state local results, not a guaranteed gap closure. Positional tables are warmed before measurement, including growth. GPU performance is untested (installed torch is CPU-only). Fixed tokens exclude sampling, networking and serving queues. Most benefit comes from the shared positional-table optimization, not signed-specific scalar fusion.

### Correctness and memory

FP64 output and gradient parity pass against the original formula (signed maxima 1.07e-14 and 1.78e-15). FP32 cached parity through prefix 4096 plus eight tokens passes below 3.41e-5 logit error. Final continued checkpoints were compared with equivalent original-formula models on held-out WikiText at 128/512: maximum logit discrepancy 1.96e-5; maximum loss discrepancy 6.02e-8 bpb. State replay is nonmutating. FP32 positional tables add 512 KiB across the two layers at capacity 4096, doubling to 1 MiB at capacity 8192. These are persistent model-level buffers, separate from growing KV cache and 512 bytes of signed running sums. Whole-process maximum RSS during the inference process was about 266 MiB, not an isolated peak-memory comparison.

An initial continuation attempt failed before its first update because tables created under inference mode could not be saved for backward. Construction now explicitly uses `inference_mode(False)` plus `no_grad`, allowing evaluation-created tables to be reused in training. The failure log is retained. Completed profiling was reused rather than rerun. No silent partial training resume occurred.

## Data and reproducibility

- Training: Salesforce/wikitext, wikitext-2-raw-v1, revision b08601e04326c79dfdd32d625aee71d232d685c3, first 2,000,000 bytes after exact line deduplication. UTF-8 byte vocabulary, 256 tokens. Training starts drawn with generator seed 3000 + model seed.
- WikiText evaluation: official test split, exact training-line exclusion, byte offset 196608 in the cleaned stream, 128 nonoverlapping 513-byte windows. Exclude complete windows matching training text. Score the same final 128 targets with 128 and 512 input contexts: 16,384 targets per setting.
- FineWeb transfer: frozen HuggingFaceFW/fineweb-edu sample-10BT train rows 1500–1599 from v31, observed revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9. First 4097 UTF-8 bytes/document, 513-byte windows, first 64 eligible windows, no score-based selection. Exclude exact training matches. Score 8,192 targets per setting. This is a small convenience subset, not the complete v31 cohort; do not compare its raw losses directly with v31's full-cohort losses.
- Exact matches are checked, not semantic/near-duplicate contamination. FineWeb is only evaluation, never used for training or LR selection.
- WikiText card describes CC-BY-SA 3.0 / GFDL source material. FineWeb card identifies ODC-BY; underlying web text can retain third-party rights. Raw corpora are not bundled. Source hashes and manifests are included; downloaders fail closed if bytes/revision change.

Requirements: Python 3.11, torch 2.6.0 CPU, numpy, pyarrow (downloader only). From this directory:

1. `python download_wikitext.py`
2. `python fineweb/prepare_data.py`
3. `python train.py` (uses an existing training_profile.json if present; delete only your copy of that file to rerun initial profiling)
4. `python thread_profile.py`
5. `python validate_final.py` and `python thread_parity.py`
6. `python summarize.py`
7. Optional inference reproduction: `python run.py`

Existing dataset files may be supplied with ANTI_DATA_DIR and ANTI_FINEWEB_ROWS. Bundled files contain all starting and continued checkpoint weights but no optimizer state; these are restarted-optimizer experiments, not recoverable training snapshots.

**Checkpoint convention:** continued signed files store negated `mix.out.weight` for the optimized A−U implementation. Load through `create('signed', seed, True)` then load its state dict. To use the original U−A formula, negate every `mix.out.weight` back; validate_final.py demonstrates and tests this conversion. Attention files have ordinary projections.

Limitations: one small model scale, three paired seeds, one fixed training schedule, small evaluation cohorts, CPU-only, no independent capability/retrieval benchmark, no dedicated long-context training attribution control. Formal or algebraic equivalence of the implementation does not establish a universal quality or speed advantage. No production change or publication occurred.
