# Anti-attention v5: separating context adaptation and optimizer restart

## New findings

The earlier v4 improvement was not just extra updates or a fresh optimizer. A matched training-context control now uses **exactly the same target bytes, 200 optimizer updates, learning-rate schedule, starting checkpoints and fresh AdamW**. It splits each 512-target span into four 128-target sequences: batch 16 × 128 versus batch 4 × 512. Three paired seeds, 409,600 additional target bytes per run.

Mean test bits per byte (lower is better):

| Model | Training context | Wiki test128 | Wiki test512 | FineWeb test128 | FineWeb test512 |
|---|---:|---:|---:|---:|---:|
| Attention | 128 | 2.764988 | 4.769437 | 3.352411 | 5.073878 |
| Attention | 512 | 2.822252 | 2.866416 | 3.387685 | 3.423462 |
| Signed U−A | 128 | 2.748256 | 4.731935 | 3.334828 | 5.092309 |
| Signed U−A | 512 | 2.777715 | 2.771184 | 3.351137 | 3.339175 |

**Interpretation:** training with longer available context and positional range is crucial to the 512-context recovery in these models. Merely training longer with a restarted optimizer at 128 does not provide it. However, the 512-context configuration trades away some short-context quality relative to the equally trained 128-context configuration. Sequence boundaries, positional range and available context change together; this does not identify which individual mechanism matters most.

## Optimizer-state control: useful additional improvement

Original optimizer states were not archived. We reconstructed the original seed1 training runs with the recorded data/order, initialization and schedule. Both reconstructions matched the historical checkpoint weights **exactly: maximum parameter difference zero**. From those identical reconstructed weights, we compared retained versus fresh AdamW state at 512 context. Both arms use the same 200 updates, batches and explicit new learning-rate schedule. Retaining Adam preserves its first/second moments and step counters; both arms follow the same LR values.

| Model | Adam state | Wiki128 | Wiki512 | FineWeb128 | FineWeb512 |
|---|---|---:|---:|---:|---:|
| Attention | Reset | 2.824473 | 2.873896 | 3.384822 | 3.413248 |
| Attention | Retained | 2.797841 | 2.790863 | 3.367212 | 3.358316 |
| Signed U−A | Reset | 2.759163 | 2.754216 | 3.352382 | 3.341444 |
| Signed U−A | Retained | 2.750912 | 2.737176 | 3.343925 | 3.328170 |

Retaining Adam improves all four metrics in both seed1 models. For signed suppression, the extra 512-context improvement is 0.017040 WikiText and 0.013274 FineWeb bpb. This is a measured candidate improvement, **single seed only**, not a guaranteed result. The clearest next validation is to replicate retained-state training on seeds2/3. Mixed short/long-context training could address the quality trade-off, but has not been tested here.

There is no retained-optimizer 128-context arm. Context–optimizer interactions are therefore not fully isolated. The reset-vs-retain comparison starts from exactly identical weights; it does not rely on comparing unrelated historical runs.

## Compute and performance scope

New v5 training presented 5,734,400 target bytes total:
- Six context-control continuations: 2,457,600.
- Two seed1 optimizer reconstructions: 1,638,400.
- Four optimizer-control continuations: 1,638,400.

Context-control optimizer steps total45.97 seconds. Reconstruction wall time totals40.46 seconds. Optimizer-control steps total49.97 seconds. These accounting numbers omit evaluation/checkpoint overhead except reconstruction includes batch assembly. Runs are CPU-only with two threads and were sequential; they are not comparative speed evidence.

No new inference-only optimization was attempted in this round. The included v4 report/raw profiles remain the previous findings: four-thread CPU training tuning improves both architectures, signed remains about4.5% slower than equally tuned attention at512, and the warmed positional-table inference implementation narrowed the measured gap. Those measurements are supporting history, not reruns or new v5 claims. A smaller inference gap does not establish a training-speed win.

## Dataset, protocol and limitations

Same frozen data and evaluation as v4, documented fully in v4_report.md:
- Salesforce/wikitext wikitext-2-raw-v1, revision b08601e04326c79dfdd32d625aee71d232d685c3; UTF-8 byte tokenizer, first2,000,000 deduplicated training bytes.
- New v4 heldout WikiText slice offset196608:128 windows, scoring final128 targets at128/512 contexts,16,384 matched targets per setting.
- Fixed first64 eligible windows from the v31 FineWeb-Edu sample10BT cohort, revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9;8,192 matched targets per setting. This is a small subset, not v31's complete cohort.
- Exact-match training exclusions; no claim of exhaustive near-duplicate decontamination. No test data enters training. Existing evaluation results motivated these diagnostic controls, so this is an exploratory research sequence, not a preregistered broad superiority test.
- Context-control starts use generator seed3000+model seed. Original replay uses1000+model seed and torch initialization seed1. All schedules and sampling code are included.
- Tiny131,904-parameter models, three-seed context comparison and one-seed optimizer comparison. No fresh capability/retrieval benchmark, GPU result or large-model extrapolation.

## Reproduce

Python3.11, torch2.6.0 CPU, numpy; pyarrow for the pinned WikiText downloader. From this directory:

1. `python download_wikitext.py`
2. `python fineweb/prepare_data.py`
3. `python continue_control.py`
4. `python restart_control.py`

ANTI_DATA_DIR and ANTI_FINEWEB_ROWS may point to existing matching data. Downloaders fail closed on source hash/revision mismatch. Raw corpora are not bundled. Existing v4 scripts and source checkpoints are included so no sibling project directory is required.

Checkpoint conventions:
- `control128_*.pt` and `continued_*.pt` use the optimized implementation. Signed projections are negated; load with `create('signed', seed, True)` before loading state.
- `restart_*.pt` use the original U−A implementation without projection sign reversal; load with `previous.model(kind)`.
- Newly saved artifacts contain weights, not optimizer snapshots. The restart script reconstructs optimizer state deterministically in memory. Do not treat weight-only files as full optimizer-resumable checkpoints.

The independent read-only audit verified controls, target matching and result summaries. No production or paid resources were used; no repository publication was attempted.
