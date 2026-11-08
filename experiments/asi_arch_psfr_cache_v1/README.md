# ASI-Arch #879 FusionGatedFIRNet: exact incremental decoding (v1)

**Result.** A cached decoder for FusionGatedFIRNet (`delta_net_psfr`, ASI-Arch record #879) that reproduces the
original full-prefix forward exactly. Checked on the three trained v44 checkpoints with real held-out FineWeb bytes:

* **FP64:** worst logit error 1.3e-14 (rounding only).
* **FP32:** worst logit error 1.03e-5.
* **Greedy generation:** 64 tokens × 3 sequences per seed were identical with and without the cache.

The cache has a fixed size (103,680 bytes for the whole 3-layer model, FP32, batch 1). It makes per-token decode 3.4×, 5.2× and 8.1× faster than recomputing
the prefix at prompt lengths 128, 512 and 1,024.

This is an independent implementation. ChatGPT's separate "v46" repair, recorded in SG memory slot 2293, was not
available to read. Its reported figures broadly agree with ours: 3.08×, 5.87× and 8.97× faster 32-token decoding and a
105,216-byte state.

## The bug

The exported layer (`v44_inputs/original/delta_net_psfr.py`, unchanged here) recomputes everything from zero on every
call:

* `_retention_delta_chunkwise` always starts its state `S` at zeros, so the recurrent cache it writes is never read back.
* Both FIR branches (`_DepthwiseFIRConv1d`, kernels 3 and 63) left-pad zeros on every call.

Feeding a sequence in pieces therefore gives different outputs from one full call. This run reproduces the v44 audit's
first-layer errors exactly (first FineWeb window, 65 bytes):

| Seed | Original cache, max abs error | This cache, max abs error (FP32) |
|---|---:|---:|
| 1 | 1.5367 | 1.2e-6 |
| 2 | 1.6414 | 8.9e-7 |
| 3 | 2.0412 | 8.3e-7 |

## What `psfr_cache.py` reproduces

It reuses the layer's own parameters through a separate code path (`layer_step`, `lm_step`):

* **q, k and v:** projection, then a causal depthwise ShortConvolution (kernel 4, SiLU), split into 2 heads × 32 channels.
* **beta:** `sigmoid(b_proj(x))`. Inside the kernel, q and k are L2-normalised with epsilon 1e-6.
* **Delta rule within a 32-token chunk:** `S <- S + beta * k (v - k^T S)^T`, and the output is `q^T S` taken after the update.
* **Chunk boundaries sit on absolute positions** 0, 32, 64, and so on.
  * After each complete chunk, the next one starts from `S_end - (1 - lambda) * S_chunk_start`, with `lambda = 0.6 + 0.4 * sigmoid(retention_param)` per head.
  * This is the source's chunk-boundary retention, not a per-token decay.
  * The cache applies it lazily, when the first token of the next chunk arrives, so a partial chunk is never finalised early.
* **FIR branches** read `v_direct`, so the cache keeps its last 62 steps.
* **Token-local parts** need no cache: the fusion-gate features, the epsilon-floored sigmoid gates, the per-head RMSNorm and `o_proj`.

The per-layer cache holds:
* three conv tails (3 steps each);
* the 62-step FIR tail;
* the chunk-start state and the running state (2 heads × 32 × 32 each);
* the position.

Prefill runs whole chunks in the source's chunk form, but with a carried-in state. Only a partial head or tail chunk runs
token by token.

## Equivalence

The reference is the original full-prefix forward over 512 bytes. It is compared with a prefill of P bytes, then byte-by-byte decoding
up to 512. Each figure is the maximum over seeds 1–3, with a batch of 3 real held-out FineWeb windows.

| Prefill P | FP64 prefill | FP64 decode | FP32 prefill | FP32 decode |
|---:|---:|---:|---:|---:|
| 1 | 5.3e-15 | 1.3e-14 | 3.3e-6 | 7.4e-6 |
| 31 | 8.0e-15 | 1.2e-14 | 4.3e-6 | 1.0e-5 |
| 32 | 7.1e-15 | 1.2e-14 | 3.3e-6 | 9.8e-6 |
| 33 | 7.1e-15 | 1.2e-14 | 3.3e-6 | 8.3e-6 |
| 64 | 7.6e-15 | 1.2e-14 | 2.9e-6 | 9.1e-6 |
| 100 | 7.1e-15 | 1.3e-14 | 4.8e-6 | 7.2e-6 |
| 300 | 8.9e-15 | 1.3e-14 | 5.3e-6 | 5.7e-6 |
| 511 | 8.0e-15 | 7.1e-15 | 4.8e-6 | 3.3e-6 |

The tolerances were 1e-10 for FP64 and 1e-4 for FP32, and every case passed. Further checks, all passing:

* **Random uneven pieces** (1 to 79 bytes each, mixing head, body and tail chunks): 1.1e-14 in FP64 and 4.8e-6 in FP32.
* **Chunk-form versus token-loop prefill state** after 300 bytes: 2.7e-15 in FP64.
* **Beyond the 512-byte training context:** prefill 1,024 bytes and decode 64 more; worst error 1.1e-14 in FP64 and 6.0e-6 in FP32.
* **Greedy generation in FP32:** a 256-byte prompt then 64 tokens, batch 3, gave identical tokens with the cache and with full recompute, for all three seeds.

## Speed

Setup: CPU, FP32, one thread, batch 1, inference mode, 3 seeds × 4 held-out FineWeb documents × 3 randomised paired repeats.
Each arm produces the same 33 next-byte distributions: a prompt of P bytes, then 32 real bytes fed one at a time.
Figures are paired medians. The laptop was shared with other work and the process ran at below-normal priority, so absolute
milliseconds are noisy.

| Prompt | Fusion cached (ms/token) | Fusion full recompute | Transformer KV cached | Transformer full recompute |
|---:|---:|---:|---:|---:|
| 128 | 16.3 | 55.2 | 4.2 | 9.8 |
| 512 | 21.2 | 85.5 | 4.9 | 36.0 |
| 1,024 | 21.1 | 171.7 | 5.9 | 107.7 |

| Prompt | Cache speed-up over recompute: decode / with prefill | Fusion cached vs Transformer cached | Fusion state | Transformer KV state |
|---:|---:|---:|---:|---:|
| 128 | 3.4× / 3.2× | 4.6× slower | 103,680 B | 245,760 B |
| 512 | 5.2× / 4.5× | 4.3× slower | 103,680 B | 835,584 B |
| 1,024 | 8.1× / 6.4× | 3.6× slower | 103,680 B | 1,622,016 B |

Prefill with the cache costs about the same as one full forward pass: 1.00×, 1.23× and 1.02× at the three prompt lengths.

**Reading it:**
* The cache makes Fusion's per-token cost flat in context length and its state fixed. The Transformer's KV cache grows
  linearly and is already 15.6× larger at 1,024 + 32 bytes.
* In this unfused CPU reference, each Fusion decode step is still 3.6–4.6× slower than the matched Transformer's.
* At width 64, Python and per-op overhead dominate. A Fusion step runs a short convolution, two FIRs, a delta update and a
  gate MLP, while attention decode is one small SDPA call.
* The gap shrinks as context grows. Nothing here predicts fused-kernel GPU speed or large-model behaviour.

## Limits

* **Scope:** inference only. Training uses the original forward and is unchanged.
* **Dependency adapters:** CPU adapters stand in for FLA's fused kernels (see `v44_inputs/ADAPTATIONS.md`).
* **Model size:** these are tiny models of 190,788 parameters (3 layers, width 64, byte vocabulary).
* **Batching:** decode assumes every row in a batch is at the same position, with no padding or packing, as the original does.

## Reproduce

```
pip install torch numpy einops==0.8.1
python restore_fineweb.py v44_inputs     # pinned FineWeb rows; checks the response and every window hash
python verify_psfr_cache.py v44_inputs   # writes verify_results.json (30 min here: 4 threads, shared laptop, low priority)
python time_psfr_cache.py v44_inputs     # writes timing_results.json (13.5 min here, 1 thread)
```

`results/` holds the JSON from this run, made with Python 3.12.10 and torch 2.14.1+cpu on Windows 11. The scripts stub
`resource`, a Linux-only module imported by `source_v38.py` that the models don't use. `restore_fineweb.py` only rebuilds
the FineWeb half of `prepare_inputs.py`, so it needs neither pyarrow nor `run.py`.

## Files

* `psfr_cache.py`: the cached decoder (`layer_step`, `lm_step`, `empty_state`).
* `verify_psfr_cache.py`, `time_psfr_cache.py`, `restore_fineweb.py`.
* `results/verify_results.json` and `results/timing_results.json`: full results, including every raw timing row.
* `v44_inputs/`: the minimal subset of the v44 package that the scripts need.
  * The original GAIR-NLP exports under `original/` are Apache-2.0, from ASI-Arch commit
    3113c51b04a1d4a7520a3b0238c6a08d3194d58f.
  * The other files are the CPU harness, the six trained checkpoints (hashes in `CHECKPOINTS.sha256` and the per-seed JSON),
    `ledger.json` and `fineweb_request.json` for the data restore, and the v44 report and adaptations notes.
  * Corpus text is not included; `restore_fineweb.py` rebuilds it.
