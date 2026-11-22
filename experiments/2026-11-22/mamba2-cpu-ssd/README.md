# Exact fast CPU path for Hugging Face Mamba-2

`mamba2_cpu_ssd.py` makes transformers' `Mamba2ForCausalLM` compute the same function on CPU with less work and far less memory traffic. On CPU, transformers runs Mamba-2 through `Mamba2Mixer.torch_forward`, which builds (batch, chunk, l, s, heads, state)-sized temporaries and repeats B and C for every head. This patch uses einsum contractions that compute C.B^T once per group of heads. Only the full-sequence path (training, scoring, prompt prefill) is replaced. One-token decoding keeps transformers' own recurrent step, and the patched prefill writes the same cache.

```python
from transformers import Mamba2ForCausalLM
from mamba2_cpu_ssd import patch_mamba2
model = Mamba2ForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")
patch_mamba2(model)   # returns the number of mixers patched; unpatch_mamba2(model) undoes it
```

Patching swaps each mixer's class to `FastMamba2Mixer`, a `Mamba2Mixer` subclass defined in the module, so a patched model still pickles (`torch.save` / `torch.load`) when `mamba2_cpu_ssd` is importable. The kernel is adapted from the `ssd_minimal` reference in state-spaces/mamba and the surrounding code follows transformers' `torch_forward`. Both are Apache-2.0; see `THIRD_PARTY_NOTICES.md`.

Tested with transformers 4.57.6 and torch 2.10.0+cu128 on CPU. `patch_mamba2` warns when the installed `torch_forward` source differs from the tested one (sha256 `16cec33185821c28...`) or cannot be read; in that case run `test_equivalence.py`.

## Same outputs

`test_equivalence.py` compares the patched model with stock transformers on 15 random cases: 1-4 layers, 1-4 groups, projection biases on and off, the conv bias on and off, a dt limit, a left-padding mask, an untied head, and lengths 1 to 300 (including lengths that are not a multiple of the chunk). All comparisons are in float32 unless stated.

- logits: worst max-abs difference 3.8e-06
- every parameter gradient (cross-entropy loss): worst max-abs difference relative to that gradient's scale 4.7e-06. This is float32 rounding: the contraction order differs, and both paths carry rounding error of this size on poorly conditioned parameters such as `D` and `A_log`.
- prefill with cache (half the text, and a single token), then step-by-step decoding: worst max-abs logit difference 2.4e-06
- greedy `generate()` on a left-padded batch: identical tokens
- a patched model through `torch.save` / `torch.load`: logits identical (max abs 0.0e+00); `unpatch_mamba2` restores the stock class and stock logits exactly (max abs 0.0e+00); patching twice changes nothing ([2, 2])
- a float64 model: logits within 4.8e-07 of stock; under CPU bf16 autocast, logits identical to stock (both paths keep the SSD core in float32 under autocast)
- pretrained `AntonV/mamba2-130m-hf` on a 125-token English paragraph: logits max-abs difference 2.2e-04 at a logit scale of 167.6, arg-max agreement 100.00%, loss 3.481917 vs 3.481916 nats, greedy generation of 40 tokens identical

Result: all checks pass (`results/equivalence.json`).

## Time and memory

`bench.py` uses the same weights and inputs and one CPU thread. Each implementation runs in its own fresh process per round; rounds alternate which implementation goes first. Each pass records wall time, user and system CPU time and page faults. "fwd+bwd" is a forward and backward pass without an optimizer step. Peak RSS includes the model and Python. The stock path allocates large short-lived temporaries, so its speed depends on the memory allocator and on the machine's memory pressure. Every case is therefore run under the default glibc allocator and under tcmalloc (`LD_PRELOAD`).

**xeon-skylake-16vcpu-swapping**: Intel Xeon Processor (Skylake, IBRS, no TSX), 16 logical CPUs, 1 thread used; at the start 11.4 of 30.6 GiB memory available and 30.4 of 32.0 GiB swap in use, load 13.0.

| case | allocator | stock | patched | speed-up (wall) | speed-up (user CPU) | stock system time | stock minor faults per pass | peak RSS stock / patched |
|---|---|---|---|---|---|---|---|---|
| fwd+bwd, 0.47M-param byte model (4 layers, d 128, chunk 32), batch 8 x 256 | glibc | 5.457 s | 0.612 s | **8.9x** | 4.0x | 2.97 s | 786,483 | 1,352 / 1,152 MiB |
| fwd+bwd, 0.47M-param byte model (4 layers, d 128, chunk 32), batch 8 x 256 | tcmalloc | 1.942 s | 0.616 s | **3.2x** | 3.2x | 0.00 s | 863 | 1,175 / 1,003 MiB |
| forward, mamba2-130m, 1 x 256 tokens | glibc | 75.440 s | 3.859 s | **19.5x** | 11.3x | 38.81 s | 9,786,222 | 2,494 / 1,342 MiB |
| forward, mamba2-130m, 1 x 256 tokens | tcmalloc | 28.497 s | 3.270 s | **8.7x** | 8.7x | 0.06 s | 8,640 | 2,526 / 1,354 MiB |
| forward, mamba2-130m, 1 x 512 tokens | glibc | 165.289 s | 8.710 s | **19.0x** | 12.8x | 78.99 s | 19,699,225 | 3,718 / 1,417 MiB |
| forward, mamba2-130m, 1 x 512 tokens | tcmalloc | 66.063 s | 6.430 s | **10.3x** | 10.5x | 0.10 s | 17,320 | 4,514 / 1,442 MiB |
| forward, mamba2-130m, 1 x 2048 tokens | glibc | not run | 38.346 s | - | - | - | - | - / 1,782 MiB |
| forward, mamba2-130m, 1 x 2048 tokens | tcmalloc | not run | 24.949 s | - | - | - | - | - / 1,950 MiB |
| fwd+bwd, mamba2-130m, 1 x 256 tokens | glibc | not run | 8.997 s | - | - | - | - | - / 3,248 MiB |
| fwd+bwd, mamba2-130m, 1 x 256 tokens | tcmalloc | not run | 8.339 s | - | - | - | - | - / 2,976 MiB |

Across these runs the patched path was 3.2-8.9x faster for fwd+bwd (0.47M-param byte model (4 layers, d 128, chunk 32), batch 8 x 256); 8.7-19.5x faster for forward (mamba2-130m, 1 x 256 tokens); 10.3-19.0x faster for forward (mamba2-130m, 1 x 512 tokens). Under glibc, system time (mostly page-fault handling for the stock path's temporaries) was 55% of the stock CPU time for fwd+bwd 0.47M-param byte model (4 layers, d 128, chunk 32), batch 8 x 256, 52% of the stock CPU time for forward mamba2-130m, 1 x 256 tokens, 48% of the stock CPU time for forward mamba2-130m, 1 x 512 tokens; tcmalloc reuses that memory instead. Expect different ratios on other machines, thread counts, lengths and memory loads, and run `bench.py` to measure your own.

"not run": stock transformers was not timed for those cases. At its measured speed for 256 and 512 tokens, the 2048-token forward and the 130m forward+backward would each take several minutes per pass.

## Checkpoints

Two byte-level Mamba-2 models from the OM2 study, in standard transformers format (`checkpoints/<name>/`); they load with stock transformers and need no patch. They were trained on one CPU thread by the study's own single-group copy of this chunked kernel (`om2bench.py`, arm `m2`). Their logits with `patch_mamba2` equal the study model's, and stock transformers agrees to the export-check value below.

| checkpoint | params | updates x batch x bytes | train time | test bits/byte (macro) | article | fiction | dialogue | export check (stock vs study, max abs) |
|---|---|---|---|---|---|---|---|---|
| `m2-seed5617` | 471,008 | 2,000 x 8 x 256 | 21.9 min | 2.1828 | 2.1721 | 2.1908 | 2.1854 | 1.1e-05 (48 windows) |
| `m2-seed5649` | 471,008 | 2,000 x 8 x 256 | 21.9 min | 2.1737 | 2.1567 | 2.1850 | 2.1794 | 1.3e-05 (48 windows) |

Test set: the 303 held-out windows of the P056 corpus; every scored byte sees 129-256 bytes of context. Each model card has a usage example and a note on saving.

## Names used here

- **P056**: a human-written three-domain byte corpus (WikiText-2 articles, Project Gutenberg fiction, OpenAssistant dialogue) built for these experiments, with a source-entity split and train/test overlap exclusion. Sources and licences are in `THIRD_PARTY_NOTICES.md`. The corpus text is not redistributed here.
- **OM2**: the study that trained the checkpoints (Transformer, Mamba-2, two versions of our own Mamba, and a gated exact-token store, all matched at about 0.47M parameters, with one all-metrics index). Its code, results and checkpoints are in `../om2-mamba-v2/`; `export_hf.py` needs its `om2bench.py`.

## Files

- `mamba2_cpu_ssd.py`: `ssd_chunked` (the kernel), `fast_torch_forward`, `FastMamba2Mixer`, `patch_mamba2` / `unpatch_mamba2`
- `test_equivalence.py`: the checks above (`--skip-pretrained` for offline use)
- `bench.py`: the timing tables above (`--label` names the machine; `--allocators glibc,tcmalloc`)
- `export_hf.py`: turns an OM2 `m2` run into a transformers checkpoint and checks its logits three ways
- `make_readme.py`: renders this page from the JSON files
- `results/equivalence.json`, `results/bench_*.json`, `checkpoints/*/metrics.json`: the raw numbers
- `THIRD_PARTY_NOTICES.md`, `SHA256SUMS`

## Scope and limits

- CPU. CUDA users already have the mamba_ssm kernels; this patch does not touch that path.
- The speed-ups compare against transformers' own CPU fallback, not against hand-written C++ or the CUDA kernels.
- Agreement means floating-point rounding (the contraction order differs), not bit identity.
- This folder states no licence of its own; the third-party code it adapts keeps its Apache-2.0 licence (`THIRD_PARTY_NOTICES.md`).

