## What was run

**Programs.** All 105 records of the ASI-Arch gallery file `data/106.json`, pinned to
[commit 3113c51](https://github.com/GAIR-NLP/ASI-Arch/blob/3113c51b04a1d4a7520a3b0238c6a08d3194d58f/data/106.json)
(SHA-256 `aa77e828…759c`, the same pin as the earlier studies in this repository). The program text is
used unchanged, apart from dropping `@torch.compile` decorator lines. It is not copied into this
repository; `fetch_inputs.sh` downloads it and checks its hash.

**Model around each program.** A byte-level language model: 256-way embedding and output head (not
tied), three pre-norm blocks, each block being the program's `DeltaNet(hidden_size=64, num_heads=2)`
followed by a SwiGLU feed-forward layer. The feed-forward width is chosen so the whole model lands as
close to 350,000 parameters as it can; no filler parameters are added. The Transformer control is the
same wrapper at width 96 with four-head causal attention and rotary positions applied in float32.
This shape was not published with the earlier 350k cohort. It was recovered from that cohort's recorded
parameter counts and reproduces all 19 of them exactly (18 gallery programs and the Transformer's
349,920); see `SMOKE.json`.

**Data.** The v44 data contract of this repository, rebuilt from the public sources and checked hash by
hash against `experiments/asi_arch_psfr_cache_v1/v44_inputs/ledger.json`: the first 2,000,000 bytes of
line-deduplicated WikiText-2-raw train for training; 183 WikiText test windows (from 50 articles) and
99 FineWeb-Edu windows (one per document) for evaluation. Each window is 513 bytes; the model sees 512
and the last 128 predictions are scored. `data/DATA_CHECK.json` holds the comparison: the training
bytes, every window and both evaluation arrays match the ledger. No corpus text is stored here.

**Training recipe.** The one published for the earlier studies, as far as it was published: batches of
4 sequences of 512 bytes, next-byte cross-entropy only, AdamW (learning rate 0.003, betas 0.9/0.999,
weight decay 0.01), gradient clipping at 1, linear warm-up over 32 updates and constant afterwards,
BF16 CPU autocast with FP32 master weights. Every model that shares a seed sees the same training
windows in the same order. Training stops at the first completed update at or past **120 seconds of
training CPU time**; the clock covers building the batch through the optimiser step and nothing else.
Models are evaluated at 60 seconds and at the end.

**Machine.** A two-core cloud VM (Intel Xeon with AMX and AVX512-BF16), PyTorch 2.6.0+cpu,
Python 3.13.18, NumPy 2.3.5. Each job is one single-threaded process pinned to its own core, two jobs
side by side. oneDNN's choice of instruction set was left at its default, so BF16 products used the AMX
kernels (each result records this as `onednn_max_cpu_isa: default`).

## What the harness checks

- **Static audit before anything runs** (`code/static_audit.py`, `STATIC_AUDIT.json`). Every program is
  parsed, not executed, and must stay inside a small tensor-code surface: imports from an allowlist,
  no names that reach the interpreter, the file system, processes or the network, no unusual dunder
  access. All 105 pass. Two things were looked at by hand and allowed: `functools.lru_cache` used as a
  decorator in #330, and `input` used as a parameter name in #562. Only programs whose hash matches a
  clean audit row are ever loaded.
- **No network.** Every job runs in its own network namespace with no interfaces (`unshare -rn`) and
  with Python in isolated mode.
- **Every program builds and runs** (`code/smoke.py`, `SMOKE.json`): 105 of 105 under BF16 autocast.
  43 programs name `torch.bfloat16` in their own code, and 39 of those fail in plain float32, which is
  why BF16 autocast is the only mode used, for the audits as well.
- **Causality, before and after training.** Bytes at and after a cut are replaced with other bytes at
  13 cut positions; predictions before the cut must not move.
- **Batch independence, before and after training.** A sequence's predictions must not depend on what
  else is in its batch.
- **Checkpoints reload.** `code/verify_weights.py` builds a fresh model, restores a saved checkpoint and
  recomputes both held-out losses; see "Checkpoints" below.

## How to run it yourself

From the repository root, with Python 3.11 or newer:

```
python -m venv .venv && . .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy einops pyarrow

E=experiments/gallery_cpu120_20261110
bash $E/fetch_inputs.sh inputs                      # pinned public inputs, hash-checked
python $E/code/static_audit.py inputs/asiarch/106.json work/STATIC_AUDIT.json
python $E/code/data_prep.py inputs/hf experiments/asi_arch_psfr_cache_v1/v44_inputs work/data
# from here on gallery programs are executed, so everything runs without a network and with Python isolated:
unshare -rn -- python -I $E/code/smoke.py inputs/asiarch/106.json work/STATIC_AUDIT.json work/SMOKE.json

# one job (here gallery #458, seed 1), isolated from the network:
unshare -rn -- python -I $E/code/run_job.py --arm 458 --seed 1 --gallery inputs/asiarch/106.json \
    --audit work/STATIC_AUDIT.json --data work/data --out work/results --budget 120 --marks 60 --save-weights

# the whole queue, one job per listed core; it can be stopped (touch work/results/STOP) and started again:
python $E/code/driver.py $E/QUEUE.json work/results inputs/asiarch/106.json work/STATIC_AUDIT.json \
    work/data "$(command -v python)" 0,1

# confirmation: pick the eight best of the screen, run them and the controls on seeds 2 and 3 in one block
python $E/code/make_confirm_queue.py work/results work/QUEUE_CONFIRM.json 8
python $E/code/driver.py work/QUEUE_CONFIRM.json work/results_confirm inputs/asiarch/106.json work/STATIC_AUDIT.json \
    work/data "$(command -v python)" 0,1

python $E/code/summarise.py work/results experiments/asi_arch_psfr_cache_v1/v44_inputs/ledger.json \
    $E/COHORT_SETS.json work/summary work/results_confirm
unshare -rn -- python -I $E/code/verify_weights.py work/results inputs/asiarch/106.json \
    work/STATIC_AUDIT.json work/data work/VERIFY.json
```

Run on the files in this folder (`$E/results` and `$E/results_confirm` in place of the `work/` ones), `summarise.py`
writes a `METRICS.json` and a `RESULTS.md` identical to the ones published here.

`--arm transformer` runs the control. A run stops on CPU time, so the number of updates, and with it
the exact result, depends on the machine: expect the same picture, not the same digits.
