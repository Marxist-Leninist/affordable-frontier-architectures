# All 105 gallery programs, 120 CPU-seconds each, with confirmation on fresh seeds

Run on 10 and 11 November 2026 by Claude (Anthropic's assistant) at Scott's request, with a harness written for this purpose and independent of the one behind the earlier gallery studies. The harness, the data manifest and every per-run result are in this folder.

## In short

- **Coverage.** The earlier campaign had screened 25 of the 105 original gallery programs. All 105 have now been trained and evaluated under one recipe, 80 of them for the first time.
- **Screen (one seed each).** 98 of 105 programs beat the Transformer control's seed 1 on both held-out sets. 17 programs scored better than the previous best, #458 DLGM, on that single seed. Most of that does not survive: see the next point.
- **Confirmation (fresh seeds, controls re-run alongside).** Of the eight best programs of the screen, 5 fell behind DLGM on both sets on seeds 2 and 3. The one clear result is **#259 `bias_init_mix_gate`**: over four fresh seeds (2, 3, 4, 5) it averages 1.7732 nats per byte on WikiText and 2.0945 on FineWeb-Edu, against 1.7990 and 2.1237 for DLGM in the same block: -0.0259 and -0.0292, better in 4 of 4 seeds on WikiText and 4 of 4 on FineWeb-Edu. Its parent design, #151 `adaptive_mix_gate`, is ahead of DLGM too, by less (-0.0167 and -0.0083; 4 of 4 and 3 of 4 seeds).
- **One run is noisy.** The same control with the same seed, run twice, differed by up to 0.020 nats on WikiText and 0.029 on FineWeb-Edu, only because a different number of updates fitted into 120 CPU-seconds. Differences of that size between two single runs mean nothing.
- **For whoever runs the next tests.** `fetch_inputs.sh` and `code/` rebuild the exact v44 data from public sources (every hash matches the published ledger) and run any gallery program by number. Nothing else is needed.

## Limits first

- Tiny byte-level models (about 350,000 parameters) trained for two CPU-minutes. Nothing here says how any of these designs behaves at a useful size, on a GPU, or with more training.
- The held-out windows are the v44 windows this repository has used before. They are not fresh, and the screen picked its candidates on them. The confirmation uses fresh seeds, not fresh text.
- The comparison is at equal training CPU time on one machine. A program that does less work per update gets more updates; that is the point of the comparison, and it also means the ranking depends on this machine's kernels (AMX for BF16) and on PyTorch 2.6.
- Stand-ins for the flash-linear-attention modules are plain PyTorch, not the official fused kernels (`code/fla_cpu.py`).
- Each program is used as written. Schedules inside a program that count thousands of forward passes were not rescaled to a run of a few hundred updates, and extra losses that a few programs expose (for example `reg_loss`) were not used: every model is trained on next-byte cross-entropy only.
- This is a different harness, machine and evaluation set from the v65 study, so the numbers are not interchangeable with v65's. The order of the three controls is the same in both.
- One program, #902 `entropy_floor`, cannot be brought down to the size budget in this wrapper (625,890 parameters at the smallest feed-forward width). It was run and is reported, but not ranked.

## Results

Cross-entropy in nats per byte, lower is better. Full tables: [RESULTS.md](RESULTS.md). Every number, per seed and per window: [METRICS.json](METRICS.json) and `results/`.

### Controls in the screen (three seeds)

| Control | Parameters | WikiText (mean, sd) | FineWeb-Edu (mean, sd) | Updates in 120 s |
| --- | ---: | ---: | ---: | ---: |
| Transformer, width 96, rotary positions | 349,920 | 1.8623 (0.0179) | 2.1922 (0.0265) | 662, 674, 679 |
| #879 psfr ("Fusion" in the earlier reports) | 349,956 | 1.8077 (0.0037) | 2.1467 (0.0081) | 376, 391, 386 |
| #458 dlgm ("DLGM" in the earlier reports) | 349,944 | 1.7923 (0.0047) | 2.1207 (0.0070) | 403, 421, 408 |

### The screen: best 15 of 105 on seed 1

| Rank | Program | Updates | WikiText | FineWeb-Edu | Screened before |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | #259 bias_init_mix_gate | 493 | 1.7381 | 2.0692 | no |
| 2 | #151 adaptive_mix_gate | 490 | 1.7489 | 2.0897 | no |
| 3 | #957 psafg | 434 | 1.7674 | 2.0982 | yes |
| 4 | #908 aft | 484 | 1.7795 | 2.0979 | yes |
| 5 | #804 phsg5 | 432 | 1.7734 | 2.1140 | no |
| 6 | #437 dual_path_fusion | 444 | 1.7719 | 2.1173 | no |
| 7 | #641 hwg | 403 | 1.7924 | 2.0971 | yes |
| 8 | #556 dfpcr | 449 | 1.7875 | 2.1025 | yes |
| 9 | #922 phfg | 407 | 1.7896 | 2.1038 | no |
| 10 | #750 adaptive_hier_gate | 404 | 1.7946 | 2.1060 | yes |
| 11 | #559 mor | 393 | 1.7936 | 2.1073 | yes |
| 12 | #645 htcg | 445 | 1.7933 | 2.1114 | no |
| 13 | #1252 adgr | 414 | 1.7879 | 2.1174 | no |
| 14 | #676 ser_minfloor | 409 | 1.7976 | 2.1082 | no |
| 15 | #441 msdfdm | 424 | 1.7871 | 2.1204 | no |
| 18 | #458 dlgm (control, seed 1) | 403 | 1.7958 | 2.1162 | yes |
| 35 | #879 psfr (control, seed 1) | 376 | 1.8112 | 2.1424 | yes |
|  | Transformer (control, seed 1) | 662 | 1.8826 | 2.2214 |  |

Against the Transformer averaged over its three seeds, window by window, 72 programs are better on both sets with 95% intervals that exclude zero, and 7 are worse on both. The slowest programs manage fewer than a hundred updates in the time and finish last (the range is 84 to 493 updates).

### Confirmation on seeds 2 and 3

The rule was fixed before the screen had finished (`code/make_confirm_queue.py`): the eight programs with the lowest seed-1 sum of the two losses, leaving out the two gallery controls. They were run on seeds 2 and 3 in one block, and the three controls were run again on the same seeds in that block, spread among them, so that both sides of each comparison were measured in the same stretch of time. Seed 1 is shown but not counted, because it is what the candidates were picked on. An asterisk marks a 95% interval over held-out windows that excludes zero.

| Model | Seed 1 (selection) | Mean of seeds 2-3: WikiText | FineWeb-Edu | vs DLGM: WikiText | FineWeb-Edu | Seeds better than DLGM |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| #259 bias_init_mix_gate | 1.7381 / 2.0692 | 1.7607 | 2.0798 | -0.0296* | -0.0386* | 2 of 2 / 2 of 2 |
| #151 adaptive_mix_gate | 1.7489 / 2.0897 | 1.7775 | 2.0999 | -0.0128* | -0.0186* | 2 of 2 / 2 of 2 |
| #437 dual_path_fusion | 1.7719 / 2.1173 | 1.7888 | 2.1158 | -0.0015 | -0.0027 | 1 of 2 / 1 of 2 |
| **#458 dlgm (DLGM)** | 1.7958 / 2.1162 | 1.7903 | 2.1185 |  |  | control |
| #908 aft | 1.7795 / 2.0979 | 1.7998 | 2.1303 | +0.0094* | +0.0118* | 0 of 2 / 0 of 2 |
| #804 phsg5 | 1.7734 / 2.1140 | 1.8063 | 2.1313 | +0.0160* | +0.0128* | 0 of 2 / 1 of 2 |
| **#879 psfr (Fusion)** | 1.8112 / 2.1424 | 1.8228 | 2.1374 |  |  | control |
| #641 hwg | 1.7924 / 2.0971 | 1.8161 | 2.1518 | +0.0258* | +0.0333* | 0 of 2 / 0 of 2 |
| #957 psafg | 1.7674 / 2.0982 | 1.8218 | 2.1536 | +0.0315* | +0.0351* | 0 of 2 / 0 of 2 |
| #556 dfpcr | 1.7875 / 2.1025 | 1.8381 | 2.1634 | +0.0477* | +0.0449* | 0 of 2 / 0 of 2 |
| **Transformer (width 96)** | 1.8826 / 2.2214 | 1.8513 | 2.1705 |  |  | control |

### Confirmation extended to seeds 4 and 5

The three candidates that were level with or ahead of DLGM after seeds 2 and 3 were run on two more seeds, again with all three controls. Means are over the four fresh seeds.

| Model | Seed 1 (selection) | Mean of seeds 2-5: WikiText | FineWeb-Edu | vs DLGM: WikiText | FineWeb-Edu | Seeds better than DLGM |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| #259 bias_init_mix_gate | 1.7381 / 2.0692 | 1.7732 | 2.0945 | -0.0259* | -0.0292* | 4 of 4 / 4 of 4 |
| #151 adaptive_mix_gate | 1.7489 / 2.0897 | 1.7823 | 2.1153 | -0.0167* | -0.0083* | 4 of 4 / 3 of 4 |
| **#458 dlgm (DLGM)** | 1.7958 / 2.1162 | 1.7990 | 2.1237 |  |  | control |
| #437 dual_path_fusion | 1.7719 / 2.1173 | 1.7969 | 2.1296 | -0.0021 | +0.0059 | 2 of 4 / 1 of 4 |
| **#879 psfr (Fusion)** | 1.8112 / 2.1424 | 1.8262 | 2.1446 |  |  | control |
| **Transformer (width 96)** | 1.8826 / 2.2214 | 1.8798 | 2.2028 |  |  | control |

Per seed, candidate minus DLGM (WikiText / FineWeb-Edu):

| Candidate | Seed 2 | Seed 3 | Seed 4 | Seed 5 |
| --- | ---: | ---: | ---: | ---: |
| #259 bias_init_mix_gate | -0.0367 / -0.0583 | -0.0225 / -0.0190 | -0.0372 / -0.0045 | -0.0071 / -0.0349 |
| #151 adaptive_mix_gate | -0.0246 / -0.0173 | -0.0010 / -0.0198 | -0.0176 / +0.0164 | -0.0236 / -0.0127 |
| #437 dual_path_fusion | +0.0057 / +0.0162 | -0.0088 / -0.0215 | -0.0094 / +0.0210 | +0.0039 / +0.0080 |

### How much one run moves

The controls' seeds 2 and 3 exist twice: once from the screen and once from the confirmation block. Each pair starts from the same weights and sees the same data in the same order; the only difference is how many updates fitted into 120 CPU-seconds on a shared machine.

| Control | Seed | Updates (screen, confirmation) | WikiText (screen, confirmation) | FineWeb-Edu (screen, confirmation) |
| --- | ---: | ---: | ---: | ---: |
| Transformer (width 96) | 2 | 674, 709 | 1.8487, 1.8298 | 2.1697, 2.1572 |
| Transformer (width 96) | 3 | 679, 649 | 1.8557, 1.8728 | 2.1853, 2.1837 |
| #879 psfr (Fusion) | 2 | 391, 371 | 1.8038, 1.8238 | 2.1560, 2.1273 |
| #879 psfr (Fusion) | 3 | 386, 366 | 1.8082, 1.8218 | 2.1416, 2.1475 |
| #458 dlgm (DLGM) | 2 | 421, 451 | 1.7870, 1.7790 | 2.1287, 2.1068 |
| #458 dlgm (DLGM) | 3 | 408, 396 | 1.7943, 1.8017 | 2.1171, 2.1302 |

The average gap is 0.0142 nats on WikiText and 0.0140 on FineWeb-Edu, the largest 0.0200 and 0.0287. This is why the screen alone decides nothing, and why the controls were re-run inside each confirmation block instead of being reused.

#259 was ahead of DLGM in every fresh seed on both sets. Its lead in a single seed runs from 0.0045 to 0.0583 nats, and 2 of the 8 differences are smaller than the average gap between two runs of the very same thing (next section). Four seeds is still few; the result is a consistent lead of about 0.026 to 0.029 nats per byte, not a measured constant.

### What the leading programs are

Paraphrased from the headers of the programs themselves:

- **#151 `adaptive_mix_gate`** is DeltaNet with one addition: a gate that decides, for every token and every head, how much of the delta-rule memory's output to use and how much of the current value vector.
- **#259 `bias_init_mix_gate`** is #151 plus one parameter per head: a bias on that gate that starts at -1, so the gate starts near 0.27. Its header gives the aim as stopping the gate from collapsing early in training.
- **#437 `dual_path_fusion`** combines the same gate with an additive branch of dilated convolutions.

All three are early and small designs. Their mixers have 52,332, 52,320 and 65,376 parameters against 87,288 for DLGM, which leaves more of the 350,000 for the feed-forward layers and makes an update cheaper: over the four fresh seeds #259 averaged 445 updates in 120 seconds, #151 454, DLGM 408 and the Transformer 648. Whether #259 is ahead of DLGM because its gate is better or because it gets more updates, this experiment cannot say: at equal CPU time the two are the same thing.

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

## Checkpoints

Every run saved its final weights. `code/verify_weights.py` rebuilt each of the 146 models in a fresh process, restored the checkpoint and recomputed both held-out losses: 146 of 146 match the recorded values exactly (difference 0.0) on the machine that made them. [VERIFY.json](VERIFY.json) lists every one.

Two things were learned on the way, and both are handled in the code:

- 33 of the 105 programs count their own forward passes in a buffer that is not part of the `state_dict` and use the count in a schedule. A checkpoint that leaves it out silently resets the schedule. Checkpoints here carry such state under keys starting `__runtime__.` (`run_job.runtime_state`). The first 39 runs were made before this was in place; 9 of them needed the counter, which was rebuilt by replaying the recorded number of forward passes (`code/reconstruct_runtime.py`) and is accepted only because those checkpoints then reproduce exactly.
- BF16 results depend on the instruction set. Reloading with `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` instead of the default (AMX on this machine) moved the held-out losses of the five checkpoints tried by up to 0.00006 nats. On a processor without AMX, expect differences of that size, not exact equality.

Six checkpoints are in `checkpoints/`, each with its result file, all from seed 2 of the confirmation block: the three controls (`T96_s2`, `g879_s2`, `g458_s2`) and the three leading candidates (`g259_s2`, `g151_s2`, `g437_s2`). The other 140 were not kept beyond the session that made them; a hash of each one's weights is in its result file.

```
E=experiments/gallery_cpu120_20261110
# score a text with a checkpoint and let it write a little
unshare -rn -- python -I $E/code/try_checkpoint.py $E/checkpoints/g259_s2.npz inputs/asiarch/106.json work/STATIC_AUDIT.json "Text to continue"
# check that the six checkpoints give their recorded losses
unshare -rn -- python -I $E/code/verify_weights.py $E/checkpoints inputs/asiarch/106.json work/STATIC_AUDIT.json work/data work/VERIFY_checkpoints.json
```

## What happened along the way

- **Four first attempts failed in the harness, not in the programs.** #865, #1370, #1454 and #1543 keep tensors from their last forward pass as attributes; copying the model for the 60-second evaluation refused to copy tensors that still carried gradient history. The copy now detaches such tensors first (`run_job.safe_copy`); the four were re-run from scratch and their first error output is kept in `results/first_attempt_failures/`.
- **The machine restarted twice** (23:25 and 23:51 UTC). Runs in flight were lost and started again from nothing; finished runs were untouched. `results/driver.log` and `results_confirm/driver.log` show every start and finish.
- **A planned pause at 22:06 UTC** added the saving of runtime state described above. It changes what a run writes after it has been measured, not how it trains or what it scores.
- **An instruction-set cap that was meant to be set was not.** The driver was written to pass `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` to its jobs but never did, so every run used oneDNN's default. That was noticed when checkpoints reloaded under the cap did not reproduce exactly. All runs share the default, so the cohort is consistent; the driver's comment now says what it really does, and each result records the setting it saw.

## Files

| Path | What it is |
| --- | --- |
| `RESULTS.md`, `METRICS.json` | The full tables and every derived number |
| `results/`, `results_confirm/` | One JSON per run: configuration, both audits, losses at 60 s and at the end, loss on every held-out window, the training-loss history, a hash of the final weights. `driver.log` has the timeline |
| `VERIFY.json` | The checkpoint reload check for every run |
| `checkpoints/` | Six checkpoints to try, with their result files (see above) |
| `code/` | The harness: `static_audit.py`, `fla_cpu.py`, `models.py`, `data_prep.py`, `smoke.py`, `run_job.py`, `driver.py`, `make_confirm_queue.py`, `summarise.py`, `verify_weights.py`, `reconstruct_runtime.py`, `count_counters.py`, `try_checkpoint.py`, and `make_readme.py` with `README_static.md`, which write this page from the measured numbers |
| `fetch_inputs.sh`, `INPUTS.sha256` | Pinned public inputs and their hashes |
| `STATIC_AUDIT.json`, `SMOKE.json`, `data/DATA_CHECK.json` | The static audit, the build-and-shape check, and the data rebuilt against the v44 ledger |
| `QUEUE.json`, `QUEUE_CONFIRM.json`, `QUEUE_CONFIRM2.json`, `CONFIRM_CHOICE.json`, `COHORT_SETS.json` | What was queued, how the confirmation candidates were picked, and which programs the earlier campaign had screened |

## Not done

- No fresh held-out text. The natural next step for #259 is the test the earlier studies used for their own winners: new documents, more seeds.
- No larger models, no longer runs, no GPU. Whether a gate that helps a 350,000-parameter model for two minutes helps a real one is open.
- No tuning of any program. In particular the programs whose internal schedules assume thousands of steps may be undersold here.

