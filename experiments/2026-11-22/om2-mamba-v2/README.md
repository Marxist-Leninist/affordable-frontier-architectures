# OM2: our Mamba v2, a state-gated exact-token store, and one all-metrics index (22 Nov 2026)

A matched CPU benchmark of eight byte-level language models at about 0.47M parameters, run for the AGI Architecture Index (https://zqx.lat/agi-architectures/). It asked two questions: can our own Mamba be improved, and does the index's first speculative design ("state-gated retrieval", later called the Mamba-4 proposal) help on natural text? Every model is measured on every metric and combined into one index.

## Protocol

- Data: the P056 corpus (WikiText-2 articles, public-domain Project Gutenberg fiction, OpenAssistant oasst1 dialogue; human-written, source-entity split, train/test overlap exclusion). Hashes are in `data/`; the corpus text is not redistributed.
- Every model: 4 layers, d_model 128, byte vocabulary, tied embeddings, about 471-504K parameters; the same 2,000 AdamW updates of 8 x 256 bytes (lr 3e-3, warmup 100, cosine to 10%, weight decay 0.01 on matrices, clip 1.0); the same sampled byte stream for each seed (5617, 5649); final checkpoint, no test-set selection.
- Quality: held-out bits per byte over all 303 test windows (every scored byte sees 129-256 bytes of context), macro-averaged over the three domains. Exact recall: a 32-byte span repeated after 16-176 bytes of another document; next-byte accuracy on its bytes 5-32.
- Speed and memory: one same-machine table (`speed_all.py`: Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz, one thread, each model in its own process, two rounds): seconds per training step, training peak RSS, prefill tokens/s, decode ms/token, decode state bytes. Training runs themselves ran on two machines, so their wall times are not compared.

## One index

| rank | model | family | **combined** | quality (bpb) | exact recall | training speed (s/step) | decode (ms/token) | prefill | state memory at 256 tokens | training memory | bits gained per train minute |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mamba-2 | Mamba | **115.1** | 115 (2.178) | 108 (56%) | 58 (0.38) | 104 (2.3) | 62 | 716 (143 KiB) | 73 (1049 MiB) | 0.534 |
| 2 | Our Mamba v2, positive decay | Mamba (ours) | **112.6** | 116 (2.168) | 107 (56%) | 59 (0.38) | 87 (2.7) | 62 | 716 (143 KiB) | 73 (1049 MiB) | 0.468 |
| 3 | Our Mamba v2 | Mamba (ours) | **109.7** | 116 (2.168) | 108 (56%) | 57 (0.40) | 79 (2.9) | 57 | 716 (143 KiB) | 69 (1105 MiB) | 0.450 |
| 4 | v2 + placebo sidecar | Mamba (ours) + sidecar | **108.7** | 116 (2.163) | 107 (56%) | 56 (0.40) | 76 (3.0) | 56 | 716 (143 KiB) | 69 (1107 MiB) | 0.442 |
| 5 | v2 + state-gated exact store ("Mamba-4" idea) | Mamba (ours) + exact store | **101.6** | 117 (2.157) | 108 (56%) | 55 (0.41) | 75 (3.1) | 55 | 378 (271 KiB) | 68 (1113 MiB) | 0.438 |
| 6 | v2 + always-on exact store | Mamba (ours) + exact store | **101.0** | 117 (2.153) | 108 (56%) | 55 (0.41) | 71 (3.2) | 55 | 378 (271 KiB) | 69 (1099 MiB) | 0.437 |
| 7 | Transformer | Transformer | **100.0** | 100 (2.379) | 100 (52%) | 100 (0.22) | 100 (2.3) | 100 | 100 (1024 KiB) | 100 (761 MiB) | 0.762 |
| 8 | Our Mamba v1 | Mamba (ours) | **84.1** | 117 (2.151) | 108 (56%) | 6 (3.77) | 123 (1.9) | 18 | 2327 (44 KiB) | 22 (3488 MiB) | 0.047 |

Each column is an index with the Transformer = 100 (higher is better), raw value in brackets. Combined = weighted geometric mean: quality 35%, recall 15%, train_speed 15%, decode_speed 15%, state_memory 10%, prefill_speed 5%, train_memory 5%. Rows with a quality index below 80 are shown but not ranked. Speed and memory: one machine (Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz), one thread, every arm in its own process.

## Findings

- Every Mamba-family model beats the Transformer on held-out text by 0.20-0.23 bits/byte here (Mamba-2 minus Transformer -0.2005 [-0.2093, -0.1921] (per seed -0.1916, -0.2094)).
- Our Mamba v1 (per-channel decay, doubling scan) has the best quality (2.1509); Mamba-2 minus v1 +0.0274 [+0.0236, +0.0312] (per seed +0.0336, +0.0211). It needs 3.77 s per training step and 3488 MiB, so it ranks last overall.
- Our Mamba v2 (chunked scalar-decay core with 4x v1's state, normalised B/C, conv on x/B/C, gated norm, signed decay) trains 9.5x faster per step than v1 and gives up 0.0168 bits/byte for it (v2 minus v1 +0.0168 [+0.0127, +0.0209] (per seed +0.0150, +0.0185)). It beats Mamba-2 on quality: -0.0106 [-0.0145, -0.0070] (per seed -0.0186, -0.0026).
- v2's signed decay (transitions in (-1, 1)) changes nothing measurable: -0.0005 [-0.0042, +0.0033] (per seed +0.0009, -0.0018); it slows decoding.
- Exact-token store (one attention head over the last 256 tokens after layer 2): always-on minus same-parameter placebo -0.0102 [-0.0140, -0.0066] (per seed -0.0085, -0.0120); state-gated minus placebo -0.0059 [-0.0098, -0.0020] (per seed -0.0049, -0.0069); state-gated minus always-on +0.0044 [+0.0008, +0.0080] (per seed +0.0036, +0.0051).
- The gate did not become selective. Its sparsity cost (0.02, held off for 500 steps, ramped over 250) drove the mean gate to 0.070 with 0.0% of tokens above 0.5; evaluated with reads only where the gate exceeds 0.5 it read on 0.0% of tokens and scored 2.1977, losing the gain. Reading only when needed remains unproven; the next test should reward abstention explicitly or train the gate against a recall-failure signal.
- The recall probe did not separate the models: all scored 52-57%, and no model predicted the repeated span better than its first occurrence (negative copy gain in `results/RESULTS.md`). At this scale none has learned in-context copying, so the recall column carries little information.

## Reproduce

```
python3 make_jobs.py            # writes jobs.json (16 runs)
python3 om2_queue.py 4           # or om_queue_vast.py 8 jobs.json on a shared box
python3 summarize.py runs        # RESULTS.md, summary.json
python3 speed_all.py tf,m2k,om1,om2,om2pos,om2p,om2a,om2g speed.json 2
python3 index_all.py runs speed.json index/
```

The data files (`data/train.bin`, `dev_windows.json`, `test_windows.json`) are not included; their sha256 are in `data/SHA256SUMS`. The `m2` arm needs transformers 4.57.6 (its definition and initialisation are copied from Mamba2ForCausalLM; logits match to 4e-6).

## Files

- `om2bench.py`: all models, training, evaluation, recall probes, timing; `om3bench.py`: adds the `m2k` alias (m2 compute without transformers) used by the speed table
- `om2_queue.py`, `om_queue_vast.py`, `make_jobs.py`, `jobs.json`: job running; `summarize.py`, `speed_all.py`, `index_all.py`: analysis
- `results/`: `INDEX.md` / `index.json` (the one index), `RESULTS.md` / `summary.json` (quality, recall, contrasts), `speed_table.json`, `runs/*.json` (per run, with per-document scores and dev curves)
- `checkpoints/`: 16 safetensors state dicts (load into `om2bench.build(arm)`), `CHECKPOINTS.json` with sha256
- `data/`: corpus manifests and hashes; `THIRD_PARTY_NOTICES.md`; `SHA256SUMS`

## Limits

- Small models, one corpus, two seeds, CPU: a matched comparison of these designs, not a ranking of architecture families at scale.
- Index weights are a stated choice (see the table note); the per-metric columns are given so other weightings can be computed.
- Speed is one CPU model and one thread; the training-time column is seconds per step from the speed table.

