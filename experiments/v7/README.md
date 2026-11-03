# v7: a real-text quality comparison, not an intelligence verdict

2026-11-03. Prior v6 commit 802a234f641e4deabd8a35a768ebccb8080fc8e1. Question: does a recurrent model exceed a Transformer baseline in meaningful quality? This bounded experiment replaces synthetic templates with actual Shakespeare text and strengthens the Transformer. It does not measure general intelligence.

## Result

On this fixed small-model, short-training, held-out character-prediction task, the recurrent model achieved lower test cross-entropy in every paired seed:

| Model | Seed 1 bpc | Seed 2 bpc | Seed 3 bpc | Mean bpc |
|---|---:|---:|---:|---:|
| Normalized four-head Transformer | 3.358134 | 3.390547 | 3.381992 | 3.376891 |
| Vanilla recurrent LM | 3.189773 | 3.173202 | 3.167233 | 3.176736 |

Lower bits per character (bpc) is better. The mean difference is 0.200155 bpc, about 5.93% lower cross-entropy. This is not a percentage improvement in intelligence or accuracy. Three seeds give only a small estimate of training variability, not robustness across datasets or architectures.

**Supported:** this particular tiny recurrent baseline predicts the selected unseen real-text slice better under this fixed training/tuning protocol. **Not supported:** recurrence beats strong Transformers generally, matches frontier quality, or delivers AGI or a profitable $500 service.

## Better baseline and controlled budget

- Transformer: one block, width 32, four causal attention heads, learned affine pre-LayerNorm for attention and FFN, residuals, 4x tanh FFN, final LayerNorm and canonical sinusoidal positions. 16,800 parameters.
- Recurrent: vanilla tanh recurrent cell, width 80, learned input embeddings, hidden bias, output projection. 16,880 parameters, 0.48% more. This is not Mamba, LSTM, AGILLM or a new architecture.
- Identical 400 updates × 8 sequences × 64 characters = 204,800 training characters per run. All parameters trained with Adam, beta1 .9, beta2 .999, epsilon 1e-8, global gradient clipping 1, float64. Paired seeds use identical sampled training-window positions.
- Equal learning-rate selection budget: .001, .003, .01 on tuning seed 101, chosen by validation bpc only. Both select .01. Selection is saved before final seeds 1/2/3 access test scores. No best-seed selection, early stopping or additional tuning after viewing test outcomes.
- Equal parameters/tokens and tuning-run count do NOT establish equal FLOPs, wall time, memory, or globally optimal training. Both best learning rates lie at the grid endpoint; an optimum is not established. Training is deliberately short.

The Transformer is materially stronger than v5's unnormalized single-head block, but remains a tiny one-layer character LM. Tanh is used rather than a modern gated FFN; architecture depth, positional methods, regularization, optimizer schedule, compute budget and larger tuning searches remain untested. The large width difference also changes representation capacity. An architecture-wide winner cannot be chosen here.

## Real text and held-out protocol

Pinned source: https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt . Source Git blob: 7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9. The public-domain Shakespeare text is a real corpus, not generated templates. Exact 1,115,394 bytes, SHA256 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.

Contiguous 80/10/10 split with a discarded 64-character gap after each boundary: 892,315 training characters, 111,475 validation, 111,476 test. Vocabulary is derived from training only; all held-out characters are representable. First 16,384 target positions from each held-out split are scored in disjoint 64-character segments, with state/context reset per segment. This is approximately 14.7% of each held-out partition, not full-partition scoring. No evaluated exact 64-character context occurs anywhere in training (`data_audit.json`), although shorter repeated phrases naturally exist. Contiguous slices may differ in content and style; this single historical-text corpus cannot establish broad generalization.

`protocol.json` was written before training, with fixed models, tuning grid, budgets and scoring rule. Initial tool materialization inadvertently appended a newline. Independent review detected the source Git-blob mismatch. That entire preliminary run was superseded, the exact source bytes restored, and the identical protocol rerun. All published primary results and hashes here use corrected bytes; no hypothesis, learning-rate grid, seed or training budget was altered after seeing the preliminary test results.

## Verification and reproduce

With Python 3.10+ and NumPy already available:

    python prepare_data.py
    OPENBLAS_NUM_THREADS=1 python test_models.py
    python run.py

Data preparation downloads only the pinned source URL and verifies SHA256, or checks existing input.txt. During this experiment the connected GitHub tool supplied source bytes; no packages or paid compute were acquired. `run.py` sets one-thread BLAS before NumPy import. Do not use `python -m unittest` for this assertion-based test script; invoke it directly.

`test_models.py` samples finite differences across all 16 Transformer and four recurrent parameter groups, maximum absolute error below 5.4e-11. It tests causal prefix isolation, truncation parity, batch probability/loss/gradient invariance, and an SGD descent step. These are numerical checks, not formal verification. `results.json` preserves all tuning/final runs, source/weight/data hashes, training traces and actual CPU timings; `selection.json` preserves validation selection. `audit.md` records independent review.

No optimized incremental latency comparison was run in v7. Actual training seconds are environment observations, not serving cost. No production processes, rentals, access permissions or paid resources changed.

## Next quality gate

Before making an intelligence claim, require multiple held-out language, reasoning, retrieval, coding and tool-use tasks, stronger/deeper well-tuned Transformer controls, sufficiently trained models, larger paired-seed samples, and matched compute as well as token-budget views. Preserve the full heavy-user workload and quality/latency requirements when later measuring cost. This study is an incremental quality signal, not a substitute for those gates.
