# Recurrent-depth denoising: runnable independent pilot

This is tested code and raw results for a small language-model experiment inspired by DiffusionBlocks. Trained checkpoints were retained locally; this GitHub snapshot contains their hashes, not the weight files. It is **not a reproduction of the paper's Huginn result**. The original recurrent-depth LM implementation/configuration is not released in the official repository as checked on 2026-11-07.

**Result:** the single-step methods trained more cheaply, but the frozen four-step/four-particle predictive decoders were substantially worse than end-to-end training. This pilot does not establish a quality-preserving training-cost win. Its finite-mixture log scores do not establish the likelihood or generation quality of the continuous diffusion model.

**Crucial diagnostic:** on a small validation subset, the extra-trained unweighted model's one-pass high-noise head scored 3.43 BPB, while four-step decoding scored 10.68 with four particles and 4.73 with 32 particles. This points to a substantial decoding/scoring problem, not a blanket inability to learn language. This is one seed and 512 validation targets, not a new confirmed test-set win. All diagnostic variants appear below.

## Start here

- [Implementation](models.py), [training and scoring](run.py), [causality and Euler checks](checks.py).
- [Frozen protocol](protocol.json), [summary](summary.json), [all raw endpoint losses](results.json).
- [Validation](validation.json), [compute accounting](profile.json), [post-hoc decoder diagnostic](diagnostic.json).
- [Checkpoint SHA-256 manifest](checkpoint_manifest.json). Reproduction generates the weight files.

## Matched training data

Each seed gets the exact same first 600 randomly sampled training batches, 16 sequences of 64 bytes: **614,400 target-byte exposures**. The three paired seeds are 1, 2 and 3. All recurrent variants have identical initialization, trainable parameters and inference graph; their training objective/number of recurrent passes differs. The smaller conventional Transformer is a secondary sanity control.

| Method | Params | Mean test BPB | Seed 1 / 2 / 3 |
|---|---:|---:|---|
| Four-step end-to-end BPTT | 77,216 | 4.107 | 4.184 / 4.194 / 3.943 |
| Single-step denoising, sample-weighted CE | 77,216 | 11.955 | 12.440 / 12.012 / 11.414 |
| Single-step denoising, unweighted CE | 77,216 | 8.385 | 9.243 / 7.362 / 8.550 |
| Ordinary two-layer causal Transformer | 71,744 | 3.592 | 3.580 / 3.594 / 3.602 |

Lower bits per byte (BPB) is better. Test scores cover 8,192 target bytes in 256 non-overlapping, evenly spaced windows from the official WikiText-2 test split. The last 32 targets of each 64-byte context are scored. Validation is a separate official split, with 4,096 scored targets. Full token losses are summarized as per-window averages in each endpoint file. No seed or hyperparameter was selected using test results.

The probability being scored is q(byte | clean prefix) = the arithmetic mean of the final softmax probabilities from four fixed, target-independent Gaussian trajectories. Scoring uses log(mean probability), not mean(log probability). This is an explicitly defined finite predictive decoder, **not exact diffusion likelihood, ELBO, the paper's teacher-model generative perplexity, or a MAUVE score**. A small number of particles can badly approximate a sharp implicit distribution.

## Extra training

The cheaper methods continued toward the corresponding BPTT run's measured training CPU time, with a predeclared cap of 3,000 updates. The first BPTT run included unusually high cold-start overhead, so all seed-1 continuations hit the cap before its budget. Those are marked `capped_cpu`; do not describe them as exactly compute-matched.

| Method | Seed | Steps | Budget fraction | Test BPB | Endpoint |
|---|---:|---:|---:|---:|---|
| Single-step denoising, sample-weighted CE | 1 | 3000 | 0.883 | 13.822 | capped_cpu |
| Single-step denoising, unweighted CE | 1 | 3000 | 0.905 | 10.287 | capped_cpu |
| Ordinary two-layer causal Transformer | 1 | 3000 | 0.882 | 2.899 | capped_cpu |
| Single-step denoising, sample-weighted CE | 2 | 1763 | 1.000 | 13.128 | equal_cpu |
| Single-step denoising, unweighted CE | 2 | 1775 | 1.000 | 8.844 | equal_cpu |
| Ordinary two-layer causal Transformer | 2 | 1841 | 1.000 | 3.090 | equal_cpu |
| Single-step denoising, sample-weighted CE | 3 | 1651 | 1.000 | 13.047 | equal_cpu |
| Single-step denoising, unweighted CE | 3 | 1966 | 1.000 | 9.580 | equal_cpu |
| Ordinary two-layer causal Transformer | 3 | 2199 | 1.001 | 2.994 | equal_cpu |

Training times include optimization/loss work and exclude evaluation, data extraction, dependency setup and failed/replayed jobs. CPU time is hardware/runtime dependent. See [execution notes](execution_notes.json); no new paid resources or production services were used.

## Cost measured directly

On the recorded one-thread CPU, isolated warmed optimization steps took 41.84 ms for BPTT and 15.88 ms for weighted single-step training: **2.63x step speedup**.

Unique non-parameter storage retained for backward was 19.32 MiB versus 6.85 MiB, a **2.82x reduction in this activation proxy**. This is not total peak RSS or GPU VRAM. Both recurrent models retain the same parameter and optimizer storage. The profiler's recognized FLOP counts are incomplete because fused attention and elementwise operations are not fully counted.

Both recurrent decoders use 4 core calls per particle at inference. With four particles, the primary score therefore needs 16 core calls per context; the ordinary Transformer needs one two-layer forward pass. Training savings are not inference savings.

## What was implemented

The independent model has width 48, four attention heads, FFN width 128, a 256-byte vocabulary and 64-byte learned positional embeddings. One clean causal self-attention block encodes the prefix. A shared cross-attention/FFN core reads a noisy target state, the clean prefix and sinusoidal log-sigma conditioning. Target queries cannot read other noisy targets. For target at index i+1, context access stops at input index i.

The target codebook is the L2-normalized trainable token embedding. The model uses EDM input/output/skip preconditioning with sigma-data 0.5. Training noise is log-normal, log(sigma) ~ Normal(-1.2, 1.2), clamped to [0.002,80]. The weighted variant applies samplewise (sigma²+0.25)/(0.5 sigma)² to token CE, divided by a fixed 500 for numerical loss scale. The unweighted variant was specified before results were observed. Both use the same AdamW settings as BPTT, with no tuning sweep; different objectives may need different tuning.

At inference, Karras rho=7 levels descend from 80 to 0.002. Each Euler update is z_next = z + (sigma_next - sigma)/sigma * (z - D), with D the softmax-weighted mean of normalized codebook vectors. The BPTT control differentiates through this exact four-call graph from pure noise. It is a valid matched unrolled control, not the original Huginn architecture.

## Diagnostic, not model selection

After observing poor predictive scores, a validation-only check varied inference depth and particle count on 512 targets from 16 windows, using seed 1 only. It did not change any primary endpoint or select a checkpoint. It uses the last predeclared extra-training checkpoint for single-step methods and the 600-step BPTT checkpoint.

| Method | Depth | Particles | Validation BPB |
|---|---:|---:|---:|
| Four-step end-to-end BPTT | 1 | 4 | 8.605 |
| Four-step end-to-end BPTT | 1 | 32 | 8.310 |
| Four-step end-to-end BPTT | 4 | 4 | 4.171 |
| Four-step end-to-end BPTT | 4 | 32 | 4.173 |
| Four-step end-to-end BPTT | 32 | 4 | 4.820 |
| Four-step end-to-end BPTT | 32 | 32 | 4.823 |
| Single-step denoising, sample-weighted CE | 1 | 4 | 3.959 |
| Single-step denoising, sample-weighted CE | 1 | 32 | 3.708 |
| Single-step denoising, sample-weighted CE | 4 | 4 | 13.567 |
| Single-step denoising, sample-weighted CE | 4 | 32 | 5.636 |
| Single-step denoising, sample-weighted CE | 32 | 4 | 18.765 |
| Single-step denoising, sample-weighted CE | 32 | 32 | 7.624 |
| Single-step denoising, unweighted CE | 1 | 4 | 3.554 |
| Single-step denoising, unweighted CE | 1 | 32 | 3.430 |
| Single-step denoising, unweighted CE | 4 | 4 | 10.681 |
| Single-step denoising, unweighted CE | 4 | 32 | 4.725 |
| Single-step denoising, unweighted CE | 32 | 4 | 13.979 |
| Single-step denoising, unweighted CE | 32 | 32 | 6.109 |

The same file also compares label-conditioned denoising CE at several sigma values. That quantity is not held-out predictive loss: it intentionally receives a corrupted target.

## Reproduce

Use Python 3.12 and the recorded package versions in [requirements.txt](requirements.txt). A CPU-only setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.3.5 pyarrow==19.0.1
bash reproduce.sh fresh_run
```

The script creates a new directory, downloads pinned public WikiText-2 parquet data, verifies parquet and serialized-text hashes, runs checks, freezes the protocol, and runs short sequential jobs. [Data provenance](source_data_manifest.json) pins revision b08601e04326c79dfdd32d625aee71d232d685c3. Equal-data scores are deterministic on the recorded stack; time-budget endpoints vary by machine. Reproduced checkpoints can be scored with `torch.load(..., weights_only=True)` using `models.py` and `run.evaluate`.

All 21 checkpoints passed finite-value, parameter-count, SHA-256 and metric-mean checks. Four seed-1 held-out scores were re-evaluated from saved checkpoints exactly. Source/data hashes and causality/prefix/oracle-Euler checks passed. [Environment](environment.json) and [artifact manifest](artifact_manifest.json) identify the runtime and files.

## Relation to the original work

The [paper](https://arxiv.org/html/2506.14202v4) describes a much larger Huginn-style LM, LM1B data, average 32 recurrences and 8-step truncated BPTT. This pilot instead uses a tiny byte model, WikiText-2, four fully differentiated recurrences, a different attention topology and a specified finite-mixture decoder. It does not validate or refute the reported larger-scale improvement.

The [official repository](https://github.com/SakanaAI/DiffusionBlocks) currently releases ViT code; [issue 3](https://github.com/SakanaAI/DiffusionBlocks/issues/3) requests the recurrent LM implementation. Its sampler uses the correct negative-dt Euler update. Its ViT CE weight broadcast is not samplewise; this pilot explicitly tests intended samplewise weights plus an unweighted alternative. That observation is not attributed to the unreleased LM implementation.

The code and results are useful for testing refinements. A convincing follow-up should reproduce an available LM implementation, include inference-distribution/particle sensitivity, and compare quality at matched training and inference costs on a larger held-out set.
