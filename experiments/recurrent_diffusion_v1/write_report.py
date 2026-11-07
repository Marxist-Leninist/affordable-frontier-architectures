from pathlib import Path
import json
P=Path(__file__).parent;s=json.loads((P/'summary.json').read_text());p=json.loads((P/'profile.json').read_text());d=json.loads((P/'diagnostic.json').read_text());validation=json.loads((P/'validation.json').read_text())
labels={'bptt':'Four-step end-to-end BPTT','diffusion_weighted':'Single-step denoising, sample-weighted CE','diffusion_ce':'Single-step denoising, unweighted CE','transformer':'Ordinary two-layer causal Transformer'}
table='| Method | Params | Mean test BPB | Seed 1 / 2 / 3 |\n|---|---:|---:|---|\n'
for r in s['equal_data']:table+=f"| {labels[r['kind']]} | {r['parameters']:,} | {r['mean_test_bpb']:.3f} | "+' / '.join(f'{z:.3f}' for z in r['seed_test_bpb'])+' |\n'
extra='| Method | Seed | Steps | Budget fraction | Test BPB | Endpoint |\n|---|---:|---:|---:|---:|---|\n'
for r in s['extra_training']:extra+=f"| {labels[r['kind']]} | {r['seed']} | {r['steps']} | {r['cpu_budget_fraction']:.3f} | {r['test_bpb']:.3f} | {r['endpoint']} |\n"
diag='| Method | Depth | Particles | Validation BPB |\n|---|---:|---:|---:|\n'
for r in d['rows']:
 if 'depth' in r:diag+=f"| {labels[r['kind']]} | {r['depth']} | {r['particles']} | {r['bpb']:.3f} |\n"
b,w=p['rows'][:2];speed=b['median_step_cpu_seconds']/w['median_step_cpu_seconds'];mem=b['unique_saved_nonparameter_storage_bytes']/w['unique_saved_nonparameter_storage_bytes']
readme=f'''# Recurrent-depth denoising: runnable independent pilot

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

{table}
Lower bits per byte (BPB) is better. Test scores cover 8,192 target bytes in 256 non-overlapping, evenly spaced windows from the official WikiText-2 test split. The last 32 targets of each 64-byte context are scored. Validation is a separate official split, with 4,096 scored targets. Full token losses are summarized as per-window averages in each endpoint file. No seed or hyperparameter was selected using test results.

The probability being scored is q(byte | clean prefix) = the arithmetic mean of the final softmax probabilities from four fixed, target-independent Gaussian trajectories. Scoring uses log(mean probability), not mean(log probability). This is an explicitly defined finite predictive decoder, **not exact diffusion likelihood, ELBO, the paper's teacher-model generative perplexity, or a MAUVE score**. A small number of particles can badly approximate a sharp implicit distribution.

## Extra training

The cheaper methods continued toward the corresponding BPTT run's measured training CPU time, with a predeclared cap of 3,000 updates. The first BPTT run included unusually high cold-start overhead, so all seed-1 continuations hit the cap before its budget. Those are marked `capped_cpu`; do not describe them as exactly compute-matched.

{extra}
Training times include optimization/loss work and exclude evaluation, data extraction, dependency setup and failed/replayed jobs. CPU time is hardware/runtime dependent. See [execution notes](execution_notes.json); no new paid resources or production services were used.

## Cost measured directly

On the recorded one-thread CPU, isolated warmed optimization steps took {b['median_step_cpu_seconds']*1000:.2f} ms for BPTT and {w['median_step_cpu_seconds']*1000:.2f} ms for weighted single-step training: **{speed:.2f}x step speedup**.

Unique non-parameter storage retained for backward was {b['unique_saved_nonparameter_storage_bytes']/2**20:.2f} MiB versus {w['unique_saved_nonparameter_storage_bytes']/2**20:.2f} MiB, a **{mem:.2f}x reduction in this activation proxy**. This is not total peak RSS or GPU VRAM. Both recurrent models retain the same parameter and optimizer storage. The profiler's recognized FLOP counts are incomplete because fused attention and elementwise operations are not fully counted.

Both recurrent decoders use 4 core calls per particle at inference. With four particles, the primary score therefore needs 16 core calls per context; the ordinary Transformer needs one two-layer forward pass. Training savings are not inference savings.

## What was implemented

The independent model has width 48, four attention heads, FFN width 128, a 256-byte vocabulary and 64-byte learned positional embeddings. One clean causal self-attention block encodes the prefix. A shared cross-attention/FFN core reads a noisy target state, the clean prefix and sinusoidal log-sigma conditioning. Target queries cannot read other noisy targets. For target at index i+1, context access stops at input index i.

The target codebook is the L2-normalized trainable token embedding. The model uses EDM input/output/skip preconditioning with sigma-data 0.5. Training noise is log-normal, log(sigma) ~ Normal(-1.2, 1.2), clamped to [0.002,80]. The weighted variant applies samplewise (sigma²+0.25)/(0.5 sigma)² to token CE, divided by a fixed 500 for numerical loss scale. The unweighted variant was specified before results were observed. Both use the same AdamW settings as BPTT, with no tuning sweep; different objectives may need different tuning.

At inference, Karras rho=7 levels descend from 80 to 0.002. Each Euler update is z_next = z + (sigma_next - sigma)/sigma * (z - D), with D the softmax-weighted mean of normalized codebook vectors. The BPTT control differentiates through this exact four-call graph from pure noise. It is a valid matched unrolled control, not the original Huginn architecture.

## Diagnostic, not model selection

After observing poor predictive scores, a validation-only check varied inference depth and particle count on 512 targets from 16 windows, using seed 1 only. It did not change any primary endpoint or select a checkpoint. It uses the last predeclared extra-training checkpoint for single-step methods and the 600-step BPTT checkpoint.

{diag}
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

All {validation['endpoints']} checkpoints passed finite-value, parameter-count, SHA-256 and metric-mean checks. Four seed-1 held-out scores were re-evaluated from saved checkpoints exactly. Source/data hashes and causality/prefix/oracle-Euler checks passed. [Environment](environment.json) and [artifact manifest](artifact_manifest.json) identify the runtime and files.

## Relation to the original work

The [paper](https://arxiv.org/html/2506.14202v4) describes a much larger Huginn-style LM, LM1B data, average 32 recurrences and 8-step truncated BPTT. This pilot instead uses a tiny byte model, WikiText-2, four fully differentiated recurrences, a different attention topology and a specified finite-mixture decoder. It does not validate or refute the reported larger-scale improvement.

The [official repository](https://github.com/SakanaAI/DiffusionBlocks) currently releases ViT code; [issue 3](https://github.com/SakanaAI/DiffusionBlocks/issues/3) requests the recurrent LM implementation. Its sampler uses the correct negative-dt Euler update. Its ViT CE weight broadcast is not samplewise; this pilot explicitly tests intended samplewise weights plus an unweighted alternative. That observation is not attributed to the unreleased LM implementation.

The code and results are useful for testing refinements. A convincing follow-up should reproduce an available LM implementation, include inference-distribution/particle sensitivity, and compare quality at matched training and inference costs on a larger held-out set.
'''
(P/'README.md').write_text(readme)
print('README written',len(readme))
