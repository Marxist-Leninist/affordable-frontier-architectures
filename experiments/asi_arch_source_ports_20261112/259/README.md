# BAMG #259: isolated third-arm preparation

**Status: CPU correctness preparation only. No comparative training, no GPU jobs, no queue changes, no live service changes, and no claim that this breaks the Fusion/DLGM tie.**

## Identity and source

- Gallery index: 259.
- Source name: delta_net_bias_init_mix_gate.
- Gallery display name: BAMG_MemoryGate.
- Source metadata says 411.95M parameters. This is not the simulator's parameter count and is not proof of matched source-model size.
- Upstream ASI-Arch commit: 3113c51b04a1d4a7520a3b0238c6a08d3194d58f.
- Catalog JSON SHA-256: aa77e8289971b3922102f000fddf44a6a9a31ca1f3b59c834722ac130f63759c.
- Canonical program SHA-256: 08e3b4bb618c64be48db8cb402ab3dc46cbc1a204741710f25afb6fedc3b699e.
- Frozen simulator SHA-256: dc5c3c28ddbf18538ff9c7fd7de0679fd6f3fc19215284937d127d54732ee7d2.

The exact extracted source and transformed reference are retained locally as upstream259.py and reference259_dtype_preserving.py. preparation_results.json records both hashes.

## Important source finding

The source initializes BOTH mix_proj.bias and separate mix_bias to -1, then adds both. Therefore the gate at zero hidden input is sigmoid(-2) = 0.1192029220. Its prose says approximately 0.27, which would correspond to only one -1 bias. The port preserves the actual double-bias semantics, not the prose. With nonzero hidden input the randomly initialized mix projection also contributes.

## What is preserved and what is adapted

Preserved mixer structure:
- Three causal four-tap short convolutions with SiLU.
- L2-normalized q and k, beta from a bias-free projection.
- Plain delta rule with chunk size 32.
- A per-token, per-head sigmoid mixing the delta-memory output and instantaneous convolved value.
- Both -1 biases and a learnable mix-projection weight.

Explicit common comparison adaptations:
- The existing simulator's binary qkv/o matrices, block-wide output RMS and shared expert/objective machinery remain in place.
- Candidate mixer on three of four layers; every fourth and final layer retains baseline attention. Mixer remains causal in all objective modes.
- Dirac short-convolution initialization matches current Fusion/DLGM port convention, rather than asserting original upstream initialization.
- The source hardcodes a BF16 intermediate. The port uses a dtype-preserving triangular solve suitable for the FP32 simulator. This is an explicitly adapted arithmetic path.
- No low-rank or head-normalized variant is silently substituted. Plugin ID is bamg259.
- No incremental decoder or cache API is implemented. The inherited kv_bytes_per_token hook remains all-attention accounting and must not be advertised as a measured recurrent-cache saving.

## Verification scope

Copied-weight comparisons expose the raw mixer by replacing the upstream output norm/projection with identity modules; q/k/v, beta, mixing weights and convolution weights are copied. Thus equivalence is **mixer-only**, not complete original model/block equivalence.

The transformed reference removes @torch.compile and changes only the hardcoded BF16 cast to the current value dtype. CPU stand-ins supply FLA short convolution and normalization. This is not a Triton/GPU validation.

Tests cover:
- Output, input gradient and mapped parameter gradients for float64 and float32, lengths 1, 31, 32, 33, 63, 64, 65, 127 and 257.
- Causality on both sides of chunk boundaries and batch isolation.
- Attention-layer delegation, mixer causality even when the objective is noncausal, and double-bias initialization.
- Actual frozen simulator forward/backward and finite optimizer microsteps in AR, SAT-fixed, SAT-variable, NAT and local-denoise modes.
- Full simulator causality and batch isolation.
- Allocation-free module census at 560 layers.

The unmodified upstream BF16 core is also run separately with copied weights under CPU autocast. Its maximum output difference from the FP32 port is measured, not labeled an exact match. Full results are in preparation_results.json.

The synthetic microstep losses are only finite-function checks, not quality evidence.

## 560-layer accounting

Geometry: D=8192, H=64, head width 128, 560 layers, 140 attention layers, 420 candidate layers, 1372 experts.

Meta-device census preserves all 886,910,746,624 binary-body latent bytes and the existing 1,638,924,288 table bytes. Base total is 888,700,121,820 parameters.

The full BAMG adaptation adds 481,743,360 FP parameters:
- Gate and beta projections/biases: 440,455,680.
- Short-convolution weights: 41,287,680.
- Additional fixed buffers: zero.

FP32 weights, gradients, Adam first moments and Adam second moments are each 1,926,973,440 bytes, totaling 7,707,893,760 bytes if all reside simultaneously. These are storage/accounting figures, **not predicted live GPU peak**. Existing optimizer buffers, activations, workspaces and transfer costs are additional.

A possible future batch-1 FP32 streaming implementation at context 2048 would use about 20,675,952,640 bytes for attention KV plus one delta-state matrix and three short-convolution histories per candidate layer. That design is not implemented or tested here.

## Selection evidence

The historical 350k-parameter / 120-CPU-second screen on reused text favors #259 over DLGM in all four fresh seeds on both datasets. This is a reasonable selection signal, not independent final confirmation.

Paired seed-level t intervals, candidate minus DLGM (lower is better):
- WikiText mean -0.0258591 nats, df=3 95% interval [-0.0485328, -0.0031853].
- FineWeb-Edu mean -0.0291608 nats, df=3 95% interval [-0.0658615, +0.0075400].

The FineWeb seed interval includes zero. These are exploratory, unadjusted intervals after candidate selection; document bootstrap intervals are not a substitute for seed uncertainty.

## Before comparative enrollment

The coordinator must first verify safe capacity and owner agreement. The existing runner's serving-admission and VRAM-retention concerns must be resolved; this preparation does not authorize a new job.

Freeze a separate three-arm v2 protocol without altering existing index v1:
- Same practical adaptation, dataset/windows, token budget, seed schedule, objective and tuning allowance for all arms.
- Full, LR and LRK are separate comparisons.
- Primary quality metric fixed as 0.6 AR + 0.2 SAT + 0.2 NAT; diagnostics not selected afterward.
- Baseline timing reruns do not inflate independent seed counts.
- Three seeds are pilot evidence only. Pre-fix an additional confirmation cohort and multiplicity-adjusted paired three-arm tests before results; no optional stopping.
- Reused text is not called fresh holdout.
- Measured same-window timing is distinct from the old simulator heuristic.
- Raw source/config/data hashes and causality, finite-gradient and zero-skip gates accompany every result.
- No serving-readiness credit without an actual cached decoder and correctness tests.

## Reproduce the isolated tests

Set CUDA_VISIBLE_DEVICES empty, OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS to 1. Run with a 240-second timeout, nice priority 19 and a network-disabled namespace:

python3 -I test_bamg259.py PATH_TO_PINNED_106_JSON PATH_TO_GALLERY_METRICS_JSON

The test requires PyTorch, NumPy, einops, the copied frozen agillm_mini.py and the historical fla_cpu.py CPU reference helper. It writes only the extracted references and preparation_results.json in this isolated directory. The exact tested environment is recorded in that report.

The pinned catalog and historical metrics are also published in this repository under experiments/gallery_cpu120_20261110/campaign_coverage_20261112/UPSTREAM_105.json and experiments/gallery_cpu120_20261110/METRICS.json. These preparation results predate comparative GPU enrollment.
