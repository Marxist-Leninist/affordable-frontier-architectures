# AdaptiveGateNet-AFP #712: full practical-Dirac AGILLM adapter

Status: **correctness preparation only; untrained**. This bundle prepares one full-width practical-Dirac hybrid. It adds **zero trained source IDs**. Forty-one bounded CPU checks pass. No optimizer updates, binary-latent updates, model-service requests, GPU jobs, quality/throughput benchmark, queue changes, or live deployments were performed. This bundle was not published by its preparer.

## Source identity and included evidence

- Repository: https://github.com/GAIR-NLP/ASI-Arch
- Commit: `3113c51b04a1d4a7520a3b0238c6a08d3194d58f`
- File: `data/106.json`, zero-based record 25, index 712, parent 471.
- Source identifier: `delta_net_afp`; catalog title: `AdaptiveGateNet-AFP`.
- Complete catalog SHA256: `aa77e8289971b3922102f000fddf44a6a9a31ca1f3b59c834722ac130f63759c`.
- Exact extracted program SHA256: `62d59f5bf7f19ba502343f2c1aeba41311659f981660cd0feb37a58940451b7e`.

The catalog digest was verified during extraction. `upstream712.py` is the exact UTF-8 program, not an edited reconstruction. `source_entry.json` retains catalog metadata, not a new experimental result. The historical catalog parameter/score fields do not describe this AGILLM adapter and are not used for comparison. `provenance.json` identifies the frozen helper and source hashes. `MANIFEST.sha256` covers the portable deliverable.

`reference712_dtype_preserving.py` is derived mechanically from the extracted source by only removing `@torch.compile` and replacing the forced BF16 inverse cast with `inv.to(v.dtype)`. The test asserts the complete transformation exactly. Numerical reference execution uses `fla_cpu.py`, explicitly a plain-PyTorch dependency shim, not official fused FLA kernels. At the mixer comparison boundary, reference output norm and projection are replaced by identity and all mapped weights are copied. The unchanged literal source is separately checked for a finite BF16 forward only; this is not a BF16 equivalence result.

## Public adapter interface and integration

`afp712_dirac.py` exports `META`, `patch(Block, Model, cfg)` and the resulting block's `mix(h, causal)`. `patch` returns the subclass and unchanged host Model. The last block and every fourth block remain original host attention. At L=560 this means 420 mixer and 140 attention blocks; at arbitrary L the forced final attention block can change the exact fraction.

The host owns the fused binary qkv projection, binary output projection and block-wide output RMS. The adapter replaces only the mixer. It does not keep the source's separate floating-point q/k/v/o projections, optional output gate, or per-head output RMS. Host output RMS and output projection each execute once. This is a practical AGILLM adaptation, not reproduction of the complete original DeltaNet layer.

The defaults implemented are full key/value width, beta enabled, negative-eigenvalue option disabled, SiLU q/k/v short convolutions, source L2 delta normalization, base floor 0.05, and source 5-/64-tap FIRs. The adapter does not provide alternate source configuration modes. Host attention retains its original causal/noncausal behavior; mixer layers remain causal for every objective, including when `causal=False`.

## Preserved mathematical contract

For full width d, H heads and hd=d/H:

1. Host qkv is split into q/k/v, each with an independent 4-tap per-channel causal convolution and SiLU. Beta is sigmoid of a bias-free d-to-H linear projection.
2. Delta memory normalizes q and k using `rsqrt(sum(x*x)+1e-6)`. The default chunk size is 32. The port uses an equivalent lower-triangular solve and carries the ordinary delta-rule state across chunks, without inventing a retention factor. Right padding is removed from the returned output.
3. The short and long FIRs are independent per head and channel, with 5 and 64 taps. They operate on the activated direct-value branch with causal left padding.
4. Each of the short, long and delta branches supplies four statistics: head-axis mean, head-axis mean square, head-axis mean absolute value, and its own head's feature L2 norm broadcast over features. The first three statistics average over H at the same token, **not over time**. Each branch supplies 4hd features per head.
5. Gate input is full token h concatenated with those three branch-statistic vectors, dimension d+12hd. Gate weight is [H,d+12hd,4], gate bias [H,4]. Branch order is short FIR, long FIR, delta, direct value.
6. Per-path temperature is `exp(log_temp).clamp(0.1,10)`. Softmax applies to gate logits divided by this temperature. Per-head/path floor is `0.05*sigmoid(floor_param)`. Final weights are `floor + (1-sum(floor))*softmax`. These nonnegative weights sum to one. Neither the floor nor the temperature is folded away.

## Initialization and explicit departures

**Source-native retained initialization:** FIR weights are independent `Normal(0,0.02^2)` with no identity tap. Gate weights use the source's literal `kaiming_uniform_(weight,a=sqrt(5))` on the three-dimensional tensor. Its effective fan-in is 4*(d+12hd); replacing it with an independently initialized conventional linear layer would differ. Gate bias is [-0.5,-0.5,0.5,1.5] per head. Floor parameters and log temperatures start at zero, so each initial floor is 0.025 and each temperature is 1. Beta uses the ordinary bias-free Linear initializer. Initial gate outputs are input dependent because gate weights are nonzero; the zero-feature diagnostic is not a universal initial routing probability.

**Practical short-convolution departure:** q/k/v short convolution weights are exact causal Dirac filters, [0,0,0,1], with no random draw or bias. Native flash-linear-attention 0.2.2 ShortConvolution uses the inherited random Conv1d initializer, equivalent here to Uniform[-0.5,0.5], not Dirac. The exact pinned constructor is included and extracted with AST to prove seeded equality to nn.Conv1d without importing fused GPU kernels. The installed source matched its distribution RECORD as recorded by the preceding frozen-helper verification. This bundle repeats its hash, version and constructor check. This practical adapter should not be pooled with source-native-initialization trials.

**Projection/normalization departure:** source floating-point q/k/v/o projections and per-head RMS are replaced by the host's existing binary qkv/o and one block-wide RMS. The host RMS has epsilon 1e-6 and computes its statistic in FP32; source head RMS defaults to 1e-5. Host input RMS, residual path, experts, router, token table and objective handling are unchanged. Source optional output gating is not implemented. Host initialization RNG consumption differs from the source's complete layer, so seeded whole-layer initialization equivalence is not claimed.

**Dtype departure:** FP32 and FP64 delta calculations retain their input dtype. FP16/BF16 delta calculations promote q/k/v/beta to FP32, then cast the final result back. The source forcibly casts the triangular inverse to BF16, which is not reproduced for mixed-matrix FP32/FP64 testing. Gate/FIR/short-convolution operations use their ordinary tensor/autocast dtype. Only kernel-level low-precision promotion and literal-source BF16 finiteness are tested here; complete mixed-precision training and fused-kernel parity remain gates.

## Mutable state, diagnostics and unsupported behavior

There is **no source schedule and no built-in gate auxiliary loss**. The source does assign `last_fusion_weights` on every forward, retaining a graph-bearing activation even during evaluation. That is a deliberate API departure here, not an ignored state requirement.

The practical port never sets `last_fusion_weights`. `mix_with_diagnostics(h, causal=True)` instead returns `(output, detached_cpu_weights)` for mixer blocks and `(attention_output, None)` for attention blocks. It recomputes the same mixer, returns a detached cloned CPU tensor, and retains no diagnostic object inside the model. Normal `mix` does not request a CPU diagnostic copy. An external loss that expects source `last_fusion_weights` is **unsupported and blocked pending explicit trainer integration**; detached diagnostics cannot train such a loss. The host's separate router-balance auxiliary loss remains unchanged, and `take_extra_loss()` remains zero.

All new trainable state, including floor and temperature, appears in `state_dict`; there are no added persistent buffers, step counters or recurrent-state caches. Tests cover exact mixer and full-host state_dict round-trips, evaluation tensor/RNG purity, repeated outputs and source forward attribute mutation audit. This does **not** establish trainer resumability: optimizer state, binary update lifecycle, objective schedule/bandit, RNG and data-cursor persistence require a separate training contract and interrupted-run test.

Unsupported: cached incremental decoding, packed sequences, reset masks, padding-mask APIs, source optional output gate and alternate source switches. Passing unsupported cache/mask keywords is rejected by the mix signature. Empty sequence input raises ValueError. Ragged tests cover independent full sequences and valid right-padded prefixes; they do not establish packed segment isolation or streaming chunk APIs. The source's caching API is not evidence of a correct reusable cache in this port.

## Exact parameter and persistent-buffer accounting

All additions below are trainable floating-point parameters; baseline host tensors are unchanged:

- q/k/v four-tap short convolutions: 12d.
- Beta projection: Hd.
- FIR weights: (5+64)d = 69d.
- Gate weights: 4H(d+12hd) = 4Hd+48d.
- Gate bias: 4H; adaptive floor: 4H; log temperature: 4.
- Gate/floor/temperature subtotal: **4Hd+48d+8H+4**.
- Total additions per mixer: **5Hd+129d+8H+4**.

For tiny d=16, H=2, L=4, three mixers add 2,244 each, 6,732 total. The complete host baseline has 1,384 parameters and the adapter 8,116 (32,464 bytes in FP32). Persistent buffers remain 14,569 elements / 18,440 bytes. The JSON report gives each tiny parameter and buffer name, shape, dtype and count.

For the actual reduced host geometry d=256, H=4, L=12, V=8192 and ff=1024, nine mixers each add 38,180 parameters (16,420 gate/floor/temperature), 343,620 total. Baseline floating-point parameters are 108,061; adapted parameters are 451,681 / 1,806,724 bytes. Persistent buffers remain 20,548,435 elements / 20,852,376 bytes. The report lists the exact expert distribution and buffer dtypes. These are meta counts, not reduced-model training.

For d=8192, H=64, L=560, 420 mixers, and the frozen host expert distribution:

- Gate/floor/temperature: 2,490,884 parameters per mixer.
- Total additions: 3,678,724 per mixer; 1,545,064,080 overall.
- Baseline floating-point parameters: 150,450,908 (601,803,632 bytes).
- Adapted floating-point parameters: 1,695,514,988 (6,782,059,952 bytes).
- Body binary latent bytes: 886,910,746,624, unchanged.
- Binary token-table bytes: 1,638,924,288, unchanged.
- Persistent buffers: 888,689,802,393 elements / 889,110,212,296 bytes, unchanged.
- Those buffers comprise int8: 888,549,670,912 bytes; FP32: 140,127,616 elements / 560,510,464 bytes; int64: 3,865 elements / 30,920 bytes.

Full-size construction uses the meta device; no full-size tensors are allocated. These counts are **not a memory-fit claim**. They exclude transient activations, materialized binary weights, external optimizer state, execution workspaces and allocator overhead. The inherited `kv_bytes_per_token` census field is the host's attention estimate, not supported cache memory for this mixer.

## Tests and results

`test_afp712.py` passes 41 named checks covering:

- Pinned FLA native initializer and practical Dirac, exact seeded FIR/beta/gate initialization and source floor/temperature/bias defaults.
- Copied-weight outputs, input gradients, concatenated qkv gradients and every added parameter gradient in FP32/FP64, batch 2, lengths 1,7,31,32,33,63,64,65,129. Random nonzero gate, floor and temperature values prevent trivial initialization-only parity.
- Head-axis statistics, temperature clamp extremes, floor bounds/simplex and saturation gradients; explicit diagnostics agree with source weights.
- Independent sequential delta recurrence and chunk sizes 1,7,16,32,64; ragged boundaries; exact BF16/FP16 promotion contract.
- Causality, batch isolation, valid sequence prefixes, causal=False mixer behavior, unchanged attention fallback and full-model purity.
- Finite forward/backward for AR, SAT-fixed, SAT-variable, NAT and local denoising. These are zero-update correctness losses, not quality scores.
- Binary qkv/o single ownership, exactly one output RMS/projection call, exact tiny/full meta census, complete module checkpoint coverage and unsupported API rejection.

Maximum copied-weight mixer output errors are 1.1921e-7 (FP32) and 4.4409e-16 (FP64). Maximum mapped qkv gradient errors are 2.8611e-6 and 5.7732e-15 respectively. FP64 output and gradient comparisons use tight 2e-10 tolerances. See `preparation_results_dirac.json` for every check, tolerance-relevant error and census.

## Portable reproduction

Requires existing PyTorch, NumPy and einops. No installation or training data download is performed. `ASI_ARCH_REQUIREMENTS.txt` is historical provenance, not an instruction to install its complete environment.

From this folder, in a CPU environment whose resource policy permits bounded tensor tests:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python test_afp712.py
    sha256sum -c MANIFEST.sha256

Run serially, at low process priority and with a 90-second limit on shared service hosts. Local host policies take precedence; this is not permission to run training or benchmarks on a service host. The test pins one PyTorch intra-op and inter-op thread and writes its JSON next to the script. Tests use only paths relative to the bundle. The unmodified frozen host helper retains its original documented default data-path string; these unit tests do not access it. No private host paths appear in portable reports or required test inputs. Logs and bytecode caches are not part of the manifest.

## Remaining gates before any empirical ranking

1. Review this adapter and its explicit source departures; freeze the practical-Dirac full-width cohort and comparator implementations. No LR/LRK variant is prepared.
2. Fix the actual trainer's checkpoint/resume contract, objective protocol, precision, data/window hashes, seeds and tuning budget. Resolve any requested gate auxiliary-loss integration before using it.
3. Run the sanctioned serialized bounded optimizer smoke, then GPU finite-gradient and dtype/fused-kernel fidelity checks under the reviewed trainer. No optimizer smoke was run here.
4. Run matched-seed quality experiments only through the authorized reviewed GPU queue, including interrupted/resumed comparisons where appropriate. No quality, memory-fit, throughput or winner claim follows from this preparation.

## Licensing and attribution

The exact source/reference and derived mixer carry the upstream Apache-2.0 terms in `ASI_ARCH_LICENSE`. The pinned FLA implementation carries its MIT notice in `FLA_LICENSE`. The frozen user-project simulator and CPU shim retain their original bytes and existing provenance; their inclusion here does not invent a new license for those files. See `NOTICE.md` for changed-file attribution and the scope of this correctness bundle. No comparative training is included.
