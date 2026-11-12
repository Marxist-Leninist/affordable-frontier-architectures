# HeadWiseGateNet #641: isolated AGILLM full-width adapters

Status: correctness preparation only, not a trained-quality result. Both variants passed the same 31 bounded, CPU-only checks. No optimizer or binary-latent updates, benchmark training, GPU jobs, queue edits or deployment changes were performed.

## Which variant to use

- **hwg641_dirac.py / full-practical-dirac** is the prepared candidate for the next practical full-width comparison with the existing BAMG259, DLGM458 and Fusion879 adapters. Its 4-tap short q/k/v convolutions are exact causal Dirac filters (last tap 1, other taps 0), matching their common convention. This is an explicit departure from native #641 initialization.
- **hwg641.py / full-native-init** retains the verified native FLA short-convolution initialization. It is a separate initialization cohort, not a fair drop-in against Dirac-initialized comparators without a matched-initialization calibration.
- Both remain full-width. Neither is an LR or LRK implementation. Do not pool their trials with LR/LRK or with each other.

"Prepared" means the CPU correctness gate has passed. Before a quality claim, freeze the full-variant protocol, objective, seed schedule, data/window hashes, budget, precision, short-conv convention and tuning allowance for every arm. Recheck each deployed comparator, run the sanctioned bounded optimizer smoke, then GPU fidelity/finite-gradient checks under the actual trainer precision and existing authorized queue. Use separate matched seeds and a predeclared comparison; no winner is inferred here. No training job is launched by this bundle.

## Source and native FLA initialization evidence

Canonical source: [GAIR-NLP/ASI-Arch](https://github.com/GAIR-NLP/ASI-Arch), commit 3113c51b04a1d4a7520a3b0238c6a08d3194d58f, data/106.json, zero-based record 80, index 641, delta_net_hwg / HeadWiseGateNet.

The complete catalog digest was verified at extraction. The extracted program, its digest, frozen simulator, CPU helper and upstream license are included. provenance.json records the source and input hashes. The source catalog's "413.67M" metadata is not an AGILLM parameter census.

The pinned upstream requirements specify flash-linear-attention==0.2.2. The inspected installed 0.2.2 distribution had an unchanged convolution.py whose SHA256 matched its wheel RECORD:
d0b107a222cde18acc15ba3f84045da5c4c661b0d700c3c27ee0553e21eaec28.

An exact copy is included as native_fla_022_convolution.py, with its distribution metadata and MIT license. Its ShortConvolution inherits nn.Conv1d, calls the superclass initializer, and has no reset_parameters override or later weight initialization. Therefore its 4-tap, bias-free depthwise weights have the usual Kaiming-uniform initialization, equivalent to Uniform[-0.5,0.5], not Dirac. The test executes only the original constructor extracted with AST, with fast convolution disabled, and checks exact seeded equality to nn.Conv1d. This verifies native initialization without importing or running GPU/Triton kernels. It is stronger than relying on the CPU shim alone, but does not prove numerical parity with official fused kernels or reproduce the original full training RNG stream.

## Preserved mixer semantics

Let d be the full model width, H the number of heads and hd=d/H.

- Host fused qkv has shape [B,T,3,H,hd]; each q/k/v channel has its own 4-tap causal depthwise convolution followed by SiLU.
- Beta is sigmoid(Linear(d,H,bias=False)). Default beta and negative-eigenvalue behavior are preserved; beta stays in (0,1).
- The plain delta rule normalizes q and k with rsqrt(sum(x*x)+1e-6), uses 32-token chunks, and carries state without learned retention or decay. No retention factor is invented. The triangular solve is equivalent to the source's inverse construction at matched precision.
- Short and long FIR branches use 3 and 31 taps respectively. These are independent per head and channel, not shared head kernels. The last tap is 1 plus noise; every tap receives independent Gaussian noise with standard deviation 0.02, exactly as in #641.
- Gate input per head is [full token hidden state, that head's short output, long output, delta output]. The direct-value output is a branch, not an input feature.
- fusion_weight is [H,d+3hd,4], initially zero. fusion_bias is [H,4], initially [0,0,0,2] for every head.
- Softmax is over the four branches [short,long,delta,value], separately for each batch/token/head. Initial probabilities are approximately [0.096255,0.096255,0.096255,0.711235]. Branch outputs, not head-averaged statistics, feed the gate.
- Mixer layers stay causal even when causal=False; the unchanged attention layers supply bidirectional processing for NAT/local objectives.

## Explicit adaptations and exclusions

1. Host binary qkv/o own all projection latents and row scales. Original source FP q_proj/k_proj/v_proj/o_proj are not registered in the adapter. No duplicate large FP projections are present.
2. mix() returns a pre-output-normalization [B,T,d] tensor. Original source per-head RMSNorm (epsilon 1e-5 and shared head-dimension gain) is excluded. The host applies its existing one block-wide RMS (epsilon 1e-6, full-width gain) and binary o exactly once. Call-count tests verify one qkv, one host no and one o in a block forward. This is not equivalence to the source's complete FP DeltaNet layer.
3. The reference program removes only the torch.compile decorator and replaces its forced inverse cast to BF16 with a cast to v.dtype. For FP32/FP64 the adapter works in input precision and replaces the source's iterative inverse with a triangular solve. Copied-weight output/gradient claims refer to this explicitly dtype-matched mixer reference.
4. For FP16/BF16 delta inputs the adapter promotes all recurrence and q/k normalization work to FP32 and casts the final delta output back. This differs from the literal source's BF16 inverse rounding. The kernel promotion contract is tested; complete mixed-precision host training is not. The literal BF16 source is tested only for finite forward output.
5. Layout is mixer on three of four blocks, unchanged attention on every fourth and the final block. The 3/4 ratio is exact when L is divisible by four, including the tested L=4 and census L=560.
6. Default #641 expansion factors are one, use_gate=False, beta enabled, SiLU q/k, and l2 norm. Nondefault source constructor options are not exposed by this full-variant plugin.
7. There is no incremental decoder, recurrent-state cache, packed-sequence reset or attention_mask API. Ragged tests cover independently batched full sequences, right padding and valid prefixes, not concatenated packed documents or left-padding invariance. The inherited host kv_bytes_per_token census is an attention-storage estimate only; it does not establish cache support or serving cost.

## Evaluation and auxiliary loss policy

Neither adapter changes cfg or Model. There is no gate regularizer, annealing counter, forward step counter, evaluation accumulator or last-gate diagnostic mutation. The host's routing balance term remains unchanged (0.01 in the tested config), and take_extra_loss() remains zero. Repeated evaluation is deterministic and does not change any registered parameter or buffer. Training/evaluation mixer modes are equal for the same tensors.

## Exact accounting

Gate-only parameters per mixer: **4Hd + 12d + 4H**. This is not the full model count.

Extra parameters per mixer:
- q/k/v short convolutions: 12d.
- FIR branches: (3+31)d = 34d.
- beta: Hd.
- gate: 4Hd+12d+4H.
- Total: **5Hd + 58d + 4H**.

At d=8192, H=64, L=560 and 420 mixer layers:
- Gate alone: 2,195,712 parameters per mixer.
- All additions: 3,096,832 per mixer; 1,300,669,440 total.
- Host baseline FP parameters: 150,450,908.
- Adapted FP parameters: 1,451,120,348; 5,804,481,392 FP parameter bytes.
- Body binary latent bytes: 886,910,746,624, unchanged.
- Binary token-table bytes: 1,638,924,288, unchanged.
- All persistent buffers including table, R/C and step counters: 888,689,802,393 elements; 889,110,212,296 bytes, unchanged.

These full-scale figures are meta-device construction counts, not allocated models or measurements. The report includes per-dtype buffer counts and a complete per-tensor tiny-model census. Nonpersistent autograd temporaries, activations and external optimizer memory are not included in persistent parameter/buffer totals. The native and Dirac variants have identical counts.

## Tests and results

Each variant passes 31 checks:
- Native FLA initialization proof; exact seeded short-conv/FIR/beta initialization and gate warm start.
- Copied-weight mixer outputs, input gradients, projection gradients and all added-parameter gradients in FP32 and FP64 at lengths 1,7,31,32,33,65 with batch 2 and nonzero randomized gate weights.
- Independent serial delta recurrence and chunk sizes 1,7,16,32,64 on length 65.
- Explicit BF16/FP16 kernel promotion.
- Causal prefixes, valid ragged prefixes, batch isolation and unchanged causal/noncausal attention fallback.
- Five finite full-host forward/backward objectives: AR, SAT-fixed, SAT-variable, NAT and local denoising. Zero optimizer or latent updates.
- Binary projection ownership, output normalization call counts, exact storage, full-model causal prefixes, batch isolation and evaluation state purity.
- Literal-source BF16 finite-forward diagnostic.

Native: maximum FP32 mixer output error 5.9605e-8; qkv gradient error 1.4305e-6.
Practical Dirac: maximum FP32 mixer output error 8.9407e-8; qkv gradient error 2.8611e-6.
FP64 output error is at most 2.2205e-16 across both. Independent recurrence error is 1.3323e-15.
Full-model causal-prefix error is zero. All finite losses in JSON are correctness diagnostics, not trained quality, perplexity evidence or ranking scores.

## Reproduction

Requires existing PyTorch, NumPy and einops. No dependency installation or full training environment is needed. Original upstream requirements are supplied only as provenance, not an instruction to install that large environment.

From this directory, in a permitted CPU testing environment:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I test_hwg641.py hwg641
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I test_hwg641.py hwg641_dirac

Run serially with a 90-second timeout and low process priority on shared service hosts, subject to that host's resource policy. These are unit/tensor tests, not training commands. Tested Python 3.12.3 and PyTorch 2.10.0+cu128, CPU only, one thread. The script writes preparation_results.json or preparation_results_dirac.json next to itself.

Tests resolve all source/reference/simulator/helper files relative to their own directory. The complete original catalog is not required; its extraction digest is recorded and the extracted source digest is asserted. The native FLA implementation is used only for an isolated constructor check. All numerical source testing uses the clearly labeled CPU helper.

Publish only files listed in MANIFEST.sha256. Runtime logs and bytecode caches are excluded. No host-specific/private paths occur in the JSON reports, test dependency paths or this README. The unmodified frozen simulator retains its original documented default data path, which these unit tests do not access.
