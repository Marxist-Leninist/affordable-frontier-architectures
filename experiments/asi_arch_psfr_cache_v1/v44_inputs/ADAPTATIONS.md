# ASI-Arch tiny CPU proxy: explicit adaptations and source defects

Original exports in original/ remain byte-for-byte unchanged, Apache-2.0, from
GAIR-NLP/ASI-Arch commit 3113c51b04a1d4a7520a3b0238c6a08d3194d58f,
export data/106.json, blob e403531e663cf24c800002720d293cccf77f94df.
PSFR record879/FusionGatedFIRNet and CSM record703/ContentSharpRouter.
The paper's independently reported large-model scores are NOT reproduced here.

cpu_adapter.py supplies elementary PyTorch RMSNorm, gated RMSNorm, causal
depthwise ShortConvolution and l2norm (epsilon1e-6). These are dependency
references, not official fused FLA kernels. einops0.8.1 supplies reshaping.
The @torch.compile decorator alone is removed dynamically. Source kernels,
activations, FIR lengths, gate definitions and chunk32 remain unchanged.
Unpadded fixed-length independent batch rows only. Masked/packed inputs fail
explicitly: original FIR/delta paths do not respect packed sequence boundaries.

Both exported sources ignore incoming recurrent states; both FIR branches reset
at every call. Cache/full-prefix equivalence fails. No cache/decode benchmark
or production cost extrapolation is valid for these source versions.
PSFR retention is NOT tokenwise decay: each chunk computes ordinary delta
updates from its initial state then the next chunk's state equals updated state
minus (1-lambda)*initial state. Returned state is detached. Preserve chunk32.

CSM out_proj returns 4H logits for every head, normalizes all 4H and consumes
only the first four probabilities. With H2 this discards four probabilities.
This is preserved, not repaired to match the advertised four-way gate.
CSM returns positive alpha*entropy in the attention-output slot. Minimizing it
would encourage lower entropy, contrary to its diversity docstring. The LM
harness ignores that slot as standard attention outputs, and never flips sign.

All-three-block architecture comparison: width64, ASI heads2, three pre-norm
residual blocks and SwiGLU, untied byte256 embeddings/output, no dropout.
Modern Transformer uses four attention heads, RoPE and CPU SDPA. It has
190848 parameters, PSFR190788 (-0.0314%), CSM190878 (+0.0157%). Per-layer FFN
widths are respectively [189,188,188], [108,107,107], [162,162,161]. Nearest
feasible active FFN units, no dummy parameters. Equal tokens, not equal FLOPs.
This baseline is newly trained, not v43's checkpoint or exact FFN configuration.
No hyperparameter or checkpoint selection using heldout outcomes.

FP64 tiny checks test causal-prefix preservation, unpadded batch independence,
finite backward gradients, broken exported cache, and an independent token-loop
reference with chunk-boundary PSFR retention. These validate our CPU adapters'
internal consistency, not binary equivalence with unavailable fused kernels.
