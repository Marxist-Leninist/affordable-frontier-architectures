# What can actually be reversed in a Transformer?

## Three different questions
- An inverse function recovers an input from an output: f⁻¹(f(x)) = x, on a specified domain.
- Backpropagation computes a vector–Jacobian product Jᵀv. It is not generally multiplication by J⁻¹. Reverse-mode means reverse graph traversal, not inverse execution.
- A symmetry changes the parameterization or representation without changing the modeled function. It can help reasoning or optimization, but creates no new information.

A trained language model normally maps many different prompts or hidden states to the same sampled token. Sampling, argmax and discarded context cannot be uniquely reversed. Recovering a plausible prompt is an inference problem, not an exact inverse.

## Operation atlas

### Token lookup and positions
An embedding lookup may be injective over its finite vocabulary if all token vectors are distinct. It is not a general invertible map from continuous vectors to tokens. Adding a known positional vector can be reversed by subtracting it. A RoPE rotation is orthogonal when computed exactly; its inverse is the transpose/negative angle. Unknown positions or finite precision add ambiguities.

### Linear projections and multiplication
For y = Wx + b, a square nonsingular W gives x = W⁻¹(y − b). A tall full-column-rank W has a left inverse on its image; a wide or rank-deficient W generally loses information. A pseudoinverse chooses one solution and does not recover missing components. Conditioning controls numerical reliability, and solving a system is preferable to explicitly constructing W⁻¹. A gradient Wᵀv equals an inverse transform only in special cases such as a square orthogonal W.

### Layer normalization and RMS normalization
LayerNorm(x) = γ ⊙ [(x − mean(x)) / sqrt(var(x)+ε)] + β always removes the common shift, so it is not globally injective. At ε=0 it also removes positive scale. At ε>0 scale invariance is only approximate, not exact. Zero entries of γ lose additional coordinates.

RMSNorm deserves a qualification: with ε=0 it removes positive scale. With ε>0 and every γᵢ nonzero, let z=y/γ and r=mean(z²)<1. Then x=sqrt(ε/(1−r))z is the real-arithmetic inverse on the image. This can be numerically ill-conditioned near r=1. Calling every normalization operation noninvertible would therefore be inaccurate.

Neither normalization used *inside* a reversible coupling function has to be invertible. The untouched companion stream preserves the information needed to recompute it.

### Softmax, attention and masks
softmax(z+c·1)=softmax(z), so unrestricted logits are not identifiable. For strictly positive probabilities, log(p) recovers a representative and log(pᵢ/pⱼ) recovers logit differences. Masked positions do not retain their original logits. Temperature must be known.

For known A and y=AV, a square full-rank attention matrix A permits recovery of V. A causal softmax matrix with finite diagonal logits is triangular with a positive diagonal and therefore nonsingular in exact arithmetic. That conditional fact does not invert the whole attention module: Q, K, V depend on unknown inputs, projections can lose information, and numerical underflow or poor conditioning can destroy useful recovery. Replacing softmax with its transpose is not an inverse.

QKᵀ has a factorization symmetry: Q→QR and K→KR⁻ᵀ for an invertible R preserves the product. Applying such a gauge transformation to a complete architecture also has to respect RoPE, normalization and parameter-sharing constraints. Likewise, attention-head permutations preserve function if the output projection is permuted consistently. These are representations of the same model, not a learning gain by themselves.

### Nonlinearities and gating
ReLU discards negative magnitudes. GELU is not globally one-to-one; it has a negative-side minimum. A gated MLP such as SwiGLU can lose information through both gating and output projection. Particular restricted-domain inverses are possible, but do not establish an inverse for ordinary unrestricted MLPs. Hidden-unit permutations with matching adjacent weight permutations preserve a feed-forward network's function. General rescaling is activation-dependent: ReLU's positive homogeneity does not carry over exactly to GELU.

### Residual additions
A residual map y=x+f(x) is not automatically invertible: f(x)=−x makes it constant. If f is globally Lipschitz with constant L<1 on a suitable complete domain, x=y−f(x) defines a convergent fixed-point inverse. Ordinary attention/MLP weights do not enforce that condition. Subtracting f(y) once is usually not an exact inverse.

### Additive coupling: a useful exact construction
Split or maintain two states. With arbitrary deterministic, shape-compatible functions F and G:

y₁=x₁+F(x₂)
y₂=x₂+G(y₁)

The inverse is explicit:

x₂=y₂−G(y₁)
x₁=y₁−F(x₂)

No inverse of F or G is required. The coupling Jacobian has determinant1 in exact arithmetic, even if its constituent functions are many-to-one. This does not force the full embedding-to-token model to be a bijection. Finite-precision error can accumulate with depth and ill-conditioned dynamics, so test both reconstruction and gradients.

For an incoming loss gradient (u₁,u₂), backpropagation is:

v₁=u₁+J_G(y₁)ᵀu₂
v₂=u₂+J_F(x₂)ᵀv₁

Parameter derivatives are (∂G/∂θ_G)ᵀu₂ and (∂F/∂θ_F)ᵀv₁. The reconstructed x₂ permits recomputing local derivatives without retaining every layer's graph. This saves some activation storage but adds work. It is still backpropagation.

### Dropout, quantization, clipping and sampling
A known dropout mask can reproduce the random operation, but cannot recover values erased by that mask. RNG replay is needed for correct reversible recomputation when stochastic functions occur inside F or G. Quantization, saturation, truncation, gradient clipping and argmax are generally many-to-one. Sampling also requires the random choice and still does not recover the originating probability distribution from one token. Beam search is a search procedure, not an inverse.

### Loss and optimizer updates
Cross entropy maps predictions and targets to a scalar; this scalar does not identify all predictions or parameters. Gradient descent w′=w−η∇L(w) can have a unique inverse under strong conditions, such as a contraction of η∇L with known data and domain. Ordinary stochastic training plus Adam moments, changing batches, clipping and finite precision does not supply a practical universal inverse. Saving optimizer/RNG/batch history enables replay or rollback, not recovery from final weights alone.

Reversing the sign of a gradient is gradient ascent on the same loss. Reversing a denoising update can amplify noise rather than learn an inverse. Forward-mode automatic differentiation computes Jv rather than Jᵀv; neither is generally an inverse. A single directional derivative is not a full parameter gradient.

## Meaningful alternatives and what they would establish
1. Exact reversible coupling with reconstructed-activation backpropagation: potentially less stored activation state, normally extra compute, unchanged mathematical gradient for the same architecture.
2. Contractive/invertible residual networks: constrain Jacobians and test inversion cost, stability, expressivity and held-out quality. Not covered by this pilot's implementation.
3. Orthogonal/unitary linear maps: exact transpose inverse and controlled norm in ideal arithmetic, but constrain the model and do not invert normalization or sampling. Not covered by this pilot.
4. Target propagation or learned feedback maps: train approximate backward mappings and assess local credit assignment against full backprop. Approximate feedback is not an exact inverse. Addressed separately from this reversible-activation test.
5. An auxiliary reconstruction objective: may regularize representations or consume useful capacity. It needs a matched auxiliary-loss control; not silently added here.
6. Parameter symmetries: useful for debugging, aligning checkpoints or studying geometry, but a reparameterization does not by itself improve predictive information.

## Evidence standard
Our primary proxy is held-out next-byte cross entropy, with validation/test data excluded from training. That measures a small language-prediction task, not general intelligence. Inverse reconstruction, gradient parity, retained activation bytes, runtime and learning are separate claims requiring separate results. A small positive pilot would justify a controlled next experiment, not a 600B deployment or an assertion of human-level intelligence.

## Prior art and authoritative references
- Gomez et al., The Reversible Residual Network: Backpropagation Without Storing Activations (2017): https://arxiv.org/abs/1707.04585
- Kitaev et al., Reformer: The Efficient Transformer (2020): https://arxiv.org/abs/2001.04451
- Dinh et al., NICE: Non-linear Independent Components Estimation (2014): https://arxiv.org/abs/1410.8516
- Behrmann et al., Invertible Residual Networks (2019): https://arxiv.org/abs/1811.00995
- Vaswani et al., Attention Is All You Need (2017): https://arxiv.org/abs/1706.03762
- Salesforce WikiText dataset: https://huggingface.co/datasets/Salesforce/wikitext

The references situate established constructions; they are not evidence that this particular pilot has completed or improved learning.
