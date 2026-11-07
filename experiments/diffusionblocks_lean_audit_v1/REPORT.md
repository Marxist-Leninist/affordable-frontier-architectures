# DiffusionBlocks: Lean and equivalence audit

Date: 7 November 2026. Scope: the supplied Figure 3 screenshot, corresponding arXiv:2506.14202v4, and a read-only inspection of the authors' public ViT code. This is not a replication of the paper's language-model experiments.

## Verdict

- The intended corrected diffusion update and a residual-network update share an exact algebraic form.
- They are not generally the same training method, objective, gradient, initialization, or learned function.
- A useful closed-form solution exists for affine dynamics (and further special distributions), but these checks do not produce a general analytic solution for nonlinear Transformers or their training.
- The paper's displayed Euler equations have a sign inconsistency. The public implementation uses the correct sign.
- The screenshot's inference loop has indexing errors if interpreted literally. Those indexing errors are not present in the inspected implementation.

## 1. Correct Euler equation, and the paper error

The paper gives dz/dsigma = -sigma * score and score = (D(z,sigma)-z)/sigma^2. For a step from s to t with s>t>=0:

    z_next = z + (t-s)*(z-D)/s
           = z + alpha*(D-z)
           = (1-alpha)*z + alpha*D,
    alpha = (s-t)/s.

For s>0 and 0<=t<=s, alpha is in [0,1]. The displayed paper defines Delta_sigma=s-t>0, but its equations (3)-(5) use z + (Delta_sigma/s)*(z-D), the opposite direction.

Exact example: z=1, D=0, s=1, t=1/2. Correct Euler gives 1/2. The printed formula gives 3/2. This is an inconsistency between the stated ODE, step direction, and printed discretization, not a proof that the trained method cannot work.

The official code uses d=(z-denoised)/sigma, dt=next_sigma-sigma, and z_next=z+dt*d, which agrees with the corrected formula.

Figure 3 starts i=0 while reading z_(i-1) and sigma_(i-1), stops at T-1, then returns z_T. A consistent version starts with sigma_0,...,sigma_T and z_0; for i=0,...,T-1 it selects the block at sigma_i, computes D(x,z_i,sigma_i), and assigns z_(i+1). Shared closed interval endpoints additionally need a deterministic tie rule; the code uses bucketize and clamping.

## 2. What is mathematically equivalent?

For a residual map z_next=z+F(z), fix a nonzero alpha and define:

    D(z) = z + F(z)/alpha.

Then z+alpha*(D(z)-z)=z+F(z) exactly. Conversely, any denoiser D defines residual F(z)=alpha*(D(z)-z). This holds even for nonlinear F as a one-step reparameterization.

Conditions matter: both expressions use the same input state, conditioning, dimension, and function. Merely training a different noise-conditioned network does not establish that it equals this constructed D. Initializing the residual network at x and the diffusion process at random noise also produces different trajectories unless explicitly matched.

This reparameterization neither removes evaluation of F nor makes its training analytic. It identifies a shared update structure.

## 3. Training objectives are different

Standard end-to-end training minimizes a loss of the composed network. Block denoising minimizes local losses on noisy target samples. A local block loss is independent of other blocks' parameters only if those parameters do not occur in its computation and no parameters are shared. The Lean independence result explicitly assumes this local functional form; it does not prove a particular neural implementation has no shared parameters.

Exact toy counterexample: target=1, starting state=0, two constant denoisers D1=a and D2=b, both step sizes alpha=1/2. The final state is a/4+b/2.

    Local loss = (a-1)^2 + (b-1)^2.
    End-to-end loss = (a/4+b/2-1)^2.

At (a,b)=(1,1), local loss is 0 but final loss is 1/16. At (a,b)=(2,1), final loss is 0 but local loss is 1. Thus the objectives and their minimizers need not coincide. This particular example has a nonzero terminal noise level (schedule 1,1/2,1/4). It disproves unconditional equivalence; it does not disprove asymptotic consistency of ideal denoising with appropriate schedules, exact scores, and terminal limits.

If D always returns a fixed exact target y, the corrected step satisfies:

    z_next-y = (t/s)*(z-y).

Successive such steps telescope, and a terminal step to zero reaches y exactly. This strong perfect-denoiser assumption is not a property established for a learned language model.

## 4. Closed forms that really exist

If D_k(z)=A_k*z+b_k, set:

    M_k = (1-alpha_k)*I + alpha_k*A_k,
    c_k = alpha_k*b_k.

Then z_(k+1)=M_k*z_k+c_k, and:

    z_T = (M_(T-1)...M_0)*z_0
          + sum over k=0,...,T-1 of
            (M_(T-1)...M_(k+1))*c_k.

An empty product is I. Matrix order must be retained; matrices generally do not commute. This formula is a finite product-sum, not a guarantee of lower computation cost.

For a fixed scalar m and c, Lean verifies for every natural T:

    z_T = m^T*z_0 + (1+m+...+m^(T-1))*c.

When m!=1, it further verifies the nonrecursive geometric formula:

    z_T = m^T*z_0 + ((m^T-1)/(m-1))*c.

At m=1 the usual special case is z_T=z_0+T*c. The generic Lean proofs use standard field/commutative-ring laws. No real-number model, calculus library, matrix theorem, or Gaussian process is imported. The matrix product-sum formula was independently numerically checked for 100 time-varying 4-dimensional, 8-step examples, with maximum absolute error 1.6653345369377348e-16.

An additional analytic special case, not Lean formalized here: if the conditional target distribution is Gaussian with mean mu and covariance C, the exact conditional-mean denoiser is

    D(z,sigma) = mu + C*(C+sigma^2*I)^(-1)*(z-mu).

Under positive-definiteness/invertibility conditions, the exact probability-flow ODE solution is

    z(sigma) = mu + (C+sigma^2*I)^(1/2)
                    *(C+sigma_0^2*I)^(-1/2)*(z(sigma_0)-mu).

This is the exact continuous ODE solution, not generally equal to a finite Euler discretization. Real language distributions and Transformer denoisers do not satisfy a general affine-Gaussian assumption. No general closed-form optimizer or exact nonlinear-flow solution is established by this audit.

## 5. Additional public-code issue

The inspected ViT implementation forms per-example loss with shape [N] and weights with shape [N,1], then evaluates (loss*weights).mean(). Broadcasting creates [N,N], which yields mean(loss)*mean(weights), not mean(loss_i*weight_i). For loss=[1,3] and weights=[1,3], those are 4 and 5 respectively.

The shape behavior was independently reproduced with NumPy and the rational arithmetic is Lean checked. The paper code was not executed or changed. This is a concern about the inspected public ViT source, not an established defect in the unpublished language-model implementation or a measured impact on its results.

## Verification and limitations

See lean-check.log and verification.json for final compiler status, source hash, and axiom audit. The supplied Lean source contains no sorry, custom axioms, or native_decide proof shortcuts. Its standard dependencies may include propext, Classical.choice, and Quot.sound. Counterexamples use exact rational arithmetic, not approximate floating-point comparisons.

The numerical script is independent author-written test code. Its results are identity checks, not training-quality, speed, memory, convergence, or language benchmarks. General stochastic score-matching consistency, Tweedie's theorem, block partition probabilities, GPU implementation correctness, and the paper's empirical claims are outside this formalization.

## Sources and reproduction

- Paper: https://arxiv.org/pdf/2506.14202v4 (equations 1, 3-6; Figure 3).
- Public source at pinned commit: https://github.com/SakanaAI/DiffusionBlocks/blob/ade0b08cf7a761b487948921fe2a14eb29039bd1/model.py
- Official Lean release: https://github.com/leanprover/lean4/releases/tag/v4.35.0
- Run `lean DiffusionEquivalence.lean` with Lean 4.35.0.
- Run `python numerical_checks.py` with NumPy installed.
