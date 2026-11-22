# How the claims were checked

The theorems this toolkit relies on, and the statements on the web page, went through a research and
adversarial-verification pass on 22 Nov 2026: four research agents, one per question, then three
independent checkers told to refute each claim against a primary source (paper, arXiv abstract or
journal page, official statement). Every claim, verdict, correction and source URL is in
[research/verified_claims.json](research/verified_claims.json).

| question | claims | verdicts |
|---|---|---|
| A. Explicit prime-counting bounds RH makes unconditional, and the best unconditional bounds | 25 | 24 confirmed, 1 corrected |
| B. Analytic consequences of RH for ζ | 55 | 34 confirmed, 6 corrected, 15 unverifiable from sources reachable in the run |
| C. What still needs more than RH, and practical impact | 27 | 21 confirmed, 6 corrected |
| D. The Lean artifact itself (size, checkers, build cost) | 33 | factual record, cross-checked by C |

Only confirmed statements are used in the code or on the page.

## Corrections that mattered

- **Direction of the prime window.** Carneiro, Milinovich and Soundararajan, Theorem 5, reads: assume RH; for x ≥ 4 there is always a prime in [x, x + (22/25)√x log x]. It is stated forwards. The backward form (x − c√x log x, x] is Dudek's 2015 result with c = 4/π. The first version of this toolkit had the window backwards; the paper text was read directly and the code and tests were fixed.
- **Scope of the formal proof.** The joint Lean and Mathlib statement (21 Nov 2026) certifies RH for ζ. The same release also proves GRH for Dirichlet L-functions (theorems `LFunction_ne_zero_of_half_lt_re`, `LFunction_eq_zero_re`, `LFunction_eq_zero_re_of_isPrimitive`), accepted by several independent checkers on MIT's record. Dedekind, Hecke and automorphic L-functions are not covered.
- **Earlier ranges.** Schoenfeld's π bound was already proved unconditionally for 2657 ≤ x ≤ 1.101·10²⁶ (Johnston 2022), and Theorem 5 for x ≤ 4·10¹⁸ from computed prime gaps. RH adds the ranges above those.
- Smaller fixes: the log|ζ(1/2+it)| ≪ log t / log log t bound is Littlewood's (1924); the positive-proportion small-gap result 0.6039 assumes almost all zeros are simple as well as RH; Cipolla's square-root method is not covered by the non-residue bound, Tonelli–Shanks is.

## Numerical checks run here

- li(10¹²) − √(10¹²) ln(10¹²)/(8π) = 37,606,850,877.41, so π(10¹²) ≥ 37,606,850,878; the exact count 37,607,912,018 is inside the certified interval.
- Exact li bounds bracket a high-precision mpmath value at every tested input; the browser and Python implementations return identical integers at 1,676 inputs up to 10¹⁰⁰⁰.
- Schoenfeld's threshold 2657 is tight: just above it π(x) uses 0.98 of the allowed error; above 10⁶ never more than 0.26.
