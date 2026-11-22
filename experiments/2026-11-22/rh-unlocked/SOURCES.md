# Sources

## Theorems used by the code (each was "RH implies ...", now unconditional)

| id | statement | source |
|---|---|---|
| pi | \|π(x) − li(x)\| < √x·log x / (8π) for x ≥ 2657 | L. Schoenfeld, *Sharper bounds for the Chebyshev functions θ(x) and ψ(x). II*, Math. Comp. 30 (1976) 337–360, Corollary 1 |
| theta | \|θ(x) − x\| < √x·log²x / (8π) for x ≥ 599 | Schoenfeld 1976, Corollary 1 |
| psi | \|ψ(x) − x\| < √x·log²x / (8π) for x ≥ 73.2 | Schoenfeld 1976, Corollary 1 |
| gap | for every x ≥ 4 there is a prime in [x, x + (22/25)·√x·log x] (stated forwards, Theorem 5) | E. Carneiro, M. B. Milinovich, K. Soundararajan, *Fourier optimization and prime gaps*, Comment. Math. Helv. 94 (2019) 533–568, arXiv:1708.04122. Already known for 4 ≤ x ≤ 4·10¹⁸ from computed prime gaps |
| robin | σ(n) < e^γ·n·log log n for every n > 5040 | G. Robin, J. Math. Pures Appl. 63 (1984) 187–213 |
| bach | under GRH for Dirichlet L-functions, every odd composite n fails Miller's test for some base a < 2 (ln n)² | E. Bach, *Explicit bounds for primality testing and related problems*, Math. Comp. 55 (1990) 355–380 |

## Bounds used only for the "before RH" comparison

| bound | source |
|---|---|
| \|π(x) − Li(x)\| ≤ 9.2211 x √(ln x) e^(−0.8476 √(ln x)), x ≥ 2 | Fiori, Kadiri, Swidinsky, arXiv:2206.12557 |
| \|π(x) − li(x)\| ≤ 0.028 x (ln x)^0.801 exp(−0.1853 (ln x)^(3/5) (ln ln x)^(−1/5)), x ≥ 23 | Johnston, Yang, arXiv:2204.01980 |
| Schoenfeld's π bound holds unconditionally for 2657 ≤ x ≤ 1.101·10²⁶ | Johnston, Ramanujan J. (2022), doi:10.1007/s11139-022-00616-x |
| a prime in (x, x(1 + 1/(5000 ln² x))] for x ≥ 468991632; (x, x(1 + 1/ln³ x)] for x ≥ 89693 | Dusart, Ramanujan J. 45 (2018) 227–251 |
| a prime in (x(1 − 1/2.5·10¹¹), x] for x ≥ e⁶⁰⁰ | Cully-Hugill, Lee, arXiv:2103.05986 |
| p_(n+1) − p_n < ln² p_n for 11 ≤ p_n ≤ 4·10¹⁸ (computation) | Oliveira e Silva, Herzog, Pardi, Math. Comp. 83 (2014) |

All constants above were checked against primary sources by an independent research-and-verify pass on 22 Nov 2026.

li is the principal-value logarithmic integral ∫₀ˣ dt / log t (`mpmath.li`, offset=False).

## The RH proof

- Lean library released by Levent Alpöge on 20 Nov 2026 (~17 million lines, depends on Mathlib only); joint Lean + Mathlib community certification on 21 Nov 2026: https://leanprover-community.github.io/blog/posts/riemann-hypothesis/
- What was certified is Mathlib's `RiemannHypothesis`, the statement for the Riemann zeta function. GRH is not covered.
- The Clay Institute still lists RH as unsolved (its prize needs publication, two years and acceptance).

## Test data (downloaded 22 Nov 2026, parsed as plain integers only)

| file | contents | source | sha256 |
|---|---|---|---|
| data/A006880/b006880.txt | π(10^k), k = 0..29 | https://oeis.org/A006880/b006880.txt | 8111c06f2093b8cf138480a458945bf2eb1a01ee783d64041a7b11a102cd08a3 |
| data/A006988/b006988.txt | p_(10^k), k = 0..24 | https://oeis.org/A006988/b006988.txt | 4266921c6f1294c929a7a1e0fe8d4c8796437073f39e93e919b039b302084d11 |
