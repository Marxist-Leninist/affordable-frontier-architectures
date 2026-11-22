# rh-unlocked (v0.2)

Certified prime-counting results that became unconditional when the Riemann Hypothesis for the
zeta function became a formally verified theorem (Lean library released 20 Nov 2026; joint Lean and
Mathlib certification 21 Nov 2026). The same Lean release also proves GRH for Dirichlet L-functions
(MIT's verification record); only the deterministic primality test below uses that part.

Each bound is a published theorem of the form "RH implies ...". See [SOURCES.md](SOURCES.md).

## What it computes

| command | result | theorem |
|---|---|---|
| `python rh_unlocked.py pi 1e100` | integers lo ≤ π(x) ≤ hi, and how many leading digits are certain | Schoenfeld 1976: \|π(x) − li(x)\| < √x ln x / 8π, x ≥ 2657 |
| `python rh_unlocked.py nthprime 1e50` | lo ≤ p_n ≤ hi | the same bound, inverted; both endpoints re-checked exactly |
| `python rh_unlocked.py theta 1e100` | θ(x) interval and the digit count of the primorial | Schoenfeld 1976, x ≥ 599 |
| `python rh_unlocked.py psi 1e100` | ψ(x) interval and the digit count of lcm(1..x) | Schoenfeld 1976, x ≥ 73.2 |
| `python rh_unlocked.py window 2^2048` | integer ranges at or above x and at or below x that must contain a prime | Carneiro–Milinovich–Soundararajan 2019, Thm 5: a prime in [x, x + (22/25)√x ln x] for x ≥ 4 |
| `python rh_unlocked.py nextprime 2^2048` | B such that the next prime after y is at most B | the same theorem |
| `python rh_unlocked.py count-bits 2048` | bounds on the number of 2048-bit primes | Schoenfeld 1976 |
| `python rh_unlocked.py robin 1e12` | σ(n) < e^γ n ln ln n, next to Robin's unconditional bound | Robin 1984, n > 5040 |
| `python rh_unlocked.py demo` | before/after tables | |

In Python, `is_prime_deterministic(n)` runs Miller's test with every base below 2 (ln n)² (Bach 1990,
correct under Dirichlet GRH), and `next_prime_certified(y)` finds the first prime after y inside the RH
window. Example: the first prime after 10^100 is 10^100 + 267 (106,037 bases, about 40 s).

Inputs can be exact: `1e100`, `10^100`, `2**256`, or plain integers.

## Why the endpoints are safe

All certified endpoints use exact integer arithmetic with directed rounding, not floating point.
li(x) = γ + ln ln x + Σ (ln x)^k/(k·k!) and ln x = b ln 2 + 2 atanh((x − 2^b)/(x + 2^b)) are each
computed twice at ⌊digits/2⌋ + 60 decimal places: once rounding every step down and dropping positive
tails (a lower bound), once rounding every step up and adding an explicit tail bound (an upper bound).
π > 3.141592653589793238462643383279502884197 and 700 digits of γ are embedded (γ is checked against
mpmath in the tests). mpmath is used only for display and for starting points of root searches; every
inverted result (n-th prime, windows) is re-checked at its returned integer endpoint.

The browser calculator at https://zqx.lat/riemann-hypothesis/#certified-prime-bounds uses the same
algorithm in JavaScript BigInt and returns identical integers on 1,676 test inputs up to 10^1000.

## Tests (`python -m pytest -q test_rh_unlocked.py`, about 30 s, 31 tests)

- Every real x in [2657, 2·10⁷] for π, [599, 2·10⁷] for θ, [73.2, 2·10⁷] for ψ (both ends of every step).
- Every prime gap below 10⁸ against Theorem 5; window and next-prime bounds at 2,000+ points and all record gaps below 10⁸.
- Robin's inequality for every n in (5040, 10⁶]; 5040 itself fails it.
- π(10^k), k ≤ 29, and p_(10^k), k ≤ 24 (OEIS); 3,000 random π(x) and 300 random p_n against a sieve.
- The exact li and ln bounds bracket mpmath at 16 inputs up to 10^400.
- Deterministic primality against the sieve and on strong pseudoprimes to every prime base up to 41.

Schoenfeld's threshold is tight: just above 2657 the count uses 0.98 of the allowed error. Beyond 10⁶ it never uses more than 0.26.

## Files in this folder

| path | what it is |
|---|---|
| `rh_unlocked.py` | the toolkit (one file; needs Python 3.10+, mpmath, numpy) |
| `test_rh_unlocked.py` | 31 pytest tests; `data/` holds the OEIS tables they use |
| `certified-prime-bounds.json` | every certified endpoint shown on the web page, as exact integers |
| `RESULTS.md` | before/after tables from `python rh_unlocked.py demo` |
| `VERIFICATION.md`, `research/verified_claims.json` | how each theorem and statement was checked, claim by claim |
| `browser/prime-bounds-v3.js` | the calculator script deployed at zqx.lat (22 Nov 2026) |
| `browser/tests/` | `python browser/tests/make_fixture.py` (writes about 6 MB; about 7 min on a laptop) then `node browser/tests/test_v3.js` |
| `browser/build_site.py` | how the page was built from a pinned copy of the live page (for provenance) |

## Limits

- The window and count results need only RH for ζ. Only `is_prime_deterministic` relies on Dirichlet GRH, which the Lean release proves but which the joint Lean and Mathlib statement does not cover. Results that need GRH for Dedekind or Hecke L-functions are not here.
- The Clay Institute still lists RH as unsolved; its prize rules need publication, two years and acceptance.
- "Before RH" comparisons in the demo use closed-form published bounds; tabulated bounds can be sharper at moderate x.
