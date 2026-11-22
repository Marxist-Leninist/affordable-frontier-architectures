#!/usr/bin/env python3
"""rh_unlocked: certified prime-counting results that became unconditional when
the Riemann Hypothesis for the zeta function became a formally verified theorem
(Lean library released 20 Nov 2026; Lean and Mathlib joint certification 21 Nov 2026).

Every bound used here is a published theorem of the form "RH implies ...".
With RH proven, each one now holds with no hypothesis. Nothing here assumes GRH.

Numerics: mpmath, with the working precision chosen from the size of x so that
rounding error sits many orders of magnitude below the error term. Every real
endpoint is widened by an explicit slack before it is rounded to an integer, and
every inverted bound (n-th prime, next prime) is re-checked at its returned
endpoint, so correctness does not rest on the root-finder.

Usage:  python rh_unlocked.py pi 1e100
        python rh_unlocked.py nthprime 1e50
        python rh_unlocked.py nextprime 2^2048
        python rh_unlocked.py count-bits 2048
        python rh_unlocked.py demo
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from functools import lru_cache

import mpmath as mp
import numpy as np

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Theorems (each was "RH => ...").  Exact constants and ranges; see SOURCES.md.
# ---------------------------------------------------------------------------
THEOREMS = {
    "pi": {
        "statement": "|pi(x) - li(x)| < sqrt(x) log(x) / (8 pi) for x >= 2657",
        "x0": 2657,
        "source": "L. Schoenfeld, Sharper bounds for the Chebyshev functions theta(x) and psi(x). II, "
                  "Math. Comp. 30 (1976) 337-360, Corollary 1",
    },
    "theta": {
        "statement": "|theta(x) - x| < sqrt(x) log(x)^2 / (8 pi) for x >= 599",
        "x0": 599,
        "source": "Schoenfeld 1976, Corollary 1",
    },
    "psi": {
        "statement": "|psi(x) - x| < sqrt(x) log(x)^2 / (8 pi) for x >= 73.2",
        "x0": mp.mpf("73.2"),
        "source": "Schoenfeld 1976, Corollary 1",
    },
    "gap": {
        "statement": "for every x >= 4 there is a prime in [x, x + (22/25) sqrt(x) log(x)]",
        "x0": 4,
        "c": mp.mpf(22) / 25,
        "source": "E. Carneiro, M. B. Milinovich, K. Soundararajan, Fourier optimization and prime gaps, "
                  "Comment. Math. Helv. 94 (2019) 533-568, Theorem 5 (arXiv:1708.04122). Already known "
                  "unconditionally for 4 <= x <= 4e18 from the computed prime gaps (Oliveira e Silva, Herzog, Pardi).",
    },
    "robin": {
        "statement": "sigma(n) < e^gamma n log log n for every n > 5040",
        "n0": 5040,
        "source": "G. Robin, Grandes valeurs de la fonction somme des diviseurs et hypothese de Riemann, "
                  "J. Math. Pures Appl. 63 (1984) 187-213",
    },
}

# Best explicit bounds that did NOT assume RH, used only for the before/after comparison.
# Each entry: (name, bound on (X, logX), valid-from, valid-to or None, source).  See SOURCES.md.
LI2 = mp.mpf("1.045163780117492784844588889194613136522615578151")  # li(2): Li(x) = li(x) - li(2)

UNCONDITIONAL_PI = [  # bounds on |pi(x) - li(x)|
    ("Johnston 2022, Cor. 3.3 (Schoenfeld's bound, unconditional on this range)",
     lambda X, L: mp.sqrt(X) * L / (8 * mp.pi), 2657, mp.mpf("1.101e26"),
     "D. R. Johnston, Ramanujan J. (2022), doi:10.1007/s11139-022-00616-x"),
    ("Fiori-Kadiri-Swidinsky 2023, closed form",
     lambda X, L: mp.mpf("9.2211") * X * mp.sqrt(L) * mp.exp(-mp.mpf("0.8476") * mp.sqrt(L)) + LI2, 2, None,
     "A. Fiori, H. Kadiri, J. Swidinsky, arXiv:2206.12557 (|pi - Li| bound, x >= 2)"),
    ("Johnston-Yang 2023, Korobov-Vinogradov form",
     lambda X, L: mp.mpf("0.028") * X * L ** mp.mpf("0.801")
     * mp.exp(-mp.mpf("0.1853") * L ** (mp.mpf(3) / 5) * mp.log(L) ** (-mp.mpf(1) / 5)), 23, None,
     "D. R. Johnston, A. Yang, arXiv:2204.01980, Thm 1.4 (x >= 23)"),
]

UNCONDITIONAL_WINDOW = [  # a prime is guaranteed within this distance of x (one side)
    ("Dusart 2018, Cor. 5.5: a prime in (x, x(1 + 1/(5000 ln^2 x))]",
     lambda X, L: X / (5000 * L ** 2), 468991632, None, "P. Dusart, Ramanujan J. 45 (2018) 227-251"),
    ("Dusart 2018, Prop. 5.4: a prime in (x, x(1 + 1/ln^3 x)]",
     lambda X, L: X / L ** 3, 89693, None, "P. Dusart, Ramanujan J. 45 (2018) 227-251"),
    ("Cully-Hugill and Lee 2022: a prime in (x(1 - 1/2.5e11), x] for x >= e^600",
     lambda X, L: X / mp.mpf("2.5e11"), mp.e ** 600, None, "M. Cully-Hugill, E. S. Lee, arXiv:2103.05986"),
    ("Computed prime gaps: p_(n+1) - p_n < ln^2 p_n for 11 <= p_n <= 4e18",
     lambda X, L: L ** 2, 11, mp.mpf("4e18"), "T. Oliveira e Silva, S. Herzog, S. Pardi, Math. Comp. 83 (2014)"),
]


# ---------------------------------------------------------------------------
# Exact small range (sieve)
# ---------------------------------------------------------------------------
EXACT_LIMIT = 10**7


@lru_cache(maxsize=4)
def sieve(n: int) -> np.ndarray:
    """is_prime[0..n] as a boolean array."""
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    s[4::2] = False
    for p in range(3, math.isqrt(n) + 1, 2):
        if s[p]:
            s[p * p::2 * p] = False
    return s


@lru_cache(maxsize=4)
def _prime_counts(n: int) -> np.ndarray:
    return np.cumsum(sieve(n), dtype=np.int64)


@lru_cache(maxsize=4)
def _primes(n: int) -> np.ndarray:
    return np.flatnonzero(sieve(n))


def pi_exact(x, limit: int = EXACT_LIMIT) -> int:
    n = int(math.floor(x))
    if n < 2:
        return 0
    if n > limit:
        raise ValueError(f"exact pi only up to {limit}")
    return int(_prime_counts(limit)[n])


# ---------------------------------------------------------------------------
# Input parsing and precision
# ---------------------------------------------------------------------------
_DEC = re.compile(r"^\s*([0-9]+)(?:\.([0-9]*))?(?:[eE]\+?([0-9]+))?\s*$")
_POW = re.compile(r"^\s*([0-9]+)\s*(?:\^|\*\*)\s*([0-9]+)\s*$")


def parse_x(v):
    """Integers stay exact ints: '1e100', '10^100', '2**256', '12345'. Other decimals become mpf."""
    if isinstance(v, bool):
        raise TypeError("bool is not a number")
    if isinstance(v, int):
        return v
    if isinstance(v, (float, mp.mpf)):
        return mp.mpf(v)
    s = str(v).strip().replace("_", "")
    m = _POW.match(s)
    if m:
        base, exp = int(m.group(1)), int(m.group(2))
        if exp > 200000:
            raise ValueError("exponent too large")
        return base ** exp
    m = _DEC.match(s)
    if m:
        whole, frac, exp = m.group(1), m.group(2) or "", int(m.group(3) or 0)
        if exp > 200000:
            raise ValueError("exponent too large")
        frac = frac.rstrip("0")
        if len(frac) <= exp:
            return int(whole + frac + "0" * (exp - len(frac)))
        with mp.workdps(max(60, len(whole) + len(frac) + 20)):
            return mp.mpf(s)
    raise ValueError(f"cannot parse number: {v!r}")


def _log10_mag(x) -> float:
    if isinstance(x, int):
        return max(1.0, x.bit_length() * 0.3010299956639812)
    return max(1.0, float(mp.log10(x)))


def _workdps(x) -> int:
    # absolute rounding error of li(x) is about li(x) * 10^-dps; the error term is about sqrt(x),
    # so half the digits of x plus a 40-digit cushion keeps rounding far below the error term.
    return max(50, int(_log10_mag(x) / 2) + 40)


def _slack(v, dps: int):
    return abs(v) * mp.mpf(10) ** (-(dps - 12)) + 1


def _int_floor(v) -> int:
    return int(mp.floor(v))


def _int_ceil(v) -> int:
    return int(mp.ceil(v))


def _sci(v, digits: int = 6) -> str:
    return mp.nstr(mp.mpf(v), digits)


def common_prefix_digits(lo: int, hi: int) -> int:
    """Number of leading decimal digits shared by every integer in [lo, hi]."""
    a, b = str(lo), str(hi)
    if len(a) != len(b):
        return 0
    k = 0
    while k < len(a) and a[k] == b[k]:
        k += 1
    return k


# ---------------------------------------------------------------------------
# Exact integer arithmetic with directed rounding (no floating point, no error estimates).
# A value v is held as an integer near v * S, S = 10^Q. Every *_lo routine rounds each step
# down and drops positive tails, so it returns a guaranteed lower bound; every *_hi routine
# rounds each step up and adds an explicit bound for the tail, so it returns an upper bound.
# ---------------------------------------------------------------------------
GAMMA_DIGITS = (  # floor(Euler's gamma * 10^700); checked against mpmath in the tests
    "5772156649015328606065120900824024310421593359399235988057672348848677267776646709369470632917467495"
    "1463144724980708248096050401448654283622417399764492353625350033374293733773767394279259525824709491"
    "6008735203948165670853233151776611528621199501507984793745085705740029921354786146694029604325421519"
    "0587755352673313992540129674205137541395491116851028079842348775872050384310939973613725530608893312"
    "6760017247953783675927135157722610273492913940798430103417771778088154957066107501016191663340152278"
    "9358679654972520362128792265559536696281763887927268013243101047650596370394739495763890657296792960"
    "1009015125195950922243501409349871228247949747195646976318506676129063811051824197444867836380861749")
PI_LOW = (3141592653589793238462643383279502884197, 10**39)   # a rational strictly below pi


def _ceil_div(a: int, b: int) -> int:
    return -((-a) // b)


def _atanh2_lo(N: int, D: int, S: int) -> int:
    """Lower bound for 2 atanh(N/D) * S, 0 <= N/D <= 1/3."""
    if N == 0:
        return 0
    N2, D2 = N * N, D * D
    term, total, j = S * N // D, 0, 1
    while term > 0:
        total += term // j
        term = term * N2 // D2
        j += 2
    return 2 * total


def _atanh2_hi(N: int, D: int, S: int) -> int:
    """Upper bound for 2 atanh(N/D) * S, 0 <= N/D <= 1/3 (the tail after a term <= 1 is under 1/8)."""
    if N == 0:
        return 0
    N2, D2 = N * N, D * D
    term, total, j = _ceil_div(S * N, D), 0, 1
    while True:
        total += _ceil_div(term, j)
        if term <= 1:
            total += 1                          # tail: sum of z^(2i) * term over i >= 1 < term / 8
            break
        term = _ceil_div(term * N2, D2)
        j += 2
    return 2 * total


@lru_cache(maxsize=16)
def _ln2_bounds(S: int):
    return _atanh2_lo(1, 3, S), _atanh2_hi(1, 3, S)


def ln_int_bounds(x: int, S: int):
    """(lo, hi) with lo <= ln(x) * S <= hi, for an integer x >= 1."""
    b = x.bit_length() - 1
    p = 1 << b
    l2lo, l2hi = _ln2_bounds(S)
    return b * l2lo + _atanh2_lo(x - p, x + p, S), b * l2hi + _atanh2_hi(x - p, x + p, S)


def _ln_fixed_lo(T: int, S: int) -> int:
    """Lower bound for ln(T / S) * S, T >= S."""
    e = max(T.bit_length() - S.bit_length(), 0)
    p = S << e
    while p > T:
        e -= 1
        p >>= 1
    while (p << 1) <= T:
        e += 1
        p <<= 1
    return e * _ln2_bounds(S)[0] + _atanh2_lo(T - p, T + p, S)


def _ln_fixed_hi(T: int, S: int) -> int:
    e = max(T.bit_length() - S.bit_length(), 0)
    p = S << e
    while p > T:
        e -= 1
        p >>= 1
    while (p << 1) <= T:
        e += 1
        p <<= 1
    return e * _ln2_bounds(S)[1] + _atanh2_hi(T - p, T + p, S)


def _ei_series_lo(T: int, S: int) -> int:
    """Lower bound for sum_k (T/S)^k / (k k!) * S."""
    a, total, k = S, 0, 1
    while True:
        a = a * T // (S * k)
        term = a // k
        total += term
        if term == 0 and k * S > 2 * T:
            return total                      # positive tail omitted
        k += 1


def _ei_series_hi(T: int, S: int) -> int:
    """Upper bound: once k > 2t each term is under half the previous, so the tail is at most a_k."""
    a, total, k = S, 0, 1
    while True:
        a = _ceil_div(a * T, S * k)
        total += _ceil_div(a, k)
        if a <= 1 and k * S > 2 * T:
            return total + 1
        k += 1


def _q_for(x: int) -> int:
    return len(str(x)) // 2 + 60


def li_bounds(x: int, Q: int | None = None):
    """(lo, hi, S): lo <= li(x) * S <= hi for an integer x >= 3, li = gamma + ln ln x + sum (ln x)^k/(k k!)."""
    if x < 3:
        raise ValueError("li_bounds needs x >= 3")
    Q = Q or _q_for(x)
    if Q > len(GAMMA_DIGITS):
        raise ValueError("input too large for the stored constants")
    S = 10 ** Q
    Tlo, Thi = ln_int_bounds(x, S)
    G = int(GAMMA_DIGITS[:Q])
    lo = G + _ln_fixed_lo(Tlo, S) + _ei_series_lo(Tlo, S)
    hi = G + 1 + _ln_fixed_hi(Thi, S) + _ei_series_hi(Thi, S)
    return lo, hi, S, Tlo, Thi


def _e_pi_hi(x: int, Thi: int) -> int:
    """Upper bound, in units of 1/S, for sqrt(x) ln x / (8 pi)."""
    return _ceil_div(_isqrt_ceil(x) * Thi * PI_LOW[1], 8 * PI_LOW[0])


def _isqrt_ceil(x: int) -> int:
    r = math.isqrt(x)
    return r if r * r == x else r + 1


# ---------------------------------------------------------------------------
# Error terms (mpmath versions, used for display and for starting points only)
# ---------------------------------------------------------------------------
def E_pi(X):
    return mp.sqrt(X) * mp.log(X) / (8 * mp.pi)


def E_psi(X):
    return mp.sqrt(X) * mp.log(X) ** 2 / (8 * mp.pi)


def W_gap(X):
    return THEOREMS["gap"]["c"] * mp.sqrt(X) * mp.log(X)


# ---------------------------------------------------------------------------
# pi(x)
# ---------------------------------------------------------------------------
def _pi_units(n: int):
    """Exact bounds in units of 1/S: pi(n) > (li_lo - E_hi)/S and pi(n) < (li_hi + E_hi)/S."""
    li_lo, li_hi, S, Tlo, Thi = li_bounds(n)
    E = _e_pi_hi(n, Thi)
    return li_lo - E, li_hi + E, S, li_lo, E


def pi_rh_bounds(x):
    """Integers lo <= pi(x) <= hi from Schoenfeld's bound (x >= 2657), with directed rounding only."""
    x = parse_x(x)
    if x < 2657:
        raise ValueError("Schoenfeld's pi bound needs x >= 2657")
    n = x if isinstance(x, int) else _int_floor(x)          # pi(x) = pi(floor x)
    low, high, S, li_lo, E = _pi_units(n)
    lo = low // S + 1                                       # pi(n) > low / S
    hi = _ceil_div(high, S) - 1                             # pi(n) < high / S
    dps = _workdps(n)
    with mp.workdps(dps):
        return {"lo": lo, "hi": hi, "li": mp.mpf(li_lo) / S, "halfwidth": mp.mpf(E) / S, "dps": dps}


def _best(registry, x):
    x = parse_x(x)
    best = None
    with mp.workdps(_workdps(x)):
        X = mp.mpf(x)
        LX = mp.log(X)
        for name, f, x0, x1, src in registry:
            if X >= x0 and (x1 is None or X <= x1):
                b = f(X, LX)
                if best is None or b < best[1]:
                    best = (name, b, src)
    return best


def best_unconditional_pi(x):
    """Smallest unconditional explicit bound on |pi(x) - li(x)| in UNCONDITIONAL_PI valid at x, or None."""
    return _best(UNCONDITIONAL_PI, x)


def pi_interval(x):
    x = parse_x(x)
    if x < 2:
        return {"quantity": "pi(x)", "x": str(x), "lo": 0, "hi": 0, "method": "exact"}
    if x <= EXACT_LIMIT:
        v = pi_exact(x)
        return {"quantity": "pi(x)", "x": str(x), "lo": v, "hi": v, "method": "exact sieve"}
    r = pi_rh_bounds(x)
    out = {
        "quantity": "pi(x)", "x": str(x), "lo": r["lo"], "hi": r["hi"],
        "li_x": _sci(r["li"], 25), "halfwidth": _sci(r["halfwidth"]),
        "certified_leading_digits": common_prefix_digits(r["lo"], r["hi"]),
        "total_digits": len(str(r["hi"])),
        "method": "Schoenfeld 1976 (RH, now a theorem)",
    }
    u = best_unconditional_pi(x)
    if u:
        with mp.workdps(r["dps"]):
            uw = u[1]
            ulo, uhi = _int_floor(r["li"] - uw), _int_ceil(r["li"] + uw)
            out["before_RH"] = {
                "bound": u[0], "halfwidth": _sci(uw),
                "certified_leading_digits": common_prefix_digits(max(ulo, 0), uhi),
                "orders_of_magnitude_tighter_now": float(mp.log10(uw / r["halfwidth"])),
            }
    return out


# ---------------------------------------------------------------------------
# theta(x), psi(x) and the primorial
# ---------------------------------------------------------------------------
def _chebyshev_interval(x, x0, name, digits_key):
    """theta or psi at an integer x: (x - E2, x + E2), E2 = sqrt(x) ln^2 x / (8 pi), exact bounds."""
    x = parse_x(x)
    if x < x0:
        raise ValueError(f"Schoenfeld's {name} bound needs x >= {x0}")
    n = x if isinstance(x, int) else _int_floor(x)   # theta, psi are step functions: value at floor(x)
    S = 10 ** _q_for(n)
    _, Thi = ln_int_bounds(n, S)
    E2 = _ceil_div(_isqrt_ceil(n) * Thi * Thi * PI_LOW[1], S * 8 * PI_LOW[0])     # units of 1/S
    lo_u, hi_u = n * S - E2, n * S + E2
    l10lo, l10hi = ln_int_bounds(10, S)
    # digits of N = e^value are floor(value / ln 10) + 1
    dlo, dhi = lo_u // l10hi + 1, hi_u // l10lo + 1
    with mp.workdps(_workdps(n) + 10):
        return {"quantity": name, "x": str(n), "lo": _sci(mp.mpf(lo_u) / S, 30), "hi": _sci(mp.mpf(hi_u) / S, 30),
                "halfwidth": _sci(mp.mpf(E2) / S), digits_key: [dlo, dhi],
                "method": "Schoenfeld 1976 (RH, now a theorem); exact integer bounds"}


def theta_interval(x):
    return _chebyshev_interval(x, 599, "theta(x) = log of the product of all primes <= x", "primorial_decimal_digits")


def psi_interval(x):
    return _chebyshev_interval(x, 74, "psi(x) = log lcm(1..x)", "lcm_decimal_digits")   # theorem: x >= 73.2


# ---------------------------------------------------------------------------
# Monotone inversion helpers (results are re-checked at the returned endpoint)
# ---------------------------------------------------------------------------
def _newton_increasing(f, df, target, t0, tmin):
    t = max(mp.mpf(t0), mp.mpf(tmin))
    for _ in range(200):
        step = (f(t) - target) / df(t)
        t_new = t - step
        if t_new < tmin:
            t_new = (t + tmin) / 2
        if abs(t_new - t) <= abs(t) * mp.mpf(10) ** (-(mp.mp.dps - 5)):
            return t_new
        t = t_new
    return t


def nth_prime_rh_bounds(n):
    """Integers lo <= p_n <= hi from Schoenfeld's pi bound, with endpoint re-checks."""
    n = int(parse_x(n))
    if n < 420:  # li(2657) + E(2657) = 415.8, so below this the root sits under Schoenfeld's range
        raise ValueError("use the exact path for n < 420")
    with mp.workdps(60):
        guess = mp.mpf(n) * (mp.log(n) + mp.log(mp.log(n)))   # no float overflow for huge n
    dps = _workdps(guess) + 10
    with mp.workdps(dps):
        guess = mp.mpf(n) * (mp.log(n) + mp.log(mp.log(n)))
        N = mp.mpf(n)
        f_plus = lambda t: mp.li(t) + E_pi(t)            # pi(t) < f_plus(t)
        f_minus = lambda t: mp.li(t) - E_pi(t)           # pi(t) > f_minus(t)
        dE = lambda t: (mp.log(t) + 2) / (16 * mp.pi * mp.sqrt(t))
        d_plus = lambda t: 1 / mp.log(t) + dE(t)
        d_minus = lambda t: 1 / mp.log(t) - dE(t)
        a = _newton_increasing(f_plus, d_plus, N, guess, 2657)        # p_n > a (approximately)
        b = _newton_increasing(f_minus, d_minus, N - 1, guess, 2657)  # p_n <= b (approximately)
        step = _int_ceil((b - a) * mp.mpf(10) ** -12) + 1
        a_int, b_int = _int_floor(a), _int_ceil(b)
    # certify with exact integers: pi(A) < n  <=>  p_n > A ;  pi(B) >= n  <=>  p_n <= B
    for _ in range(60):
        A = a_int - step
        if A >= 2657 and _pi_units(A)[1] <= n * _pi_units(A)[2]:      # pi(A) < (li_hi+E)/S <= n
            break
        step *= 10
    else:
        raise ArithmeticError("could not certify the lower endpoint")
    step = _int_ceil(mp.mpf(b_int - a_int) * mp.mpf(10) ** -12) + 1
    for _ in range(60):
        B = b_int + step
        low, _, S, _, _ = _pi_units(B)
        if low >= (n - 1) * S:                                         # pi(B) > (li_lo-E)/S >= n-1
            break
        step *= 10
    else:
        raise ArithmeticError("could not certify the upper endpoint")
    return {"lo": A + 1, "hi": B, "dps": dps}


def nth_prime_interval(n):
    n = int(parse_x(n))
    if n < 1:
        raise ValueError("n must be >= 1")
    primes = _primes(EXACT_LIMIT)
    if n <= len(primes):
        p = int(primes[n - 1])
        return {"quantity": "p_n", "n": n, "lo": p, "hi": p, "method": "exact sieve"}
    r = nth_prime_rh_bounds(n)
    return {"quantity": "p_n", "n": n, "lo": r["lo"], "hi": r["hi"],
            "width": r["hi"] - r["lo"] + 1,
            "certified_leading_digits": common_prefix_digits(r["lo"], r["hi"]),
            "total_digits": len(str(r["hi"])),
            "method": "inverted Schoenfeld bound (RH, now a theorem)"}


# ---------------------------------------------------------------------------
# Guaranteed prime windows
# ---------------------------------------------------------------------------
def prime_window(x):
    """Integer ranges that must contain a prime, at or above x and at or below x.

    Above: Theorem 5 at x gives a prime in [x, x + w(x)], w = (22/25) sqrt(x) log x, x >= 4.
    Below: Theorem 5 at a point y with y + w(y) <= x gives a prime in [y, x]; y is found by
    Newton's method and then re-checked, so the result does not depend on the solver.
    """
    x = parse_x(x)
    if x < 4:
        raise ValueError("needs x >= 4")
    X = x if isinstance(x, int) else _int_ceil(x)          # the theorem at the integer X >= x
    Xf = x if isinstance(x, int) else _int_floor(x)
    above = (X, X + _w_gap_hi_int(X))                       # a prime in [X, X + w(X)]
    below = None
    dps = _workdps(X) + 5
    with mp.workdps(dps):
        f = lambda t: t + W_gap(t)                          # increasing for t >= 4
        df = lambda t: 1 + THEOREMS["gap"]["c"] * (mp.log(t) + 2) / (2 * mp.sqrt(t))
        w = W_gap(mp.mpf(X))
        if f(mp.mpf(4)) <= Xf:
            y = _int_floor(_newton_increasing(f, df, mp.mpf(Xf), mp.mpf(Xf) - w, 4))
            step = _int_ceil(w * mp.mpf(10) ** -12) + 1
            for _ in range(60):
                yc = y - step
                if yc >= 4 and yc + _w_gap_hi_int(yc) <= Xf:   # exact: a prime in [yc, yc + w(yc)] ⊂ [yc, x]
                    below = (yc, Xf)
                    break
                step *= 10
        return {"quantity": "guaranteed prime windows", "x": str(x),
                "above": above, "below": below,
                "width": _sci(w), "relative_width": _sci(w / X),
                "method": "Carneiro-Milinovich-Soundararajan 2019, Theorem 5 (RH, now a theorem)"}


def _w_gap_hi_int(X: int) -> int:
    """An integer >= (22/25) sqrt(X) ln X, from exact upper bounds."""
    S = 10 ** _q_for(X)
    _, Thi = ln_int_bounds(X, S)
    return _ceil_div(22 * _isqrt_ceil(X) * Thi, 25 * S)


def next_prime_bound(y):
    """An integer B such that the next prime after y is at most B (Theorem 5 at floor(y) + 1)."""
    y = parse_x(y)
    x0 = _int_floor(mp.mpf(y)) + 1 if not isinstance(y, int) else y + 1
    if x0 < 4:
        return {"quantity": "next prime after y", "y": str(y), "B": 5, "method": "trivial"}
    B = x0 + _w_gap_hi_int(x0)                               # a prime in [x0, x0 + w(x0)]
    return {"quantity": "next prime after y", "y": str(y), "B": B,
            "max_numbers_to_test": B - x0 + 1,
            "method": "Carneiro-Milinovich-Soundararajan 2019, Theorem 5 (RH, now a theorem)"}


def best_unconditional_window(x):
    return _best(UNCONDITIONAL_WINDOW, x)


# ---------------------------------------------------------------------------
# Deterministic primality: Miller's test with Bach's bound (needs GRH for Dirichlet L-functions)
# ---------------------------------------------------------------------------
_SMALL_PRIMES = [int(p) for p in np.flatnonzero(sieve(1000))]


def _strong_probable_prime(n: int, a: int, d: int, s: int) -> bool:
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def bach_bound(n: int) -> int:
    """An integer >= every base a < 2 (ln n)^2 (Bach 1990), from an exact upper bound on ln n."""
    S = 10 ** 30
    _, Thi = ln_int_bounds(n, S)
    return 2 * Thi * Thi // (S * S)


def is_prime_deterministic(n) -> bool:
    """Deterministic Miller test: n > 2 odd is prime iff it passes the strong test for every base
    a < 2 (ln n)^2.  Correct given GRH for Dirichlet L-functions (Bach, Math. Comp. 55 (1990) 355-380).
    The released Lean library also proves that GRH (theorems LFunction_ne_zero_of_half_lt_re and
    LFunction_eq_zero_re), accepted by independent checkers on the MIT verification page; the
    21 Nov 2026 joint statement certifies RH for zeta only.  Cost: about 2 (ln n)^2 modular powers.
    """
    n = int(parse_x(n))
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    if n < 1000 * 1000:
        return True                      # no prime factor below 1000 and n < 1000^2
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # base 2 first rejects almost every composite cheaply; then every base up to Bach's bound
    for a in range(2, min(bach_bound(n), n - 2) + 1):
        if not _strong_probable_prime(n, a, d, s):
            return False
    return True


def next_prime_certified(y):
    """The first prime > y, certified by the deterministic test, plus the RH window it must lie in."""
    y = int(parse_x(y))
    B = next_prime_bound(y)["B"]
    c = y + 1
    while c <= B:
        if c % 2 == 0 and c != 2:
            c += 1
            continue
        if is_prime_deterministic(c):
            return {"quantity": "first prime after y", "y": str(y), "p": c, "offset": c - y,
                    "window_end": B, "bases_checked": bach_bound(c),
                    "method": "search inside the RH window; primality by Miller's test with Bach's bound (GRH)"}
        c += 1
    raise ArithmeticError("no prime in the window: one of the theorems would be false")


# ---------------------------------------------------------------------------
# Counting primes in a range, e.g. all k-bit primes
# ---------------------------------------------------------------------------
def count_primes_between(a, b):
    """Bounds on #{primes p : a < p <= b} = pi(b) - pi(a)."""
    a, b = parse_x(a), parse_x(b)
    A, B = pi_interval(a), pi_interval(b)
    lo, hi = B["lo"] - A["hi"], B["hi"] - A["lo"]
    return {"quantity": "primes in (a, b]", "a": str(a), "b": str(b), "lo": lo, "hi": hi,
            "certified_leading_digits": common_prefix_digits(lo, hi), "total_digits": len(str(hi))}


def count_k_bit_primes(k: int):
    k = int(k)
    if k < 3:
        raise ValueError("k >= 3")
    r = count_primes_between(2 ** (k - 1), 2 ** k)  # 2^(k-1) and 2^k are not prime for k >= 3
    r["quantity"] = f"number of {k}-bit primes"
    return r


# ---------------------------------------------------------------------------
# Robin's inequality
# ---------------------------------------------------------------------------
def robin_bound(n):
    n = int(parse_x(n))
    if n <= 5040:
        raise ValueError("Robin's inequality is stated for n > 5040")
    with mp.workdps(max(50, len(str(n)) + 20)):
        N = mp.mpf(n)
        ll = mp.log(mp.log(N))
        rh = mp.exp(mp.euler) * N * ll
        before = rh + mp.mpf("0.6483") * N / ll   # Robin 1984, unconditional, n >= 3
        return {"quantity": "sigma(n) upper bound", "n": n, "now": _sci(rh, 20),
                "before_RH": _sci(before, 20), "method": "Robin 1984 (RH, now a theorem)"}


# ---------------------------------------------------------------------------
# Demo table
# ---------------------------------------------------------------------------
DEMO_X = ["1e30", "1e50", "1e100", "2^256", "2^1024", "2^2048", "1e1000"]


def demo() -> str:
    lines = ["## pi(x): certified intervals now vs best explicit bound before RH", "",
             "| x | pi(x) halfwidth now | leading digits now | best before | leading digits before | tighter by |",
             "|---|---|---|---|---|---|"]
    for xs in DEMO_X:
        r = pi_interval(xs)
        b = r.get("before_RH")
        lines.append(f"| {xs} | {r['halfwidth']} | {r['certified_leading_digits']} of {r['total_digits']} | "
                     f"{b['halfwidth'] if b else 'n/a'} | {b['certified_leading_digits'] if b else 'n/a'} | "
                     f"{('10^%.1f' % b['orders_of_magnitude_tighter_now']) if b else 'n/a'} |")
    lines += ["", "## Guaranteed prime window near x", "",
              "| x | window now (22/25 sqrt(x) log x) | relative | best before | tighter by |",
              "|---|---|---|---|---|"]
    for xs in ["1e30", "1e100", "2^256", "2^1024", "2^2048", "2^4096"]:
        w = prime_window(xs)
        u = best_unconditional_window(xs)
        with mp.workdps(_workdps(parse_x(xs))):
            gain = ("10^%.1f" % float(mp.log10(u[1] / mp.mpf(w["width"])))) if u else "n/a"
        lines.append(f"| {xs} | {w['width']} | {w['relative_width']} | {_sci(u[1]) if u else 'n/a'} | {gain} |")
    lines += ["", "## n-th prime", "", "| n | certified leading digits of p_n |", "|---|---|"]
    for ns in ["1e30", "1e50", "1e100"]:
        r = nth_prime_interval(ns)
        lines.append(f"| {ns} | {r['certified_leading_digits']} of {r['total_digits']} |")
    lines += ["", "## Number of k-bit primes", "", "| k | certified leading digits |", "|---|---|"]
    for k in [256, 1024, 2048]:
        r = count_k_bit_primes(k)
        lines.append(f"| {k} | {r['certified_leading_digits']} of {r['total_digits']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _jsonable(v)
        elif isinstance(v, (int, float, str, list)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, arg in [("pi", "x"), ("theta", "x"), ("psi", "x"), ("nthprime", "n"), ("window", "x"),
                      ("nextprime", "y"), ("count-bits", "k"), ("robin", "n")]:
        sp = sub.add_parser(name)
        sp.add_argument(arg)
    sub.add_parser("demo")
    sub.add_parser("theorems")
    a = p.parse_args(argv)
    fn = {"pi": pi_interval, "theta": theta_interval, "psi": psi_interval, "nthprime": nth_prime_interval,
          "window": prime_window, "nextprime": next_prime_bound, "count-bits": count_k_bit_primes,
          "robin": robin_bound}
    if a.cmd == "demo":
        print(demo())
    elif a.cmd == "theorems":
        print(json.dumps({k: {"statement": v["statement"], "source": v["source"]} for k, v in THEOREMS.items()},
                         indent=2))
    else:
        val = vars(a)[{"pi": "x", "theta": "x", "psi": "x", "nthprime": "n", "window": "x", "nextprime": "y",
                       "count-bits": "k", "robin": "n"}[a.cmd]]
        print(json.dumps(_jsonable(fn[a.cmd](val)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
