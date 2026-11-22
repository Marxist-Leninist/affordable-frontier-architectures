"""Tests for rh_unlocked.

Exhaustive where possible: every real x in [2657, 2e7] for pi, [599, 2e7] for theta,
[73.2, 2e7] for psi, every prime gap below 1e8 for the window theorem, every n in
[5041, 1e6] for Robin. Plus the published tables pi(10^k) (k <= 29) and p_(10^k) (k <= 24)
from OEIS A006880 / A006988 (data/, sha256 in SOURCES.md).
"""
import math
import os
import random

import mpmath as mp
import numpy as np
import pytest
from scipy.special import expi

import rh_unlocked as R

HERE = os.path.dirname(os.path.abspath(__file__))
N_SIEVE = 2 * 10**7
N_GAP = 10**8
INV8PI = 1.0 / (8.0 * math.pi)


def _bfile(name):
    path = os.path.join(HERE, "data", name.split("/")[0], name.split("/")[1])
    out = {}
    with open(path, encoding="ascii") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                k, v = line.split()
                out[int(k)] = int(v)
    return out


PI_POW10 = _bfile("A006880/b006880.txt")      # pi(10^k)
P_POW10 = _bfile("A006988/b006988.txt")       # p_(10^k)


@pytest.fixture(scope="module")
def primes_small():
    return R._primes(N_SIEVE).astype(np.float64)


# ---------------------------------------------------------------------------
# exhaustive checks of the three Schoenfeld bounds over all real x up to 2e7
# ---------------------------------------------------------------------------
def test_pi_bound_every_real_x_up_to_2e7(primes_small):
    p = primes_small
    k = np.arange(1, len(p) + 1, dtype=np.float64)          # pi(x) = k on [p_k, p_{k+1})
    start = np.searchsorted(p, 2657)
    assert p[start] == 2657                                  # 2657 is prime: the range starts on a step
    li = expi(np.log(p))
    E = INV8PI * np.sqrt(p) * np.log(p)
    # (i) k - li(x) < E(x) on [p_k, p_{k+1}): worst at x = p_k
    assert np.all((k - li)[start:] < E[start:])
    # (ii) li(x) - k < E(x): li - E is increasing, so worst as x -> p_{k+1}
    assert np.all((li - E)[start + 1:] < k[start:-1])
    r_up = ((k - li) / E)[start:]
    r_dn = ((li[1:] - k[:-1]) / E[1:])[start:]
    # Schoenfeld's threshold is tight: just above 2657 (x -> 2659) the bound is used at 0.98 of its size
    assert 0.97 < float(r_dn.max()) < 1.0
    big = p[start:] > 10**6
    assert max(float(r_up[big].max()), float(r_dn[big[1:] if len(big) > len(r_dn) else big].max())) < 0.3


def test_theta_bound_every_real_x_up_to_2e7(primes_small):
    p = primes_small
    theta = np.cumsum(np.log(p))
    start = np.searchsorted(p, 599)
    assert p[start] == 599
    E2 = INV8PI * np.sqrt(p) * np.log(p) ** 2
    assert np.all((theta - p)[start:] < E2[start:])                    # worst at x = p_k
    assert np.all((p[1:] - E2[1:])[start:] < theta[:-1][start:])       # worst as x -> p_{k+1}


def test_psi_bound_every_real_x_up_to_2e7(primes_small):
    p = primes_small
    pts, wts = [p], [np.log(p)]
    m = 2
    while 2.0 ** m <= N_SIEVE:
        q = p[p ** m <= N_SIEVE]
        pts.append(q ** m)
        wts.append(np.log(q))
        m += 1
    q = np.concatenate(pts)
    w = np.concatenate(wts)
    order = np.argsort(q, kind="stable")
    q, w = q[order], w[order]
    psi = np.cumsum(w)                                       # psi(x) = psi_j on [q_j, q_{j+1})
    x0 = 73.2
    j0 = np.searchsorted(q, x0, side="right") - 1            # interval containing 73.2
    left = q.copy()
    left[j0] = x0
    E2 = lambda x: INV8PI * np.sqrt(x) * np.log(x) ** 2
    assert np.all((psi - left)[j0:] < E2(left[j0:]))
    assert np.all((q[1:] - E2(q[1:]))[j0:] < psi[:-1][j0:])


# ---------------------------------------------------------------------------
# the module's certified integer intervals
# ---------------------------------------------------------------------------
def test_pi_intervals_contain_exact_values_sample():
    rng = random.Random(20261122)
    counts = R._prime_counts(N_SIEVE)
    xs = [2657, 2658, 2659, 10**4, 10**5, 10**6, 10**7, N_SIEVE] + [rng.randrange(2657, N_SIEVE) for _ in range(3000)]
    for x in xs:
        r = R.pi_rh_bounds(x)
        assert r["lo"] <= int(counts[x]) <= r["hi"], x


def test_pi_powers_of_ten_oeis():
    assert len(PI_POW10) == 30 and PI_POW10[29] == 1520698109714272166094258063
    for k, v in PI_POW10.items():
        if k >= 4:
            r = R.pi_rh_bounds(10**k)
            assert r["lo"] <= v <= r["hi"], k
        if 10**k <= R.EXACT_LIMIT:
            assert R.pi_interval(10**k)["lo"] == v


def test_nth_prime_powers_of_ten_oeis():
    assert P_POW10[24] == 58310039994836584070534263
    for k, v in P_POW10.items():
        if k >= 3:
            r = R.nth_prime_rh_bounds(10**k)
            assert r["lo"] <= v <= r["hi"], k
        if k >= 1 and 10**k <= len(R._primes(R.EXACT_LIMIT)):
            assert R.nth_prime_interval(10**k)["lo"] == v


def test_nth_prime_intervals_contain_exact_values_sample():
    rng = random.Random(7)
    primes = R._primes(N_SIEVE)
    ns = [420, 421, 1000, len(primes)] + [rng.randrange(420, len(primes) + 1) for _ in range(300)]
    with pytest.raises(ValueError):
        R.nth_prime_rh_bounds(419)
    for n in ns:
        r = R.nth_prime_rh_bounds(n)
        assert r["lo"] <= int(primes[n - 1]) <= r["hi"], n


# ---------------------------------------------------------------------------
# prime-gap window (Carneiro-Milinovich-Soundararajan), every gap below 1e8
# ---------------------------------------------------------------------------
def test_gap_window_every_prime_gap_below_1e8():
    """Theorem 5: for every real x >= 4 there is a prime in [x, x + w(x)], w = (22/25) sqrt(x) log x.

    For x in (p, q] (p, q consecutive primes) the first prime >= x is q, so the theorem needs
    q <= x + w(x); x + w(x) is increasing, so the worst case is x -> p, i.e. q - p <= w(p).
    x in [4, 5] needs 5 <= x + w(x), true at x = 4. Pairs start at p = 5.
    """
    p = R._primes(N_GAP).astype(np.float64)
    lo, hi = p[2:-1], p[3:]                                  # (5, 7), (7, 11), ...
    assert lo[0] == 5
    w = 0.88 * np.sqrt(lo) * np.log(lo)
    ratio = (hi - lo) / w
    assert float(ratio.max()) < 1.0
    assert float(ratio[lo > 10**6].max()) < 0.02
    assert 4 + 0.88 * 2 * math.log(4) >= 5


def test_next_prime_bound_random_and_record_gaps():
    primes = R._primes(N_GAP)
    rng = random.Random(42)
    ys = [4, 5, 6, 1000, 10**6, N_GAP - 1000] + [rng.randrange(4, N_GAP - 10**6) for _ in range(2000)]
    # starts of record gaps below 1e8 (OEIS A002386)
    ys += [113, 523, 887, 1129, 1327, 9551, 15683, 19609, 31397, 155921, 360653, 370261,
           492113, 1349533, 1357201, 2010733, 4652353, 17051707, 20831323, 47326693]
    for y in ys:
        nxt = int(primes[np.searchsorted(primes, y, side="right")])
        assert nxt <= R.next_prime_bound(y)["B"], y


def test_prime_windows_contain_a_prime_sample():
    primes = R._primes(N_GAP)
    rng = random.Random(3)
    for x in [4, 5, 10, 100, 2657] + [rng.randrange(4, N_GAP - 10**6) for _ in range(2000)]:
        r = R.prime_window(x)
        a_lo, a_hi = r["above"]
        assert a_lo == x
        i = np.searchsorted(primes, a_lo, side="left")           # first prime >= x
        assert primes[i] <= a_hi, x
        if r["below"] is not None:
            b_lo, b_hi = r["below"]
            assert b_hi == x
            j = np.searchsorted(primes, b_hi, side="right") - 1  # last prime <= x
            assert primes[j] >= b_lo, x
    assert R.prime_window(6)["below"] is None                    # needs y >= 4 with y + w(y) <= x
    b = R.prime_window(10)["below"]
    assert b[0] >= 4 and b[1] == 10 and b[0] <= 7                 # contains 7


def test_exact_li_bounds_bracket_mpmath():
    rng = random.Random(99)
    xs = [2657, 2658, 3001, 10**6, 10**12, 10**30, 10**100, 2**1024, 10**400]
    xs += [rng.randrange(10**(d - 1), 10**d) for d in (5, 9, 17, 33, 65, 129, 257)]
    for x in xs:
        lo, hi, S, Tlo, Thi = R.li_bounds(x)
        with mp.workdps(R._q_for(x) + 40):
            ref = mp.li(mp.mpf(x)) * S
            lnref = mp.log(mp.mpf(x)) * S
            assert lo <= ref <= hi, x
            assert Tlo <= lnref <= Thi, x
            assert (hi - lo) < ref * mp.mpf(10) ** -(R._q_for(x) - 20), x


def test_prime_window_huge_inputs():
    for x in ["1e100", "2^1024", "2^4096", "1e1000"]:
        r = R.prime_window(x)
        assert r["above"][0] <= r["above"][1] and r["below"][0] <= r["below"][1]


# ---------------------------------------------------------------------------
# Robin's inequality for every n in (5040, 1e6]
# ---------------------------------------------------------------------------
def test_robin_every_n_up_to_1e6():
    N = 10**6
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
    n = np.arange(5041, N + 1, dtype=np.float64)
    bound = math.exp(float(mp.euler)) * n * np.log(np.log(n))
    assert np.all(sigma[5041:] < bound)
    # 5040 itself violates it, which is why the theorem starts above 5040
    assert sigma[5040] == 19344 and sigma[5040] > math.exp(float(mp.euler)) * 5040 * math.log(math.log(5040))
    r = R.robin_bound(10**12)
    assert float(r["now"]) < float(r["before_RH"])


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------
def test_li_matches_independent_float_implementation():
    for k in range(4, 16):
        x = 10**k
        a = float(mp.li(x))
        b = float(expi(math.log(x)))
        assert abs(a - b) <= 1e-12 * a


@pytest.mark.parametrize("x", [10**100, 10**500, 2**4096])
def test_li_stable_under_extra_precision(x):
    dps = R._workdps(x)
    with mp.workdps(dps):
        a = mp.li(mp.mpf(x))
        s = R._slack(a, dps)
    with mp.workdps(dps + 40):
        b = mp.li(mp.mpf(x))
        assert abs(a - b) < s / 10**6


@pytest.mark.parametrize("x", [10**100, 10**300, 10**1000])
def test_intervals_nest_under_extra_precision(x, monkeypatch):
    r1 = R.pi_rh_bounds(x)
    orig = R._workdps
    monkeypatch.setattr(R, "_workdps", lambda v: orig(v) + 30)
    r2 = R.pi_rh_bounds(x)
    assert r1["lo"] <= r2["lo"] and r2["hi"] <= r1["hi"]
    with mp.workdps(r1["dps"]):
        assert (r2["lo"] - r1["lo"]) + (r1["hi"] - r2["hi"]) < r1["halfwidth"] * mp.mpf(10) ** -20


def test_monotonicity_used_by_the_inversions():
    for e in np.linspace(math.log(2657), math.log(1e300), 400):
        x = math.exp(e)
        dE = (math.log(x) + 2) / (16 * math.pi * math.sqrt(x))
        assert 1 / math.log(x) - dE > 0                        # li - E_pi increasing
    for e in np.linspace(math.log(4), math.log(1e300), 400):
        x = math.exp(e)
        assert 1 - 0.88 * (math.log(x) + 2) / (2 * math.sqrt(x)) > 0   # x - w(x) increasing
    for e in np.linspace(math.log(599), math.log(1e300), 400):
        x = math.exp(e)
        dE2 = (math.log(x) ** 2 + 4 * math.log(x)) / (16 * math.pi * math.sqrt(x))
        assert 1 - dE2 > 0                                     # x - E_psi increasing


@pytest.mark.parametrize("n", ["1e100", "1e400", "1e1000"])
def test_nth_prime_huge_n(n):
    r = R.nth_prime_interval(n)
    # width ~ sqrt(p) log^2 p / (4 pi): about half the digits, minus ~2 log10(log p)
    assert r["lo"] <= r["hi"] and r["certified_leading_digits"] >= (len(str(r["hi"])) // 2) - 12


def test_count_k_bit_primes_against_sieve():
    counts = R._prime_counts(N_SIEVE)
    exact = int(counts[2**24] - counts[2**23])
    r = R.count_k_bit_primes(24)
    assert r["lo"] <= exact <= r["hi"]
    r20 = R.count_k_bit_primes(20)
    assert r20["lo"] == r20["hi"] == int(counts[2**20] - counts[2**19])


def test_deterministic_primality_matches_sieve():
    s = R.sieve(N_SIEVE)
    for n in list(range(0, 3000)) + list(range(10**6, 10**6 + 40000)):
        assert R.is_prime_deterministic(n) == bool(s[n]), n


def test_deterministic_primality_rejects_strong_pseudoprimes():
    # Carmichael numbers and strong pseudoprimes to many prime bases (Jaeschke; Jiang-Deng; Sorenson-Webster)
    for n in [561, 1105, 1729, 2465, 2821, 6601, 8911, 3215031751, 2152302898747, 3474749660383,
              341550071728321, 3825123056546413051, 318665857834031151167461,
              3317044064679887385961981]:
        assert not R.is_prime_deterministic(n), n
    for p in [2**61 - 1, 2**89 - 1, 2**107 - 1, 2**127 - 1, 10**18 + 3, 10**30 + 57]:
        assert R.is_prime_deterministic(p), p


def test_next_prime_certified_inside_window():
    import sympy
    for y in [10**12, 10**18, 10**24, 10**30, 2**80]:
        r = R.next_prime_certified(y)
        assert r["p"] == sympy.nextprime(y) and y < r["p"] <= r["window_end"], y


def test_unconditional_comparators_are_weaker_than_rh_bounds():
    for xs in ["1e30", "1e100", "2^1024", "1e1000"]:
        b = R.best_unconditional_pi(xs)
        w = R.best_unconditional_window(xs)
        x = R.parse_x(xs)
        with mp.workdps(R._workdps(x)):
            assert b[1] > R.E_pi(mp.mpf(x)) and w[1] > R.W_gap(mp.mpf(x))
    # inside Johnston's range the 'before' bound is Schoenfeld's bound itself
    assert R.best_unconditional_pi(10**20)[0].startswith("Johnston 2022")


def test_parse_x():
    assert R.parse_x("1e100") == 10**100
    assert R.parse_x("2^256") == 2**256 and R.parse_x("2**256") == 2**256
    assert R.parse_x("1.5e3") == 1500 and R.parse_x("12_345") == 12345
    assert isinstance(R.parse_x("2.5"), mp.mpf)
    with pytest.raises(ValueError):
        R.parse_x("__import__('os')")


def test_cli_json(capsys):
    assert R.main(["pi", "1e100"]) == 0
    out = capsys.readouterr().out
    assert '"lo"' in out and '"hi"' in out
    assert R.main(["nextprime", "2^2048"]) == 0
