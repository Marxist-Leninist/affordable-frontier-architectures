"""Regenerate li_fixture_v3.json (about 6 MB) for test_v3.js from the Python toolkit and mpmath.

Run from anywhere:  python browser/tests/make_fixture.py   (a few minutes)
Inputs: every 10^k for 4 <= k <= 1000, 2^k for k = 12, 19, ..., 3320, 200 seeded random integers,
and edge values around 2657 and Johnston's 1.101e26.
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for up in (os.path.join(HERE, "..", ".."), os.path.join(HERE, "..", "..", "..")):  # package or repo layout
    if os.path.exists(os.path.join(up, "rh_unlocked.py")):
        sys.path.insert(0, os.path.normpath(up))
        break
import mpmath as mp            # noqa: E402
import rh_unlocked as R        # noqa: E402

xs = {10**k for k in range(4, 1001)} | {2**k for k in range(12, 3322, 7)}
rng = random.Random(1122)
for _ in range(200):
    d = rng.randrange(4, 1000)
    xs.add(rng.randrange(10**(d - 1), 10**d))
xs |= {2657, 2658, 2659, 3000, 10**12, 1101 * 10**23, 1101 * 10**23 + 1}
cases = []
for x in sorted(v for v in xs if 2657 <= v <= 10**1000):
    lo, hi, S, _, Thi = R.li_bounds(x)
    E = R._e_pi_hi(x, Thi)                              # the same integers pi_rh_bounds uses
    pi_lo, pi_hi = (lo - E) // S + 1, R._ceil_div(hi + E, S) - 1
    with mp.workdps(len(str(x)) // 2 + 80):              # enough for the loose bracket check in the test
        li_floor = int(mp.floor(mp.li(mp.mpf(x))))
    cases.append({"x": str(x), "li_floor": str(li_floor), "li_lo": str(lo), "li_hi": str(hi),
                  "Q": len(str(S)) - 1, "pi_lo": str(pi_lo), "pi_hi": str(pi_hi)})
gamma = open(os.path.join(HERE, "gamma.txt"), encoding="ascii").read().strip()
json.dump({"gamma_digits": gamma, "cases": cases}, open(os.path.join(HERE, "li_fixture_v3.json"), "w"))
print("wrote", len(cases), "cases")
