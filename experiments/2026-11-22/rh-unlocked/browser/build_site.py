"""Build the staged zqx.lat/riemann-hypothesis page from the LIVE files (exact-text edits only).

Input:  site/live2/index.html (live copy, sha256 checked), site/stage/prime-bounds-v3.js,
        site/stage/tools/certified-prime-bounds.json, site/next_prime_1e100.json
Output: site/stage/index.html and site/stage/tools/* (toolkit, tests, data).
Every replacement must match exactly once, or the build stops.
"""
import hashlib
import html
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIVE = os.path.join(HERE, "live2", "index.html")
LIVE_SHA = "fc191969654155649f58ba576d9a38e1ca74b876481ccbc91441649161f6fa25"
STAGE = os.path.join(HERE, "stage")

src = open(LIVE, "rb").read()
if hashlib.sha256(src).hexdigest() != LIVE_SHA:
    sys.exit("live index.html changed since it was pulled; pull again and rebuild")
page = src.decode("utf-8")


def replace_once(old, new):
    global page
    n = page.count(old)
    if n != 1:
        sys.exit(f"expected exactly one match, found {n}: {old[:80]!r}")
    page = page.replace(old, new)


def group_prefix(prefix, total):
    out = ""
    for i, ch in enumerate(prefix):
        if i > 0 and (total - i) % 3 == 0:
            out += ","
        out += ch
    return out


def sup(expr):  # "10^100" -> 10<sup>100</sup>
    base, exp = expr.split("^")
    return f"{base}<sup>{exp}</sup>"


data = json.load(open(os.path.join(STAGE, "tools", "certified-prime-bounds.json")))
nxt = json.load(open(os.path.join(HERE, "next_prime_1e100.json")))
assert int(nxt["offset"]) == 267 and int(nxt["p"]) == 10**100 + 267

# ---- 1. script, input range, examples, prime-count card ----------------------------------
replace_once('<script src="./prime-bounds-v2.js" defer></script>', '<script src="./prime-bounds-v3.js" defer></script>')
replace_once('maxlength="108"', 'maxlength="1008"')
replace_once('<p id="prime-bounds-hint">From 0 through 10<sup>100</sup>. Type <code>10^12</code>, <code>2^256</code>, or the full number without commas.</p>',
             '<p id="prime-bounds-hint">From 0 through 10<sup>1000</sup>. Type <code>10^12</code>, <code>2^256</code>, <code>2^2048</code>, or the full number without commas.</p>')
replace_once('<button type="button" data-prime-x="2^256" aria-label="Try 2 to the power 256">Try 2<sup>256</sup></button></div>',
             '<button type="button" data-prime-x="2^256" aria-label="Try 2 to the power 256">Try 2<sup>256</sup></button>'
             '<button type="button" data-prime-x="2^2048" aria-label="Try 2 to the power 2048">Try 2<sup>2048</sup></button>'
             '<button type="button" data-prime-x="10^1000" aria-label="Try 10 to the power 1000">Try 10<sup>1000</sup></button></div>')
replace_once('<article class="card"><h3 id="prime-bounds-count-label">Prime-count error radius</h3><p class="prime-number" id="prime-bounds-count"></p><p id="prime-bounds-count-note"></p></article>\n</div>',
             '<article class="card"><h3 id="prime-bounds-count-label">Prime-count error radius</h3><p class="prime-number" id="prime-bounds-count"></p><p id="prime-bounds-count-note"></p></article>\n'
             '<article class="card" id="prime-bounds-pi-card" hidden><h3>Prime count, π(<var>x</var>)</h3><p class="prime-number" id="prime-bounds-pi"></p><p id="prime-bounds-pi-note"></p>'
             '<details class="prime-ends"><summary>Both ends of the certified interval</summary><p class="prime-mono" id="prime-bounds-pi-ends"></p></details></article>\n</div>')

# ---- 2. worked example: the browser now evaluates li(x) itself --------------------------
replace_once('<dt>Sharper prime-count interval</dt><dd>37,606,850,877 to 37,609,049,685</dd>',
             '<dt>Certified prime-count interval</dt><dd>37,606,850,878 to 37,609,049,684</dd>')
replace_once('<p>The sharper interval uses certified numerical evaluation of the logarithmic integral and outward rounding. The browser avoids evaluating the integral for arbitrary inputs. Its displayed radius is deliberately more conservative.</p>',
             '<p>The calculator now computes this interval itself, for every input from 2,657 up. It bounds the logarithmic integral in exact integer arithmetic: the lower bound rounds every step down and the upper bound rounds every step up, so no step relies on an error estimate. The radius card keeps the simpler, more conservative bound.</p>')

# ---- 3. static card: what the proof lets us state now ------------------------------------
rows_pi = []
for r in data["pi"]:
    start = group_prefix(r["lo"][:min(r["digits"], 15)], r["total"]) + "…"
    before = "none" if r["before_digits_closed_form"] == 0 else f'{r["before_digits_closed_form"]} of {r["total"]}'
    rows_pi.append(f'<tr><th scope="row">{sup(r["x"])}</th><td class="prime-mono">{start}</td>'
                   f'<td>{r["digits"]} of {r["total"]}</td><td>{before}</td></tr>')
labels = {"10^30": "10<sup>30</sup>th prime", "10^50": "10<sup>50</sup>th prime", "10^100": "10<sup>100</sup>th prime"}
rows_more = []
for r in data["nth_prime"]:
    rows_more.append(f'<tr><th scope="row">{labels[r["n"]]}</th><td class="prime-mono">{group_prefix(r["lo"][:min(r["digits"], 15)], r["total"])}…</td><td>{r["digits"]} of {r["total"]}</td></tr>')
for r in data["k_bit_primes"]:
    rows_more.append(f'<tr><th scope="row">Primes with exactly {r["k"]:,} bits</th><td class="prime-mono">{group_prefix(r["lo"][:min(r["digits"], 15)], r["total"])}…</td><td>{r["digits"]} of {r["total"]}</td></tr>')
rows_win = []
for r in data["windows"]:
    if r["x"] in ("10^30", "2^256"):
        continue
    who = "Dusart 2018" if r["before_bound"].startswith("Dusart") else "Cully-Hugill and Lee 2022"
    rows_win.append(f'<tr><th scope="row">{sup(r["x"])}</th><td>{html.escape(r["width_now"])}</td><td>{html.escape(r["width_before"])} ({who})</td><td>10<sup>{r["orders_tighter"]:.0f}</sup></td></tr>')

card = f'''<!-- rh-unlocked:start (Claude laptop lane, 22 Nov 2026; built by tools/build_site.py from tools/rh_unlocked.py) -->
<div class="card prime-worked" id="prime-bounds-now">
<h3>What the proof lets us state now</h3>
<p>Before 20 November each number below was guaranteed only if RH was true. They come from the same exact, outward-rounded arithmetic as the calculator, so every listed digit is certain. Up to 1.101 × 10<sup>26</sup> the prime-count bound was already proved without RH (Johnston), and the prime-gap bound up to 4 × 10<sup>18</sup> by computed prime gaps; above those ranges these statements are new.</p>
<p class="prime-caption">Digits of prime counts nobody has computed</p>
<div class="table-scroll" role="region" aria-label="Digits of prime counts nobody has computed" tabindex="0"><table class="prime-table">
<thead><tr><th scope="col"><var>x</var></th><th scope="col">π(<var>x</var>) begins</th><th scope="col">Certain digits</th><th scope="col">Before RH (closed form)</th></tr></thead>
<tbody>{"".join(rows_pi)}</tbody></table></div>
<p class="prime-small">π(10<sup>29</sup>) is the largest power of ten whose prime count has been computed exactly (<a href="https://oeis.org/A006880">A006880</a>). “Before RH” counts the digits fixed by the best closed-form unconditional bound, <a href="https://arxiv.org/abs/2206.12557">Fiori, Kadiri and Swidinsky 2023</a>: |π(<var>x</var>) − Li(<var>x</var>)| ≤ 9.2211 <var>x</var> √(ln <var>x</var>) e<sup>−0.8476 √(ln <var>x</var>)</sup>. Their tabulated bounds do better than the closed form at the smaller sizes.</p>
<p class="prime-caption">Other counts</p>
<div class="table-scroll" role="region" aria-label="Other certified counts" tabindex="0"><table class="prime-table">
<thead><tr><th scope="col">Quantity</th><th scope="col">Begins</th><th scope="col">Certain digits</th></tr></thead>
<tbody>{"".join(rows_more)}</tbody></table></div>
<p class="prime-caption">Guaranteed prime window next to <var>x</var></p>
<div class="table-scroll" role="region" aria-label="Guaranteed prime window next to x" tabindex="0"><table class="prime-table">
<thead><tr><th scope="col"><var>x</var></th><th scope="col">Now (22/25 √<var>x</var> ln <var>x</var>)</th><th scope="col">Best published without RH</th><th scope="col">Narrower by</th></tr></thead>
<tbody>{"".join(rows_win)}</tbody></table></div>
<dl><dt>First prime after 10<sup>100</sup></dt><dd>10<sup>100</sup> + 267</dd></dl>
<p>That last line is a proof, not a probable-prime check. Every number between 10<sup>100</sup> and 10<sup>100</sup> + 267 has a compositeness witness, and 10<sup>100</sup> + 267 passes Miller’s test for every base up to {int(nxt["bases_checked"]):,}. <a href="https://www.ams.org/journals/mcom/1990-55-191/S0025-5718-1990-1023756-8/">Bach</a> showed that, under GRH for Dirichlet L-functions, every odd composite fails for some base below 2 (ln <var>n</var>)². The same Lean release proves that GRH; <a href="https://math.mit.edu/~drew/grh/">MIT’s record</a> lists its checks, while the joint Lean and Mathlib statement certifies RH for ζ. The window and count results above need only RH.</p>
<p class="research-links"><a href="./tools/rh_unlocked.py">Python toolkit</a><a href="./tools/test_rh_unlocked.py">Its 31 tests</a><a href="./tools/certified-prime-bounds.json">Every endpoint (JSON)</a><a href="./tools/SOURCES.md">Theorems and sources</a><a href="./tools/README.md">How to run it</a></p>
</div>
<!-- rh-unlocked:end -->
'''
replace_once('<details class="card prime-method"><summary>How the guarantees are calculated</summary>',
             card + '<details class="card prime-method"><summary>How the guarantees are calculated</summary>')

# ---- 4. method details: the prime-count interval ------------------------------------------
replace_once('<p>All interval endpoints and radii here use integer arithmetic.',
             '<p><strong>Prime-count interval.</strong> li(<var>x</var>) = γ + ln ln <var>x</var> + Σ<sub><var>k</var>≥1</sub> (ln <var>x</var>)<sup><var>k</var></sup>/(<var>k</var>·<var>k</var>!), with ln <var>x</var> = <var>b</var> ln 2 + 2 atanh((<var>x</var> − 2<sup><var>b</var></sup>)/(<var>x</var> + 2<sup><var>b</var></sup>)). The browser evaluates each part twice with ⌊digits/2⌋ + 60 decimal places, once rounding down and once rounding up with explicit bounds on the omitted tails, and uses π &gt; 3.141592653589793238462643383279502884197 in the error term. Then π(<var>x</var>) is more than the lower bound minus that error and less than the upper bound plus it. It agrees integer for integer with the <a href="./tools/rh_unlocked.py">Python toolkit</a> on 1,676 inputs up to 10<sup>1000</sup>, and its intervals contain every published π(10<sup><var>k</var></sup>) for <var>k</var> ≤ 29.</p>\n'
             '<p>All interval endpoints and radii here use integer arithmetic.')

# ---- 5. FAQ: famous equivalent statements now theorems ------------------------------------
faq = '''<!-- rh-unlocked-faq:start (Claude laptop lane) -->
<details class="card" id="faq-equivalents"><summary>Which famous equivalent statements are now theorems?</summary>
<p><strong>Every statement that was proved equivalent to RH now holds.</strong> Examples: <a href="https://en.wikipedia.org/wiki/Divisor_function#Growth_rate">Robin’s inequality</a>, σ(<var>n</var>) &lt; e<sup>γ</sup><var>n</var> ln ln <var>n</var> for every <var>n</var> &gt; 5040, and <a href="https://arxiv.org/abs/math/0008177">Lagarias’s</a> σ(<var>n</var>) ≤ H<sub><var>n</var></sub> + e<sup>H<sub><var>n</var></sub></sup> ln H<sub><var>n</var></sub>; <a href="https://math.mit.edu/~drew/grh/">MIT’s record</a> lists Lean theorems combining both with the proof. Li’s coefficients are all non-negative. The de Bruijn–Newman constant is exactly 0, because RH is equivalent to Λ ≤ 0 and <a href="https://arxiv.org/abs/1801.05914">Rodgers and Tao</a> had proved Λ ≥ 0.</p>
<p><strong>Results proved “on RH” become unconditional too.</strong> The Lindelöf hypothesis holds: ζ(1/2 + <var>it</var>) grows more slowly than any power of |<var>t</var>|. The 2<var>k</var>-th moments of ζ on the critical line have order <var>T</var>(ln <var>T</var>)<sup><var>k</var>²</sup> for every fixed <var>k</var> &gt; 0 (<a href="https://arxiv.org/abs/1305.4618">Harper’s upper bound</a> with known lower bounds). Montgomery’s pair-correlation theorem holds, and at least 19/27 of the zeros are simple (<a href="https://arxiv.org/abs/1302.5018">Bui and Heath-Brown</a>). The Mertens function satisfies M(<var>x</var>) = O(<var>x</var><sup>1/2+ε</sup>). A 2026 preprint by <a href="https://arxiv.org/abs/2602.22502">Chamberland and Straub</a> combines RH with a computation to put a prime between <var>n</var>³ and (<var>n</var> + 1)³ for every <var>n</var> ≥ 1.</p>
<p><strong>Still open:</strong> whether every zero is simple, the full pair-correlation conjecture, the independence of the zeros’ heights that the prime-race percentages use, and Legendre’s question of a prime between consecutive squares, which RH does not settle.</p>
</details>
<!-- rh-unlocked-faq:end -->
'''
replace_once('\n</div>\n</section>\n<section id="sources"', '\n' + faq + '</div>\n</section>\n<section id="sources"')

# ---- 6. styles ----------------------------------------------------------------------------
css = ('.table-scroll{overflow-x:auto;margin:6px 0 14px}.table-scroll:focus-visible{outline:2px solid var(--accent);outline-offset:3px}'
       '.prime-table{border-collapse:collapse;width:100%;font-size:.95em}'
       '.prime-caption{font-weight:650;margin:16px 0 6px}'
       '.prime-table th,.prime-table td{border-bottom:1px solid var(--line);padding:7px 12px 7px 0;text-align:left;vertical-align:top;white-space:nowrap}'
       '.prime-table thead th{color:var(--muted);font-weight:600}'
       '.prime-mono{font-family:var(--mono);overflow-wrap:anywhere}'
       '.prime-small{color:var(--muted);font-size:.92em}'
       '.prime-ends summary{cursor:pointer;color:var(--muted)}.prime-ends p{margin:8px 0 0;font-size:.9em}'
       '#prime-bounds-results p,.prime-method p{overflow-wrap:anywhere}\n')
replace_once('</style>', css + '</style>')

os.makedirs(STAGE, exist_ok=True)
open(os.path.join(STAGE, "index.html"), "w", encoding="utf-8", newline="\n").write(page)

# ---- 7. downloadable toolkit ----------------------------------------------------------------
tools = os.path.join(STAGE, "tools")
for name in ("rh_unlocked.py", "test_rh_unlocked.py", "README.md", "SOURCES.md"):
    shutil.copyfile(os.path.join(ROOT, name), os.path.join(tools, name))
for sub in ("A006880/b006880.txt", "A006988/b006988.txt"):
    os.makedirs(os.path.join(tools, "data", os.path.dirname(sub)), exist_ok=True)
    shutil.copyfile(os.path.join(ROOT, "data", sub), os.path.join(tools, "data", sub))
shutil.copyfile(os.path.join(HERE, "build_site.py"), os.path.join(tools, "build_site.py"))
print("staged", len(page), "bytes; sha256", hashlib.sha256(page.encode()).hexdigest()[:16])
