/* Exact outward integer arithmetic for the zqx Riemann Hypothesis explainer.
   v3 (22 Nov 2026) keeps v2's windows, radius and 2^256 example, accepts inputs up to 10^1000
   (2^k up to 2^3321), and adds a certified interval for pi(x): li(x) is bounded below and above
   in exact integer arithmetic with directed rounding, so no step relies on an error estimate.
   The same algorithm in Python (tools/rh_unlocked.py) gives identical endpoints. */
"use strict";
const MAX_X = 10n ** 1000n;
const COUNT_START = 2657n;
const JOHNSTON_END = 1101n * 10n ** 23n; // 1.101 × 10^26
const SAMPLE_X = 10n ** 12n;
const BINARY_SAMPLE_X = 2n ** 256n;
function parseX(raw) {
  if (typeof raw !== "string" || raw.length > 1008)
    throw new RangeError("Enter a whole number, 10^k, or 2^k, from 0 through 10^1000.");
  const s = raw.trim();
  const sci = /^(10|2)\s*\^\s*(\d{1,4})$/.exec(s);
  let x;
  if (sci) {
    const base = BigInt(sci[1]), k = BigInt(sci[2]);
    if (base === 10n && k > 1000n) throw new RangeError("The largest decimal power accepted is 10^1000.");
    if (base === 2n && k > 3321n) throw new RangeError("The largest binary power accepted is 2^3321.");
    x = base ** k;
  } else if (/^\d{1,1001}$/.test(s)) {
    x = BigInt(s);
  } else {
    throw new RangeError("Enter a whole number or a power such as 10^12 or 2^256.");
  }
  if (x > MAX_X) throw new RangeError("The largest input accepted is 10^1000.");
  return x;
}
function ceilSqrt(x) {
  if (x < 2n) return x;
  let hi = 1n << BigInt(Math.ceil(x.toString(2).length / 2));
  for (;;) {
    const next = (hi + x / hi) >> 1n;
    if (next >= hi) return hi * hi === x ? hi : hi + 1n;
    hi = next;
  }
}
const divCeil = (n, d) => (n + d - 1n) / d;
const rootLogNumerator = (x) => ceilSqrt(x) * BigInt(x.toString(2).length) * 693148n;
const gapWidth = (x) => divCeil(22n * rootLogNumerator(x), 25000000n);
const countRadius = (x) => divCeil(rootLogNumerator(x), 25000000n);
const smallPrimes = (() => {
  const composite = new Uint8Array(Number(COUNT_START)), p = [];
  for (let i = 2; i < composite.length; i++) {
    if (composite[i]) continue;
    p.push(i);
    for (let j = i * i; j < composite.length; j += i) composite[j] = 1;
  }
  return p;
})();
function smallCount(x) {
  // All callers are within [0,2656]. This is an exact, bounded sieve.
  const n = Number(x);
  let lo = 0, hi = smallPrimes.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (smallPrimes[m] <= n) lo = m + 1;
    else hi = m;
  }
  return lo;
}
/* ---- li(x) with directed rounding ---------------------------------------------------------
   Values are integers near v * S, S = 10^Q. *Lo functions round every step down and drop
   positive tails (guaranteed lower bounds); *Hi functions round every step up and add an
   explicit tail bound (guaranteed upper bounds). li(x) = gamma + ln ln x + sum (ln x)^k/(k k!). */
const GAMMA_DIGITS = "5772156649015328606065120900824024310421593359399235988057672348848677267776646709369470632917467495146314472498070824809605040144865428362241739976449235362535003337429373377376739427925952582470949160087352039481656708532331517766115286211995015079847937450857057400299213547861466940296043254215190587755352673313992540129674205137541395491116851028079842348775872050384310939973613725530608893312676001724795378367592713515772261027349291394079843010341777177808815495706610750101619166334015227893586796549725203621287922655595366962817638879272680132431010476505963703947394957638906572967929601009015125195950922243501409349871228247949747195646976318506676129063811051824197444867836380861749"; // floor(Euler's gamma * 10^700), after "0."
const PI_LOW_NUM = 3141592653589793238462643383279502884197n, PI_LOW_DEN = 10n ** 39n; // below pi
const bitLen = (n) => n.toString(2).length;
function atanh2Lo(N, D, S) {               // 2 atanh(N/D) S, 0 <= N/D <= 1/3
  if (N === 0n) return 0n;
  const N2 = N * N, D2 = D * D;
  let term = (S * N) / D, sum = 0n;
  for (let j = 1n; term > 0n; j += 2n) { sum += term / j; term = (term * N2) / D2; }
  return 2n * sum;
}
function atanh2Hi(N, D, S) {               // tail after a term <= 1 is below 1/8
  if (N === 0n) return 0n;
  const N2 = N * N, D2 = D * D;
  let term = divCeil(S * N, D), sum = 0n;
  for (let j = 1n; ; j += 2n) {
    sum += divCeil(term, j);
    if (term <= 1n) { sum += 1n; break; }
    term = divCeil(term * N2, D2);
  }
  return 2n * sum;
}
const ln2Cache = new Map();
function ln2Bounds(S) {
  let v = ln2Cache.get(S);
  if (!v) { v = [atanh2Lo(1n, 3n, S), atanh2Hi(1n, 3n, S)]; ln2Cache.set(S, v); }
  return v;
}
function lnIntBounds(x, S) {               // ln x = b ln 2 + 2 atanh((x - 2^b)/(x + 2^b))
  const b = BigInt(bitLen(x) - 1), p = 1n << b, [l, h] = ln2Bounds(S);
  return [b * l + atanh2Lo(x - p, x + p, S), b * h + atanh2Hi(x - p, x + p, S)];
}
function lnFixed(T, S, hi) {               // ln(T/S) for T >= S
  let e = Math.max(bitLen(T) - bitLen(S), 0), p = S << BigInt(e);
  while (p > T) { e -= 1; p >>= 1n; }
  while ((p << 1n) <= T) { e += 1; p <<= 1n; }
  const l2 = ln2Bounds(S)[hi ? 1 : 0];
  return BigInt(e) * l2 + (hi ? atanh2Hi(T - p, T + p, S) : atanh2Lo(T - p, T + p, S));
}
function eiSeriesLo(T, S) {
  let a = S, sum = 0n;
  for (let k = 1n; ; k += 1n) {
    a = (a * T) / (S * k);
    const term = a / k;
    sum += term;
    if (term === 0n && k * S > 2n * T) return sum;
  }
}
function eiSeriesHi(T, S) {                // once k > 2 ln x each term is under half the last
  let a = S, sum = 0n;
  for (let k = 1n; ; k += 1n) {
    a = divCeil(a * T, S * k);
    sum += divCeil(a, k);
    if (a <= 1n && k * S > 2n * T) return sum + 1n;
  }
}
function liBounds(x) {                     // [lo, hi, S, Thi] with lo <= li(x) S <= hi, x >= 3
  const Q = Math.floor(x.toString().length / 2) + 60;
  if (Q > GAMMA_DIGITS.length) throw new RangeError("input too large for the stored constants");
  const S = 10n ** BigInt(Q);
  const [Tlo, Thi] = lnIntBounds(x, S);
  const G = BigInt(GAMMA_DIGITS.slice(0, Q));
  const lo = G + lnFixed(Tlo, S, false) + eiSeriesLo(Tlo, S);
  const hi = G + 1n + lnFixed(Thi, S, true) + eiSeriesHi(Thi, S);
  return [lo, hi, S, Thi];
}
function certifiedPi(x) {
  // pi(x) lies strictly inside li(x) +- sqrt(x) ln x / (8 pi) for x >= 2657: Schoenfeld 1976 under RH,
  // proved unconditionally up to 1.101e26 by Johnston 2022.
  if (x < COUNT_START) throw new RangeError("certifiedPi needs x >= 2657");
  const [lo, hi, S, Thi] = liBounds(x);
  const E = divCeil(ceilSqrt(x) * Thi * PI_LOW_DEN, 8n * PI_LOW_NUM);   // >= sqrt(x) ln x / (8 pi), units
  const piLo = (lo - E) / S + 1n;                                       // pi(x) > (lo - E) / S
  const piHi = divCeil(hi + E, S) - 1n;                                 // pi(x) < (hi + E) / S
  const a = piLo.toString(), b = piHi.toString();
  let same = 0;
  if (a.length === b.length) while (same < a.length && a[same] === b[same]) same++;
  return { lo: piLo, hi: piHi, digits: same, total: b.length };
}
function groupPrefix(prefix, total) {      // commas placed as they would be in the full number
  let out = "";
  for (let i = 0; i < prefix.length; i++) {
    if (i > 0 && (total - i) % 3 === 0) out += ",";
    out += prefix[i];
  }
  return out;
}
function bounds(x) {
  if (typeof x !== "bigint" || x < 0n || x > MAX_X) throw new RangeError("Invalid x");
  const u = x >= 4n ? gapWidth(x) : null;
  const above = u === null ? (x <= 2n ? [2n,2n] : [3n,3n]) : [x,x+u];
  const y = u === null ? -1n : x-u;
  let below = null, belowDirect = true;
  if (x >= 2n && y >= 4n) {
    // f is increasing on [4,∞), so f(y) < U(x) and y+f(y) < x.
    below = [y,x];
    belowDirect = false;
  } else if (x >= 2n) {
    const capped = x < COUNT_START ? x : COUNT_START - 1n;
    const p = BigInt(smallPrimes[smallCount(capped) - 1]);
    below = [p,p];
  }
  return {
    x, width:u, above, below, belowDirect,
    exactCount:x < COUNT_START ? smallCount(x) : null,
    radius:x < COUNT_START ? null : countRadius(x),
    finite:x >= COUNT_START && x <= JOHNSTON_END,
    sample:x === SAMPLE_X, binarySample:x === BINARY_SAMPLE_X
  };
}
function showBounds(doc) {
  const form = doc.getElementById("prime-bounds-form");
  if (!form) return;
  const el = (id) => doc.getElementById(id);
  const input = el("prime-bounds-x");
  const message = el("prime-bounds-message");
  const results = el("prime-bounds-results");
  const interval = (pair) => !pair ? "None" :
    pair[0] === pair[1] ? format(pair[0]) : "[" + format(pair[0]) + ", " + format(pair[1]) + "]";
  const format = (n) => n.toString().replace(/\B(?=(\d{3})+(?!\d))/g,",");
  function render() {
    let x;
    try { x = parseX(input.value); }
    catch (err) {
      input.setAttribute("aria-invalid","true");
      message.textContent = err.message;
      results.hidden = true;
      return;
    }
    const r = bounds(x);
    input.removeAttribute("aria-invalid");
    message.textContent = "Results for x = " + format(x) + ".";
    el("prime-bounds-above").textContent = interval(r.above);
    el("prime-bounds-above-note").textContent = r.width === null
      ? "A prime found directly in the small-integer table."
      : "Contains a prime. Window width: " + format(r.width) + ".";
    el("prime-bounds-below").textContent = interval(r.below);
    el("prime-bounds-below-note").textContent = !r.below
      ? "There is no prime at or below this value."
      : r.belowDirect ? "A prime found directly in the small-integer table."
      : "Contains a prime. The first endpoint is x minus the same width.";
    el("prime-bounds-count-label").textContent = r.exactCount === null
      ? "Prime-count error radius" : "Exact prime count";
    el("prime-bounds-count").textContent = r.exactCount === null ? format(r.radius) : format(r.exactCount);
    el("prime-bounds-count-note").textContent = r.exactCount !== null
      ? "Primes at or below x, counted with an exact small sieve."
      : (r.finite
        ? "Maximum error from li(x) under Johnston’s unconditional finite-range result."
        : "Maximum error from li(x) using Schoenfeld’s global bound and the verified RH.");
    el("prime-bounds-live-example").hidden = !r.sample;
    el("prime-bounds-binary-example").hidden = !r.binarySample;
    const piCard = el("prime-bounds-pi-card");
    if (piCard) {
      if (r.exactCount !== null) piCard.hidden = true;
      else {
        const c = certifiedPi(x);
        el("prime-bounds-pi").textContent = c.digits > 0
          ? groupPrefix(c.lo.toString().slice(0, c.digits), c.total) + "…" : interval([c.lo, c.hi]);
        el("prime-bounds-pi-note").textContent = "The first " + c.digits + " of its " + c.total +
          " digits are certain. " + (r.finite ? "Unconditional in this range (Johnston)."
            : "Holds because the Riemann Hypothesis is verified.");
        el("prime-bounds-pi-ends").textContent = format(c.lo) + " to " + format(c.hi);
        piCard.hidden = false;
      }
    }
    results.hidden = false;
  }
  form.addEventListener("submit", (e) => { e.preventDefault(); render(); });
  form.querySelectorAll("button[data-prime-x]").forEach((button) => {
    button.addEventListener("click", () => { input.value = button.dataset.primeX; render(); });
  });
  render();
}
if (typeof document !== "undefined") showBounds(document);
if (typeof module === "object" && module.exports)
  module.exports = {parseX,ceilSqrt,gapWidth,countRadius,bounds,certifiedPi,liBounds,lnIntBounds,
    groupPrefix,MAX_X,JOHNSTON_END,COUNT_START};
