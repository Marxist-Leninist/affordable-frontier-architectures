// Tests for prime-bounds-v3.js. Run: python browser/tests/make_fixture.py && node browser/tests/test_v3.js
const A = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const M = require(path.resolve(process.argv[2] || path.join(__dirname, "..", "prime-bounds-v3.js")));
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "li_fixture_v3.json"), "utf8"));

// ---- openai-mcp's v2 test.js, unchanged except where v3 widens the accepted input range ----
A.equal(M.parseX('10^100'),10n**100n);
A.equal(M.parseX(' 10 ^ 12 '),1000000000000n);
A.equal(M.parseX('1000000000000'),1000000000000n);
for (const x of ['','-1','1e12','10^1001','10^-2','9'.repeat(1002),'9'.repeat(1009),'<a>']) A.throws(()=>M.parseX(x),RangeError); // v3 range
for (let i=0n;i<20000n;i++){const s = M.ceilSqrt(i);A(s*s>=i);if (s)A((s-1n)**2n<i);}
const isPrime = new Uint8Array(100001).fill(1);
isPrime[0]=isPrime[1]=0;
for(let p=2;p*p<isPrime.length;p++)if(isPrime[p])for(let j=p*p;j<isPrime.length;j+=p)isPrime[j]=0;
const sum = Array.from(isPrime, (_,i)=>i);sum[0]=0;
for(let i=1;i<sum.length;i++)sum[i]=sum[i-1]+isPrime[i];
const inRange=([a,b])=>sum[Number(b)]-(a>0n?sum[Number(a-1n)]:0)>0;
for(let i=0n;i<=10000n;i++){
 const r=M.bounds(i);
 A(r.above[0]>=i);A(inRange(r.above));
 if(i>=2n){A(r.below[1]<=i);A(inRange(r.below));}
 else A.equal(r.below,null);
 if(i<2657n) A.equal(r.exactCount,sum[Number(i)]);
}
const x=10n**12n;
const mid=M.bounds(x);
A.equal(mid.width,24398810n);
A.equal(mid.radius,1109037n);
A(mid.above[1]>=1000000000039n);
A(mid.finite);A(!M.bounds(M.JOHNSTON_END+1n).finite);
A(M.bounds(M.JOHNSTON_END).finite);
const hi=M.bounds(M.MAX_X);
A.equal(hi.above[1]-hi.x,hi.width);
A.equal(hi.x-hi.below[0],hi.width);
A(hi.radius>10n**50n);
const bigBinary = 2n**256n;
A.equal(M.parseX('2^256'),bigBinary);
A.equal(M.parseX('2 ^ 332'),2n**332n);
A.equal(M.parseX(bigBinary.toString()),bigBinary);
A.equal(M.parseX('2^3321'),2n**3321n);                 // v3: was rejected above 2^332
A.throws(()=>M.parseX('2^3322'),RangeError);
A.throws(()=>M.parseX('2^-1'),RangeError);
A(M.bounds(bigBinary).binarySample);
A(!M.bounds(bigBinary+1n).binarySample);
A(!M.bounds(bigBinary).finite);
A.equal(M.ceilSqrt(bigBinary),2n**128n);

// ---- v3: certified pi(x) ----
const PI10 = ["4","25","168","1229","9592","78498","664579","5761455","50847534","455052511","4118054813",
  "37607912018","346065536839","3204941750802","29844570422669","279238341033925","2623557157654233",
  "24739954287740860","234057667276344607","2220819602560918840","21127269486018731928","201467286689315906290",
  "1925320391606803968923","18435599767349200867866","176846309399143769411680","1699246750872437141327603",
  "16352460426841680446427399","157589269275973410412739598","1520698109714272166094258063"];
for (let k = 4; k <= 29; k++) {
  const c = M.certifiedPi(10n ** BigInt(k)), v = BigInt(PI10[k - 1]);
  A(c.lo <= v && v <= c.hi, "pi(10^" + k + ")");
}
const t12 = M.certifiedPi(10n ** 12n);
A.equal(t12.lo, 37606850878n); A.equal(t12.hi, 37609049684n);   // li - E = ...877.41, li + E = ...684.20
for (let n = 2657; n <= 100000; n += (n < 6000 ? 1 : 29)) {
  const c = M.certifiedPi(BigInt(n)), v = BigInt(sum[n]);
  A(c.lo <= v && v <= c.hi, "pi(" + n + ")");
}
// same algorithm as tools/rh_unlocked.py: identical integers at every fixture input
let n = 0;
for (const cs of fixture.cases) {
  const xv = BigInt(cs.x);
  const [lo, hiB, S] = M.liBounds(xv);
  A.equal(S, 10n ** BigInt(cs.Q), "Q at " + cs.x.slice(0, 20));
  A.equal(lo, BigInt(cs.li_lo), "li_lo at " + cs.x.slice(0, 20));
  A.equal(hiB, BigInt(cs.li_hi), "li_hi at " + cs.x.slice(0, 20));
  const li0 = BigInt(cs.li_floor) * S;                             // mpmath: li_floor <= li < li_floor + 1
  A(lo < li0 + S && hiB >= li0, "brackets mpmath at " + cs.x.slice(0, 20));
  const c = M.certifiedPi(xv);
  A.equal(c.lo, BigInt(cs.pi_lo)); A.equal(c.hi, BigInt(cs.pi_hi));
  n++;
}
A.equal(M.groupPrefix("436197", 98), "43,619,7");
A.equal(M.groupPrefix("1234567", 7), "1,234,567");
const t0 = Date.now(), big = M.certifiedPi(10n ** 1000n), ms = Date.now() - t0;
A(big.digits >= 490 && ms < 5000);
console.log("v3 ok: v2 checks (range-adapted), pi(10^k) k<=29, pi(x) for 2657<=x<=1e5, " + n +
  " inputs identical to the Python toolkit, 10^1000 in " + ms + " ms with " + big.digits + " certified digits");
