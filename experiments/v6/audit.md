# Independent review

A separate reviewer inspected the source and ran bounded checks on 2026-11-03. No files were changed during review and no correctness defect was found.

- Three unit tests passed.
- Independent prefix checks at lengths 1, 31, 32, 128 and 512 passed for attention and recurrent models.
- Existing v5 finite-difference gradient routine: attention maximum error 6.08e-11, recurrent 3.68e-11.
- Trained seed-1 model cached/reference BPC matched: attention 1.418654, recurrent 0.470650; probability errors at most 1.78e-15, including additional length-512 checks.

The review distinguishes numerical equivalence from capability quality; full-prefix recomputation from production cached serving; runtime-only lengths 128/512 from useful long-context validation; persistent buffer bytes from total process memory; pooled timing percentiles from fixed-context tail latency; synthetic held-out combinations from natural-language generalization; and float64 CPU behavior from low-precision/GPU behavior or cost.

The main experiment and unittest source reproduce the reported six-seed-pair quality/parity gates and cache tests; this note records separate manual review, not an additional stored test suite.
