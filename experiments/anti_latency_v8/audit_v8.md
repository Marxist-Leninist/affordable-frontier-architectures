# Independent read-only audit

No blocking issue found. Reviewer independently reproduced all18 mean scores and21 contrasts, per-document and per-seed deltas, bootstrap intervals and document favor counts exactly. Timing medians, byte-throughput calculations and conditional3968.254 output-only economic floor match. All four primary mixed-minus-long contrasts have consistent sign across all three seeds.

All54 scoring rows contain100 document losses. All input hashes match18 distinct local v7 checkpoints and the manifest published with v7 before v8. Model source is unchanged from v7. Target alignment correctly scores the same128 bytes in every condition. Input-window sizes are not uniform per-target history lengths; report explicitly states ranges1–128,129–256,385–512.

Review prompted a supplemental check against the actual deduplicated2M training stream, not just raw WikiText; this passed with zero target matches. Current cohort has no repeated target spans or first513-byte windows. Exact checks are not near-duplicate decontamination. Loader now fails closed on the prior published checkpoint manifest, and dataset preparation verifies the entire saved provenance receipt including prior snapshot list. Hardened paths were validated without changing scores.

Run-owner tensor tests cover18 full/optimized/cache/causality comparisons and context-target alignment. Reviewer did only lightweight JSON/source/hash inspection, not competing inference benchmarks. Document-bootstrap uncertainty is conditional on this fixed cohort and three trained seeds; no multiplicity adjustment or general frontier/financial guarantee is implied.
