# Independent final report review

PASS: README and timing summary agree with the independently audited loss results and raw timing records. Recomputed all six timing groups (72 observations each), phase/total quantiles, median index allocations, invariant state/weight sizes, and all 144 matched constant/gate pairs. All 432 rows have complete matched variants and valid randomized order positions. Primary intervals agree with the independent audit.

README correctly withholds improvement claims against the historical local constant and for the GRU primary contrast. It does not claim speed superiority/equivalence, measured service throughput, frontier-quality parity, or commercial economics success. Shared-host and repeated-measure limitations, unadjusted comparisons, calibration flexibility, and exact-only dedup are disclosed. The historical cost assumptions are labeled inherited analytical scenarios rather than new measurements.

Chronology is described as reviewed execution order/hash integrity, not externally timestamped registration. Reproduction discloses omitted weights/corpora, predecessor training and prior snapshot dependencies, aliases for frozen paths, output overwrites, fail-closed source hashes, and required sequential correctness-before-timing order. The snapshot downloader preserves the existing manifest and checks it rather than silently refreezing it. Its viewer endpoint tracks the current dataset and fails if revision/content changes, so future availability of the exact snapshot is not guaranteed; this is a reproducibility availability limitation, not a correctness failure.

No scientific source, protocol, selection, score, timing, or summary files were modified. No heavy rerun was performed.
