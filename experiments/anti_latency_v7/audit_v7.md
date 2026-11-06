# Independent read-only final audit

No blocking publication issue found. Reviewer independently recomputed all24 quality means, paired deltas/ranges, per-window aggregates, timing medians and throughputs. Reports agree with raw results. All18 rows have200 updates,409,600 targets,200 loss/timing records and zero reconstruction weight error.

Corrected target accounting reconciles to12,288,000; disposable profiles add442,368. Invalid pilot's2,048,000 confirmed lower bound and unmeasured possible partial reconstruction are disclosed. Wall315.75796s, reconstruction113.10327s and optimizer183.42958s agree; component times are correctly partial.

Deep-copy isolation is present in training and profiling. Unit regression checks source/other-arm immutability. Validation enforces initial Adam step800 and final1000,18 optimized/cache checks and six historical aggregate-loss replays. Exact historical replay here means the four heldout aggregate bpb values per model/seed, not every training-loss trajectory.

Timing labels match code: batch4,5 warmups+31 randomized measurements; one next-token decode excludes prefix construction. Signed projection conversion is correct. Parameter count131,904 and conditional hardware-only economic floor are correct. No frontier/GPU/profitability or strict quality-preservation claim is established.

Reviewer used lightweight JSON arithmetic and source inspection, not competing torch benchmarks. The run owner executed tensor/optimizer/cache tests. The invalid pilot's aliasing defect was caught by this review before conclusions, corrected and fully rerun.
