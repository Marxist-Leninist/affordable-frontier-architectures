# Longer single-pass NAT comparison: 16,000 updates, three seeds

This is a distinct follow-up to the 800-update pilot, not a replacement or a controlled attribution of its differences. The Transformer has 765,568 parameters and the optimized bidirectional Mamba-1 has 765,184. Both predict 32 blank suffix bytes in one pass from a 96-byte prefix. No diffusion, autoregressive suffix feedback, GPU performance, general intelligence or 600B-scale improvement is claimed.

Read [the requested post-hoc composite](nat_long_20261111/COMPOSITE_INDEX.md), [the full report](nat_long_20261111/report/REPORT.md), [protocol](nat_long_20261111/protocol_long.json), [summary](nat_long_20261111/summary_long.json), [standalone timing](nat_long_20261111/standalone_timings.json), and [checkpoint manifest](CHECKPOINT_MANIFEST.json). Report figures include every fitted seed. Three seeds and conditional document bootstrap intervals do not establish a universal architecture ranking.

## Reproduction

Use Linux x86-64, Python 3.12, a C++17 compiler and the official PyTorch CPU build (2.6.0+cpu in the measured environment), plus the versions in requirements.txt. The recurrence shared library is built locally from scan_cpu.cpp; no compiled binary is distributed. CPU affinity IDs 1 and 2 must exist. Adapt affinity to available CPUs if necessary and disclose the changed hardware/placement for new timings.

1. In frontier_research_v8, run `python prepare_data.py`. In frontier_research_v31, run its `python prepare_data.py`. These retrieve licensed source data and verify pinned hashes/revision. FineWeb viewer exact retrieval depends on that snapshot remaining available and fails closed on changes. No corpus bytes are included here. See THIRD_PARTY_NOTICES.md for license caveats.
2. To reproduce from scratch, copy source, protocol and dataset scripts into a fresh directory with the same sibling layout, without copying completed runs, evaluation outputs or status receipts. In nat_long_20261111, run `python orchestrate_long.py`. It caps its owned training jobs and refuses existing outputs; do not run it against the archived completed run.
3. After all fits finish, run `python evaluate_long.py`, then `python benchmark_long.py`, then `python summarize_long.py` in nat_long_20261111. Each measured phase must run alone. These commands refuse duplicate completed outputs. The complete-model scan upgrade is an additional bounded diagnostic, not the main Transformer/Mamba comparison.
4. To regenerate the published report from included JSON without loading models or corpora, run `python nat_long_20261111/generate_report_long.py`. Raw-prefix examples were removed from public evaluation JSON, and report input hashes are recomputed from those exact sanitized files. Numerical measurements are unchanged; SANITIZATION.json records the transformation.

## Checkpoints and scope

The 18 listed checkpoints are the six newly trained small research runs at matched data, the first whole update crossing 900 training CPU seconds, and separate 2,000-update synthetic-task adaptation. They include optimizer and RNG state and are unchanged from the measured runs. The optional 300-second snapshots and tuning weights are omitted. Load with `torch.load(path, map_location="cpu", weights_only=True)` and use `models_long.NAT(kind)` with its `model` state dictionary. Keep ordinary PyTorch security precautions for downloaded files.

The primary training objective is text-only. Synthetic tasks are separately adapted, not zero-shot intelligence tests. Loss is conditional suffix cross-entropy in bits/byte, not autoregressive full-text perplexity. Dataset selection, target overlaps, learning-rate-grid boundary, hardware timing limits, partial compute estimates and exact resource accounting are in the report. No rentals, purchases or live trainer changes were performed.

The repository's existing license and permissions are unchanged. Third-party dataset/source rights remain as described in the notices. Whether checkpoints appear in each hosting service depends on its current storage quota; a file manifest alone is not a publication receipt.
