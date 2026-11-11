# Recurrent-order NAT inference sweep

This is exploratory inference testing, not a validated search for the best loop count. All models were trained at r=2 only. Other counts are out-of-training-loop-distribution evaluations. Zero loops means no recurrent-middle blocks, not no model: 8 input and 8 output attention blocks still execute alongside embeddings, positional encoding, final normalization and output head. These outer attention blocks retain prefix conditioning. This independently implemented NAT motif is not an NVIDIA Dory replication. Each output is produced in one parallel decoding operation with recurrent hidden-state processing. More loops mean more internal passes, not diffusion denoising or feedback of predicted suffix tokens.

## Cohorts and coverage

All-101-count cohort: **{{WINDOWS}} windows per domain**, selected by a frozen runtime-only rule using synthetic inputs before held-out evaluation. Larger cohort: **128 windows/domain**, only at preregistered counts 0, 1, 2, 4, 8, 32 and 100. These are different cohorts; do not blend their scores. Domains are WikiText test and FineWeb held-out. Six checkpoints cover split/whole and seeds 1, 2 and 3.

All **{{CELLS}}** planned count/seed/domain/phase cells appear in cells.csv/json. Status totals: {{STATUSES}}. Missing, capped, nonfinite and malformed completed cells are retained with null metrics, never zero scores. Nonfinite status is instability, not ordinary missingness. Plots show all available seed traces, with gaps instead of interpolated missing results. Lines between measured counts are visual guides; the selected-count plots do not measure intervening counts. Summary means use completed seeds only; completed_seeds and complete identify partial groups. Seed SD is sample SD across fitted seeds, not a confidence interval or generalization guarantee. Partial means can be biased by missingness and must not be compared as if complete.

No best count is selected or called validated. Even the larger fixed cohort is held-out exploratory evidence with multiple comparisons. No document bootstrap is performed by this generator. Any future paired bootstrap requires matching arrays/groups and complete paired data, and would be conditional on fitted seeds, not replace seed uncertainty. No significance, convergence or scaling claim is made.

## Architecture and runtime

Split is A repeated r times then B repeated r times. Whole is (A,B) repeated r times. Both store 764,304 parameters across 36 unique attention blocks (8 input, 20 recurrent middle, 8 output), embeddings and head. Each uses width 48 and FFN hidden width 117. They apply 16 fixed outer layers plus 20r recurrent layers. Thus r=100 applies 2,016 total layers versus 56 at training. Active module parameters at r=0 are 353,424; at positive counts all 764,304 participate. Reinjection is omitted equally in both variants. This counts complete invoked module tensors, not individual indexed embedding entries. Nominal dense-block operation counts match between variants at a fixed count, not across counts. Actual runtime need not match.

Forward CPU/wall seconds are **sums over the evaluation cohort**, excluding metric computation, initialization, synthetic calibration and I/O. They are not robust request latency, warmed throughput or a standalone hardware benchmark. Global and per-checkpoint caps still apply. Runtime-only cohort prediction is a budget estimate, not a guarantee of full coverage.

## Provenance, licensing and reproducibility

Architectural motivation: [NVIDIA's official Dory model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/blob/main/README.md). This independent PyTorch NAT implementation derives from the prior NAT experiment; it does not use NVIDIA weights or implementation. Preprocessing uses pinned WikiText/FineWeb sources from that experiment. Exact data source identities/licenses remain in its data manifests. Viewing the NVIDIA model card does not grant a code or dataset redistribution license. Preserve the research repository's existing license and verify upstream code/data/checkpoint rights before publication. This report assigns no license and implies no NVIDIA endorsement.

After authorized execution, from the experiment directory:

    python reporting/report_sweep.py --input sweep_results --output sweep_report
    python reporting/test_report_sweep.py

The generator requires finalized status, runtime selection, cohort manifests and matching checkpoint hashes. provenance.json records raw JSON file hashes, checkpoint hashes and report source hashes. cells.csv/json carries cohort hashes. Training and inference reproduction use the separately frozen protocols and gated run_all.py/sweep.py; do not bypass their execution gates. The report generator imports only Python's standard library and never loads models, checkpoint tensors or dataset text.

Public-safe output whitelist: this report, aggregate cells/summary CSV and JSON, SVG plots, provenance.json and report_status.json. Outputs omit raw text, token arrays, per-window losses/groups, credentials, absolute local paths and arbitrary manifest metadata. Publishing these outputs or checkpoints is a separately authorized action; this generator uploads nothing. Exclude synthetic fixture reports from real research releases.

## Per-seed plots

{{PLOTS}}
