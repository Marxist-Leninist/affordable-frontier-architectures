# Dory residual-scaling follow-up: partial result summary

## Publication status
This is a small summary of verified recorded outputs from a completed numerical study, not the full reproducibility package. The full machine-readable final evaluation and report package is temporarily unavailable following a workspace change. Model and full-state backups were verified separately; no weights, source files, per-window records, sampled generations or corpus text are included here. Earlier Dory studies are separate releases.

## Method
True single-pass non-autoregressive prediction: a 96-byte prefix and 32 independent output slots. Whole-loop topology at r=2; 764,304 parameters, 36 unique blocks and 56 applications. The intervention scales both residual branches by 1/sqrt(112), otherwise retaining the same initial weights and data per paired seed (1 and 2). Fixed AdamW: batch 8, learning rate 0.0005, warmup 100 updates, matrix decay 0.01, gradient clip 1.

Unchanged controls resumed their original 2,000-update states to avoid repeating training. Endpoints compare 4,000 TOTAL updates and, separately, 1,800 CUMULATIVE training CPU seconds. Historical control timing is included: this is not a contemporaneous speed benchmark.

## Fresh final evaluation cohort
FineWeb-Edu sample-10BT train rows 1700–1799, revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9; observed response revision matched. The 100 complete documents contributed their first 4,097 UTF-8 bytes, partitioned into nonoverlapping 128-byte windows. Exact known training, validation and prior-window exclusions and duplicate filtering were applied. There were 1,985 scored windows, four within-cohort duplicates removed, and no other exact exclusions. Freshness is relative to the inspected study history, not proof of global non-contamination.

- Cohort SHA256: 2f09c8cc383dcfbfe80748cdbb89cf0621823d6afa94f1d4ddb14633c4b7607a
- Response SHA256: ca75465c061b2fb469a17d01d03a2b9ee49e7a2434759231817a263d83146359
- [Dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/raw/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md): ODC-BY; underlying web-content rights remain. No corpus text is redistributed.

## Recorded results
Bits per byte (bpb), lower is better:

| Endpoint | Model | Seed 1 | Seed 2 | Mean |
|---|---|---:|---:|---:|
| 4,000 total updates | Unchanged | 4.610516548156738 | 4.612830638885498 | 4.611673593521118 |
| 4,000 total updates | Scaled | 4.607799053192139 | 4.603617191314697 | 4.605708122253418 |
| 1,800 cumulative CPU seconds | Unchanged | 4.604859828948975 | 4.608706951141357 | 4.606783390045166 |
| 1,800 cumulative CPU seconds | Scaled | 4.604604721069336 | 4.601738452911377 | 4.6031715869903564 |

Every one of the eight endpoints had byte accuracy 0.1588633507490158, exact 32-byte match rate 0, and greedy space fraction 1.0. The small likelihood improvement did not show useful generation improvement. Detailed sampled-output records are currently unavailable and no qualitative review is claimed.

Train-only controls evaluated on the SAME fresh cohort:
- Unigram: bpb 4.60633746376834; byte accuracy 0.15886335012594457.
- Factorized first-order Markov: bpb 4.587449316320217; byte accuracy 0.16218513853904282; space fraction 0.9733784634760705. It predicts independent marginals from the last observed prefix byte using 32 transition powers, never gold-suffix feedback. It beats every model endpoint here.

Development matched-step scaled-minus-control bpb was -0.0016188621520996094 (seed 1) and -0.004431247711181641 (seed 2); accuracy differences were zero. Both failed the prespecified -0.03 bpb threshold with at most 0.005 accuracy loss. There were no outcome-based seed, checkpoint or row changes.

## Completion and accounting
All four main fits and all eight final evaluations exited successfully. Originals remained unchanged; there were no GPU rentals or production changes.
- Main aggregate CPU: 5537.736173206 seconds; wall: 5530.064800478998 seconds.
- Final acquisition and scoring aggregate CPU: 437.39378052 seconds; wall: 443.9229900289938 seconds.
- Preparation was conservatively debited 150 CPU / 600 wall seconds, not measured as a total.
- Numerical allocation: 7,200 new CPU / 9,000 active-command wall seconds.

Report generation encountered a validation issue reconciling a small non-fit child CPU baseline. The arithmetic was independently resolved, but the source amendment and complete report attachment remain unavailable after the workspace change. This was not a failed numerical study.

## Limits and decision
Two seeds, a tiny model, a narrow NAT byte objective and modest training do not establish intelligence, frontier-scale performance, transfer to a 600B model or monetary profit. No live model change is justified. The 30% net-revenue margin, mixed-workload, useful-quality and serving-SLO gates remain unproven. No CPU-to-GPU-dollar conversion is made.
