# ASI-Arch coverage: what has actually been trained

Snapshot: 12 November 2026, approximately 04:53 UTC. This report adds coverage accounting and a continuation plan, not new quality results.

## The counts refer to different experiments

| Evidence | Coverage | What it establishes |
|---|---:|---|
| Earlier original-gallery pilot, reported 8–9 November | 22 of 105 at that time | A historical progress report from small CPU Wiki/FineWeb experiments. |
| Later original-gallery CPU campaign | 105 source IDs, 146 runs including controls/confirmation | AR-only byte-model screening at approximately 350k parameters and 120 CPU-seconds per primary run. Not the current AGILLM multiobjective test. |
| Original-module GPU operator checks | 105 of 105 passed, zero training steps | Shape, causality and finite-gradient compatibility. Not trained quality. |
| Current AGILLM successful fixed-budget records | 89 eligible runs, 32 plugin/control variants | Includes local modifications, controls and repeated seeds. This is not 32 distinct upstream architectures. |
| Upstream sources with eligible AGILLM adapted-port training | **2: #458 DLGM and #879 Fusion** | Small binary-backbone simulator results with full/LR/LRK/HN adaptations. |
| Canonical upstream sources without that training | **103 remaining** | Every remaining ID is enumerated in [COVERAGE.json](COVERAGE.json). #259 has correctness preparation, but no qualifying training run. |

The historical “22 tested” statement was a real earlier progress update. The later ledger records 25 “previously screened” IDs, so it cannot recover the exact membership of the older 22 without an earlier snapshot. No three records were arbitrarily removed to manufacture that list.

The source export named 106.json actually contains 105 unique IDs, names and program hashes at the pinned revision. Earlier upstream revisions contained 106; removal of #880 SpectralContextMixer produced 105. Paper v2 describes 105. No separate authoritative 103-total ID list has been located. Here, “103 remaining” means exactly the set of 105 canonical records minus the two trained source IDs. It is an explicit calculation, not a claim that a supplied 103-entry list was found.

## Auditable counting

[COVERAGE.json](COVERAGE.json) contains each ID, title, exact source hash, historical CPU status, current AGILLM status, variants/seeds and run evidence. [UPSTREAM_105.json](UPSTREAM_105.json) preserves the pinned public catalog bytes. The source revision is 3113c51b04a1d4a7520a3b0238c6a08d3194d58f, catalog SHA256 aa77e8289971b3922102f000fddf44a6a9a31ca1f3b59c834722ac130f63759c.

The included run_snapshots directory contains allowlisted classification fields from the observed records. It omits host names, paths and unrelated fields, and retains each original record's SHA256. Classification requires success, 1,200 updates, 2,457,600 token targets, finite AR/SAT/NAT at that endpoint, a passing causality check, zero skipped updates, the recorded simulator hash and the documented geometry. Metadata, zero-step checks, failed runs and incomplete endpoints do not count.

Plugin-to-source mapping is explicit and reviewed. More variants or seeds never increase the distinct-source count. A prepared adapter never counts as trained. The recorded simulator hashes are only 16-character prefixes, and the old records did not store per-run plugin or data hashes. Current source hashes cannot retroactively prove which bytes a historical run used.

Rebuild the accounting from the included input snapshot, starting in experiments/gallery_cpu120_20261110:

```sh
python3 code/campaign_coverage.py \
  campaign_coverage_20261112/UPSTREAM_105.json \
  campaign_coverage_20261112/run_snapshots METRICS.json \
  /tmp/rebuilt_coverage.json
cd code
python3 -m unittest test_campaign_coverage test_reproduction -v
```

Rebuilding changes the observation timestamp, not the source membership or counts. Thirteen regression tests cover accounting and the previous reproduction repairs. These tests do not execute models.

## Continue all remaining sources, not only finalists

The [static continuation plan](CONTINUATION_PLAN.json) enumerates all source hashes and flags source-specific schedule/auxiliary-loss features. All 105 original modules expose DeltaNet and q/k/v/o projections, but an unchanged wrapper would retain floating-point projections and its own output normalization. That is not equivalent to the current binary AGILLM ports and could double-apply output processing.

Use common loading/testing scaffolding with explicit source-specific mixer adapters. Preserve defining equations, initialization, retention, statistic axes and schedules; document every projection, normalization, dtype or auxiliary-loss adaptation.

The first bounded preparation batch is:
1. Calibrate the reference harness against #458 and #879.
2. Reuse prepared #259 BAMG_MemoryGate; its tiny correctness and five finite microsteps are not a quality benchmark.
3. Prepare #641 HeadWiseGateNet, then #712 AdaptiveGateNet-AFP.
4. Admit a first 1,200-update seed for each new source only after correctness, memory and serving-safety gates.
5. Continue the remaining source IDs in bounded batches, recording unsupported cases, failed tests, OOMs and timeouts instead of hiding them. Complete a fixed three-seed pilot for every eligible source before interpreting pilot rankings.

#259 is a prepared, inexpensive short-budget hypothesis, not the paper favorite. Its original paper merged rank is 93/105; #641's is 13/105. Historical AR-only CPU loss favors #259 in point estimates, but its paired FineWeb seed confidence interval against DLGM includes zero. Current AGILLM NAT/SAT quality is unknown for both candidates.

Direct source allocation formulas give #641 gate parameters 4Hd + 12d + 4H and #712 gate/floor/temperature parameters 4Hd + 48d + 8H + 4. At d=8192 and H=64 these are 2,195,712 and 2,490,884 per mixer layer. They are gate-only counts, not full-model memory, latency or proof of GPU fit.

## Freeze the next comparison before launch

The next cohort must be a separate version, not a silent rewrite of historical results:
- Fixed primary metric: 0.6 AR + 0.2 SAT + 0.2 NAT; publish every component and diagnostics.
- Match objective schedule, data windows, token budget, shape, shared-weight initialization, seed schedule and evaluation masks across compared arms. Freeze the exact launch manifest and all source/config/data hashes before admitting a job.
- Keep full, low-rank, smaller-rank and head-normalized adaptations separate. Preserve archived v1. Current DLGM and Fusion ports use different delta chunk sizes; any validated common-size implementation is a new adaptation version and needs matched reruns.
- Use the existing token arrays only as an explicitly reused dataset. The old preparation code did not preserve the token-remapping LUT or dataset revisions. Newly collected text cannot be called a fresh equivalent holdout without a pinned, saved vocabulary map and a newly versioned dataset for every arm.
- Three paired seeds are a pilot, not decisive evidence. Before a winner claim, register a fixed additional confirmation cohort and adjusted paired comparisons across all selected arms. Do not substitute document bootstrap for independent seed uncertainty or stop as soon as a favorable result appears.
- Measure runtime and peak memory in matched execution windows; publish interruptions and hardware. Do not treat the earlier bytes/time preference heuristic as measured compute cost.
- Keep the historical composite index intact. Unrelated cache experiments must not give one new arm a serving bonus: current ports do not yet have validated cached decoding. A new common-evidence methodology must not renormalize weights as partial results arrive.

## Operational status

No new benchmark was launched for this update. Existing HN and learned-gate jobs retain their coordinated lane. The serving GPU was busy with less than 800 MiB free at the latest check. Admission must fail closed on unavailable/inconsistent health, and needs real capacity coordination; idle polling alone cannot prevent an inference race. A paused process retains GPU memory.

The next job remains pending safe admission and an orderly runner handoff. No second runner, live trainer interruption, queue takeover or paid resource was introduced. Correctness preparation can continue independently.

This small adapted simulator cannot establish a full-scale 560-layer winner. The [existing comparison](https://zqx.lat/asi-arch/dlgm-vs-fusion/) and [public coverage page](https://zqx.lat/asi-arch/catalog-status/) retain that distinction.
