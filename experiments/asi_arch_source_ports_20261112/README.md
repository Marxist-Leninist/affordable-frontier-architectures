# Prepared ASI-Arch source ports

These are **CPU correctness preparations, not newly trained quality results**. They add zero trained architectures to the [current coverage ledger](../gallery_cpu120_20261110/campaign_coverage_20261112/README.md). The current AGILLM trained source set remains #458 DLGM and #879 Fusion, with 103 canonical records still awaiting that training.

## Bundles

- [#259 BAMG_MemoryGate](259/README.md): practical Dirac-convolution adaptation, 31 correctness groups and five finite optimizer microsteps. The source's two separate -1 gate biases are both preserved, giving sigmoid(-2), not the prose's approximately 0.27.
- [#641 HeadWiseGateNet](641/README.md): two explicitly separate full-width variants, 31 CPU checks each. hwg641.py uses verified native FLA 0.2.2 random short-convolution initialization; hwg641_dirac.py matches the existing practical Dirac convention. Do not pool these variants.

- [#712 AdaptiveGateNet-AFP](712/README.md): full practical-Dirac adapter, 41 CPU checks and zero optimizer updates. Source gate/statistics/floor/temperature/FIR mathematics are retained; graph-bearing diagnostic state is replaced by explicit detached CPU diagnostics. External regularization expecting the original diagnostic attribute remains unsupported.

Each directory includes the frozen simulator and CPU helper, exact upstream program, dtype-adapted reference, tests, results, licenses and SHA256 manifest. The #641 bundle also includes pinned native FLA source/metadata and constructor-only proof of its initialization.

For the next practical full-width comparison, review bamg259 and hwg641_dirac against the actual deployed baselines, then freeze a new matched protocol. Preserve historical results and clearly label chunk-size, normalization, precision and initialization adaptations. No plugin is installed in the live queue by this publication.

## Limits

- No comparative GPU training, new trained checkpoints, cached decoding, fused-kernel parity or full-scale quality claim.
- Copied-weight comparisons establish CPU mixer correctness within the documented adaptation; they do not prove the whole original source model is unchanged.
- #259 adds 481,743,360 floating-point parameters at the stated 560-layer geometry; #641 adds 1,300,669,440 and #712 adds 1,545,064,080 across 420 mixer layers. Counts are not measured live GPU fit.
- #259 remains the first prepared candidate. #641 is an additional candidate, not a replacement or declared winner.
- Only admit new jobs after correctness, fixed source/data/config/seed manifests, safe capacity and a serving-compatible handoff. Current inference and existing jobs remain untouched.
- The wider campaign continues beyond these candidates. Preparation status must never be counted as trained coverage.

The candidate bundles have different correctness histories: #259 includes five tiny finite optimizer steps; #641 and #712 run forward/backward checks without optimizer updates. All still need sanctioned accelerator validation and the same fixed-budget empirical protocol.

The [public coverage page](https://zqx.lat/asi-arch/catalog-status/) accounts for every canonical source ID. STATUS.json gives machine-readable preparation status. Source hashes and numerical tolerances are recorded in the per-candidate reports.
