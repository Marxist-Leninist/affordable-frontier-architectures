# Fresh-text comparison of three saved seed-2 checkpoints

## Result

On the same prospectively frozen 50-document FineWeb-Edu cohort, #259 has lower next-byte negative log-likelihood than #458/DLGM and T96. Lower is better:

| Saved checkpoint | Parameters | Mean NLL, nats/byte |
|---|---:|---:|
| #259 |349,932|2.126158087210845|
| #458/DLGM |349,944|2.217316312449732|
| T96 Transformer |349,920|2.242369545177342|

The primary paired #259 minus #458 difference is −0.09115822523888728 nats/byte; #259 wins 43/50 documents. The fixed 2,000-draw paired document-bootstrap interval is [−0.11639361660789788, −0.06476967651019286].

Secondary descriptive comparisons: #259 minus T96 is −0.11621145796649734, interval [−0.13925478185821638, −0.09107836068534453], 45/50 wins. #458 minus T96 is −0.025053232727610078, interval [−0.05718210310592248, 0.007902246830629883], 30/50 wins. Pairwise intervals are unadjusted and conditional on these fixed models and this cohort, not training-seed confidence intervals.

## Frozen method

Use the public saved checkpoints and unchanged model/gallery source at [commit 295dddd](https://github.com/Marxist-Leninist/affordable-frontier-architectures/tree/295dddddbbdb27c68c6c668d95b67d6c985a44da/experiments/gallery_cpu120_20261110). Exact checkpoint links, source identities and hashes are in provenance.json. No training or checkpoint selection by new results occurred.

[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), sample-10BT/train: the first 50 eligible distinct documents from rows 10000–10099, selected 10000–10050 excluding short row 10014. The selection rule was frozen before acquisition; the exact cohort was frozen before any model scores. Each contributes its unnormalized 513-byte UTF-8 prefix, input bytes [0:512], scored logits [384:512] against original bytes [385:513]: 128 targets/document, 6,400/model/pass. Batch 4 with final batch 2; two complete passes per model. Repeated per-document losses match exactly.

Keep original FP32 stored weights and CPU BF16 autocast, including #259’s explicit BF16 inverse intermediate. T96 uses explicitly disclosed math SDPA with FP32 intermediates instead of the original automatically selected flash backend. Cross-entropy uses FP64 logits. This is mixed native arithmetic, not an all-BF16 test or exact rerun of the original backend recipe.

The primary comparison and all secondary analyses were fixed before scoring. Means use equal document weights. Bootstrap: Python Random(20261111), 2,000 paired 50-document resamples with replacement, sorted zero-based endpoints 49 and 1949. metrics.json supplies all per-document NLLs, hashes and paired deltas for arithmetic verification.

## Validation and resources

All three models pass finite-output checks, 13 prespecified causal suffix cuts, same-shape batch independence, exact probe repetition, exact repeated full scoring, and unchanged restored checkpoint/runtime checks. Independent log reconciliation covers both batch sizes and every scored-pass matrix/convolution shape: 14,898 BF16 AVX512 matmuls, 624 BF16 AVX512 depthwise convolutions and 156 native FP32 oneMKL SGEMM_BATCH calls. FP32 logs establish CPU GEMM dispatch by serial call mapping, not a specific unlogged ISA mnemonic.

The completed corrected probe and scoring command intervals are 11.134 seconds and 11.782 seconds, respectively, including initialization/cleanup and command transport. The scoring numerical section is 5.994 seconds across all models and analysis. Maximum sampled child-plus-supervisor RSS is 428,441,600 bytes under a 2 GiB combined hard address-space cap. These are aggregate evaluation measurements, not per-model throughput, training cost or economic results. An initial compatibility probe stopped on restricted temporary-file creation during installed dependency initialization; a narrowly bounded correction was frozen before scoring. That failed attempt is preserved and is not part of the two completed-command intervals.

## Scope and reproducibility limits

This supports #259 over #458 for these saved seed-2 models on this frozen cohort. It does not establish general architecture superiority, seed robustness, serving/cache correctness, useful-scale/GPU performance or price/performance. The original checkpoints received 496, 451 and 709 updates, respectively; this is not equal-token training.

The 50 documents are a fixed nonrandom prefix-based selection. Exact overlap checks against inspected prior evidence and verified training bytes passed, but global/semantic freshness, partial overlap and uninspected pretraining overlap are not certified. The source response’s observed revision matched the Hub, but the rows query was not revision-pinned; future live responses may drift. Exact retained bytes and hashes define this cohort.

Original v44 loss reload is NOT RUN/UNVERIFIED because its exact arrays were absent. The original report’s up-to 0.00006 AMX-versus-capped-AVX512 loss movement and 1e-6 verifier remain unchanged; neither is a new local failure or pass. The separate recovered-cohort HOLD for T96 flash internal GEMMs and convolution weight-gradient ISA evidence remains unresolved by this forward-only math-backend test.

The exact tested harness and evidence are locally preserved and environment-bound. This small report package is not a validated portable fresh-checkout harness or a self-contained exact-data rerun. It contains no corpus text, response snapshots, checkpoint payloads or private filesystem paths. FineWeb-Edu attribution, [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) and [CommonCrawl terms](https://commoncrawl.org/terms-of-use) are recorded; no rights or repository licenses are changed.
