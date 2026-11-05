# v22: dynamic INT8 at matched causal inference shapes

Bounded isolated CPU study, 2026-11-05. Frozen v19 models, reused exploratory v21 FineWeb-Edu cohort, no training, calibration search, new downloads, paid compute, production edits, or corpus/weight redistribution.

## Findings

Actual serialized state-dictionary size shrinks **2.73x** for full/local Transformers (533,155 →195,489 bytes) and **2.71x** for GRU (535,127 →197,101 bytes). This is smaller model storage, not an equivalent reduction in total serving memory. After the 32 updates, FP32 and INT8 persistent neural states are identical in size: full Transformer 2,129,920 bytes; reset512 557,056; local 130,064; GRU 512. The separate compact copy index is unchanged.

INT8 produces a small loss increase in every architecture's three-seed average:

| Model | Neural FP32 →INT8 bpb | Hybrid FP32 →INT8 bpb | Neural total p50/p95 ms, FP32 →INT8 |
|---|---|---|---|
| Full2048 Transformer |4.229754 →4.240161|3.559573 →3.566223|84.10/90.93 →95.37/102.26|
| Reset512 Transformer |3.615053 →3.625744|3.155613 →3.162625|31.76/34.22 →44.19/46.97|
| Local Transformer |3.028825 →3.035316|2.715874 →2.721234|29.46/32.91 →42.79/47.82|
| Gated GRU |2.956994 →2.958939|2.684393 →2.685932|94.12/99.18 →88.52/92.26|

Hybrid total p50/p95 milliseconds are respectively 92.82/97.93 →103.88/115.84, 40.25/45.34 →52.68/56.97, 37.67/39.84 →51.18/57.25, and 102.20/106.36 →97.28/100.91. There are 72 retained matched samples per condition, 1,152 total; all phase times and samples are retained. Times should be compared **within this run**, not against previous experiments on a shared host.

Every INT8-versus-FP32 aggregate difference is smaller than the fixed 0.02-bpb descriptive screen, but positive paired document-bootstrap intervals indicate consistent small losses. This is **not exact equivalence**. Post-primary bootstrap summaries average checkpoint seeds before 5,000 document-cluster resamples and remain conditional on this reused convenience cohort. The local and GRU hybrids still differ by >0.02 bpb; no equal-quality cross-architecture winner follows. The full Transformer is still inadequately trained at these contexts.

The bounded result is that quantization improves storage but is not automatically a speed optimization: all three tested Transformer paths are slower; GRU is modestly faster. No large production throughput or cost gain is established. A dynamic-int8 CPU GRU is a candidate for further quality and serving tests, not a deployment recommendation.

Five numerical tests passed. Artifact reconstruction validates all 24 condition/seed aggregate rows exactly, source hashes, actual qint8 tensors, and complete matched timing sets. Primary scoring used 283.14 CPU seconds, timing 109.03, and the two initial test passes together about 5.06; one failed pre-result attempt took less than a minute of wall time. Summaries/audits are additional low-cost processing. Total new work remains well below the 900 CPU-second limit. Independent external review, if performed by the parent, is documented separately.

## Protocol and interpretability

Dynamic qint8 quantization targets PyTorch Linear and GRU modules, using the x86 engine. Embeddings and RMSNorm parameters, floating-point activations, softmax, and cached KV/recurrent state remain floating point. Copy-memory predictions and mixtures use float64. Backend-packed weights are real qint8 tensors, not assumed 8x compression. `model_memory.json` lists each unpacked weight's dtype, element count, payload bytes and quantization parameters, plus actual serialized state-dictionary size. This is not backend resident packed allocation or peak process RAM; packed GRU FP32 biases are reflected in serialized size but not the logical weight-only total.

Dynamic activation scale can depend on other tokens in the same call. Therefore quality is measured causally: each 2,048-input window first consumes 1,921 observed bytes and predicts one next-byte distribution, then consumes 127 observed updates. No later scored target is present when its earlier probability is computed. Every update uses batch one; FP32 has the same call shapes. The reset512 full-attention diagnostic starts at input index 1,536, so scored predictions see 385–512 bytes. Full Transformer uses the entire input. Local uses tested structural pruning of unreachable prefix for the initial prediction; this is not asserted numerically identical to an unpruned dynamically quantized implementation, since activation scales differ. It is the specified deployed-path candidate. GRU consumes the complete prefix.

Frozen mixture weights from v20 WikiText validation are full Transformer .6, reset512 Transformer .5, local .3 and GRU .2. Compact full-history exact-copy memory is unchanged. No fitting or parameter selection uses the reused evaluation cohort.

One pre-result plumbing failure was retained (`initial_failure.log`, `run_initial_failed.py`, `protocol_initial.json`): the full Transformer selection key was incorrectly named. It failed before printing or retaining any quality result. The existing `transformer_full2048` key was then used, a selection-key test was added, and hashes were re-frozen before restarting. No weight or protocol choice changed. The initial attempt's process CPU was not instrumented separately; its wall interval was under one minute. All new scoring, tests, timing and failed work remain bounded under 900 CPU seconds.

## Data and prior costs

HuggingFaceFW/fineweb-edu, sample-10BT, train, pinned revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, viewer rows 600–699, first 4,097 UTF-8 bytes per document, no normalization, joins or content-based filtering. This is the **same** snapshot examined in v21, explicitly an exploratory engineering comparison rather than fresh confirmation. All 91 nonoverlapping windows from 58 eligible documents contribute their final 128 byte targets, 11,648 targets/model/seed. Byte vocabulary 0–255; seeds 1,2,3 share data and are not corpus replications. Dataset card and provenance/license verification are inherited from v21, with raw-response and document hashes rechecked locally. ODC-BY dataset license and underlying content rights apply. [Pinned card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md).

v21 found no exact full-document duplicates against v15–v20; this does not rule out near-duplicates, training overlap or repeated testing bias. Checkpoint hashes are checked on every load. Original WikiText training and v19 adaptation, previous failed experiments and current evaluation are all research costs, not free inference gains.

## Timing

After numerical tests and scoring, all four architectures are tested with FP32/INT8 and standalone/compact-copy variants. Eight fixed eligible documents, three frozen checkpoint seeds, four repetitions with first discarded, randomized four-variant order inside each seed/document/repetition. Each probe computes 33 complete 256-way distributions: 2,048-byte prefix then 32 observed updates (reset diagnostic uses the last512 prefix bytes). Includes neural prefill, updates, softmax and all compact-copy construction/query/mixing. Excludes model loading, input tensor construction, memory traversal and final cache destruction equally. No manual garbage collection, threads fixed to one. No host exclusivity claim. This is observed-byte inference, not sampled generation, service p95, concurrent serving or commercial token throughput.

## Reproduce

Prepare sibling `frontier_research_v19/checkpoints` and `frontier_research_v21` using their pinned source/data instructions; never commit their weights or `rows.json`. Install pinned requirements in an isolated environment. In this scratch environment dependencies are at `/tmp/frontier-v10-packages:/tmp/frontier-v8-packages` via PYTHONPATH.

```
python freeze.py
python -m unittest -v test_quant
python run.py
python summarize.py
python benchmark.py
```

The archive excludes corpus text, model weights and Python caches. `probabilities.npz` contains target probabilities, not corpus text. Protocol and artifact manifests record exact source/input hashes. Final numerical tests check frozen input identity, selection keys, qint8 representation, causal future-suffix invariance, distribution normalization and FP32 agreement with historical v21 scoring. Tests do not establish general model capability or exact FP32/INT8 equivalence.

## Economic boundaries

No pricing source was newly accessed by this isolated experiment. Parent research reporting handles the required current primary-source recheck. Preserve requested targets: **US$0.05/M uncached input, $0.001/M cached input, $0.10/M billable output**, and **30% margin on net revenue**. The conditional output-only hardware floor is 3,968.25 billable tokens/s per $1/hour at 100% billable productive utilization, divided by utilization otherwise. It excludes all other costs and is not a universal mixed-workload gate.

The separate <=US$500/month subscription must preserve the same useful average and heavy full-month workloads, capabilities and latency, while total attributable cost is at most 70% of net revenue. GBP200/GBP20 stretch tiers remain separate. No new serving or subscription cost measurement is made. Storage, bandwidth, cached/uncached prefill, hidden reasoning/tool work, idle/burst capacity, reliability, payment/tax/support and training amortization remain unmeasured here. Dynamic INT8 toy byte-model CPU times cannot be converted into billable production-token costs. Strong pretrained Transformer parity, distant-information tasks, scaling evidence and complete economics remain unmet. No frontier, AGI or profitable deployment claim follows.
