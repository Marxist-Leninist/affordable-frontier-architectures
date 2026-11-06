from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;s=json.loads((P/'summary.json').read_text());profile=json.loads((P/'profile.json').read_text());compute=json.loads((P/'compute.json').read_text())
out='''# Anti-attention v7: retained-optimizer mixed-context training

Bounded isolated CPU experiment, 6 November2026. Both matched attention and signed U−A models use retained AdamW and three paired seeds. This closes the retained128 control missing from v6 and tests fixed within-update50/50 context mixing. No new hyperparameter tuning or winner selection.

## Useful result

Fixed50/50 mixing gives an intermediate quality tradeoff, not a better result than both specialist contexts. Compared with long-only training, both architectures improve every128-context metric in every seed but worsen every512-context metric in every seed. Short-only training remains better on mean128-context scores but fails badly at512. Signed mixed training improves WikiText128 by0.00735bpb over signed long-only while losing0.01676bpb at512; FineWeb improves0.00408 and loses0.01550 respectively. No matched-quality speed or frontier-cost win follows.

## Measured quality

Three-seed mean bits/byte, lower is better; identical final128 heldout targets at both evaluation contexts:

|Model|Metric|Short128 training|Long512 training|Mixed50/50 training|
|---|---|---:|---:|---:|
'''
for kind,ms in s.items():
 for metric,v in ms.items():out+=f"|{kind}|{metric}|{v['short']:.6f}|{v['long']:.6f}|{v['mixed']:.6f}|\n"
out+='\nPaired mixed-minus-specialist differences (negative is better), including seed range:\n\n'
for kind,ms in s.items():
 for metric,v in ms.items():
  out+=f"- {kind} {metric}: mixed−long {v['mixed_minus_long']:+.6f} [{min(x['mixed_minus_long'] for x in v['pairs']):+.6f}, {max(x['mixed_minus_long'] for x in v['pairs']):+.6f}]; mixed−short {v['mixed_minus_short']:+.6f}.\n"
out+='''
Interpretation must preserve the whole tradeoff, not select one evaluation context. No strict noninferiority margin was prespecified. These are three tiny byte models' seeds on reused exploratory heldouts, not independent broad datasets or a frontier-quality comparison. Full per-seed/window losses are in results.json.

## Protocol and correctness

Each architecture has131,904 parameters, matched initialization, training source, target samples, learning-rate schedule, weight decay and gradient clipping. Original800-step128-context training is replayed to recover Adam moments and counters; each reconstructed weight tensor must equal its historical checkpoint exactly. Every continuation starts from separately deep-copied model/optimizer state, runs200 updates and409,600 target bytes with the same explicit cosine LR. Short splits four513-byte spans into16 sequences of129 bytes; long keeps4. Mixed keeps two spans long and splits the other two, alternating span assignments each step. Each half predicts1024 targets, so averaging their mean losses yields exact target weighting. One backward/clip/update follows the combined mixed loss. Available context, position range, sequence layout and kernel overhead change together. Equal targets/updates do not equal FLOPs or wall time.

Independent review found source-state aliasing in the initial pilot; it was stopped and excluded, then corrected with per-arm deep copies. method_notes.md discloses failed-pilot compute. The source-isolation regression test passes. Final validation checks target identity,18 optimizer/cache checkpoints, and exact replay of all six historical retained512 loss results. Checkpoint states use original U−A convention, Adam step1000. Optimized A−U inference negates signed output projection; optimizer first moment also needs negation if training that converted representation.

## Real dataset provenance

Inherited frozen Salesforce/wikitext wikitext-2-raw-v1 revision b08601e04326c79dfdd32d625aee71d232d685c3; UTF8 byte tokenizer, first2,000,000 deduplicated training bytes. Source manifest checks all official split hashes. WikiText card's metadata/prose license versions differ (CC-BY-SA3.0/GFDL metadata, CC-BY-SA4.0 prose); retain source attribution/share-alike obligations rather than resolving that inconsistency by assumption. Test offset196608,128 windows ×128 targets=16,384; same final targets evaluated with128/512 input contexts.

HuggingFaceFW/fineweb-edu sample-10BT upstream train split, revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, rows1500–1599, capped first4097 UTF8 bytes/document; first64 eligible513-byte windows score final128=8,192 targets. This is evaluation-only in this experiment. ODC-BY dataset card, underlying web-text rights caveat. Frozen response/document hashes fail closed. Exact line/block exclusions are not exhaustive near-duplicate decontamination; document windows are correlated. No test bytes enter training. Raw corpora are excluded from publication.

## Training speed, measured separately

Two CPU threads, torch2.6.0, seed2 original-state models with actual retained Adam loaded separately per arm. Five warmups +31 randomized interleaved full optimizer steps per configuration, real training spans, same2048 targets/step. Full step includes forward, CE, backward, clipping, Adam; batch assembly is excluded. Source-state copying/loading is outside timing. No other experimental training runs concurrently. Timings are tiny shared-CPU probes, not GPU/service evidence.

|Model/configuration|Median full step ms|Target bytes/s|
|---|---:|---:|
'''
for k,t in profile['training_median_seconds'].items():out+=f'|{k}|{t*1000:.4f}|{2048/t:.1f}|\n'
out+='''
## Inference timing

Seed2 mixed-trained weights, optimized mathematically equivalent implementation; batch4,5 warmups+31 randomized interleaved measurements. Prefill emits full logits. Decode is one next-token call after the stated prefix; cache-building time excluded from decode. This decomposition does not represent end-to-end service latency.

|Prefix|Probe|Attention ms|Signed ms|Attention bytes/s|Signed bytes/s|
|---:|---|---:|---:|---:|---:|
'''
for n,r in profile['inference'].items():
 for task,v in r['median_seconds'].items():
  work=4 if task=='decode' else 4*int(n);out+=f"|{n}|{task}|{v['attention']*1000:.4f}|{v['signed']*1000:.4f}|{work/v['attention']:.1f}|{work/v['signed']:.1f}|\n"
out+='''
## Compute and economics

'''
out+=f"Corrected quality-training targets: {compute['corrected_total_quality_training_targets']:,} including reconstruction; corrected run wall time {compute['corrected_elapsed_wall_seconds']:.2f}s. Reconstruction {compute['reconstruction_seconds']:.2f}s and continuation optimizer steps {compute['continuation_step_seconds']:.2f}s are partial components, not inclusive cost. Disposable profiling adds{compute['disposable_profile_training_targets']:,} targets. Invalid pilot adds at least{compute['invalid_pilot_confirmed_targets']:,} targets plus an unmeasured possible partial reconstruction. Tests/evaluation/setup also consume compute. No new paid resources or production changes; invoice rate not observed, so no zero-cost claim.\n"
out+='''
See economics.md: retain target rates$0.05/M uncached input,$0.001/M cached input,$0.10/M output and30% margin on net revenue. All-in cost must be≤0.70 net receipts. Conditional output-only hardware floor3968.254 billable output tokens/s per$1/hour at100% productive utilization excludes other costs; at utilization u the productive floor divides by u once. Mixed input/output workload must separately count revenue and prefill/decode/cache, non-billable work, memory, idle/burst/reliability, storage/bandwidth, training amortization and operations. Byte throughput is not commercial-token throughput. No actual hardware rate, useful-token conversion, full workload SLA or frontier quality is measured; cost per useful token and commercial gate gap remain unestablished for every arm. The≤$500 subscription preserves full workload and30% net margin; no profitable scaling extrapolation follows.

## Reproduce

Python3.12, torch2.6.0 CPU, numpy; pyarrow for pinned data download. Run download_wikitext.py and fineweb/prepare_data.py or provide matching ANTI_DATA_DIR and ANTI_FINEWEB_ROWS. Source hashes fail closed. Original selected checkpoints are regenerated with baseline.py (full historical validation-only grid), then attention_{1,2,3}_0.006.pt and signed_{1,2,3}_0.006.pt copied into checkpoints/. This extra reproduction expense is not part of v7's measured continuation budget.

Run test_state_isolation.py; mixed_control.py (external900-second cap used in recorded run); profile_v7.py only after training finishes; validate_v7.py; summarize_v7.py; report_v7.py. Text-only repository edition excludes raw corpora and regenerable binary checkpoints. historical_long.json is prior retained512 evidence used to test exact replay, not new independent samples. Protocol and all recorded numeric results are retained.
'''
(P/'README.md').write_text(out)
