import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).parent
s=json.loads((P/'summary.json').read_text());r=json.loads((P/'results.json').read_text());t=json.loads((P/'timing.json').read_text());a=json.loads((P/'artifact_audit.json').read_text())
table='\n'.join(f"|{k}|{v['512']:.6f}|{v['2048']:.6f}|" for k,v in s['means'].items())
contrasts='\n'.join(f"- Mixed minus {d['b']} at {d['a_context']} bytes: {d['a_minus_b_bpb']:+.6f} bpb, paired document bootstrap95% {d['document_cluster_bootstrap95']}; per-seed effects {d['seed_effects']}." for d in s['comparisons'] if d['a']=='mixed' and d['b'] in ('context2048','reset512'))
time_summary=[]
for arm in ('context2048','reset512','mixed'):
 for ctx in (512,2048):
  rows=[x for x in t['rows'] if x['arm']==arm and x['context']==ctx]
  time_summary.append(dict(arm=arm,context=ctx,prefill_bytes_per_cpu_second=sum(x['prefill_bytes'] for x in rows)/sum(x['prefill_cpu_seconds'] for x in rows),observed_decode_bytes_per_cpu_second=sum(x['decode_observed_bytes'] for x in rows)/sum(x['decode_cpu_seconds'] for x in rows),prefill_bytes_per_wall_second=sum(x['prefill_bytes'] for x in rows)/sum(x['prefill_wall_seconds'] for x in rows),observed_decode_bytes_per_wall_second=sum(x['decode_observed_bytes'] for x in rows)/sum(x['decode_wall_seconds'] for x in rows),prefill_state_bytes=rows[0]['prefill_state_bytes'],final_state_bytes=rows[0]['final_state_bytes']))
(P/'timing_summary.json').write_text(json.dumps(time_summary,indent=2)+'\n')
timing_table='\n'.join(f"|{x['arm']}|{x['context']}|{x['prefill_bytes_per_cpu_second']:.0f}|{x['observed_decode_bytes_per_cpu_second']:.0f}|{x['prefill_state_bytes']}|{x['final_state_bytes']}|" for x in time_summary)
text=f'''# v24: fixed mixed-context continuation

Bounded isolated real-text CPU experiment,2026-11-05. No paid compute or production changes. Three fixed seeds; no tuning or evaluation-based checkpoint selection.

## Quality

Lower bits/byte is better. Nested512 and full2048 score the **same final128 next-byte targets**. Means across3 seeds on {s['windows']} windows from {s['documents']} eligible documents ({s['targets_per_condition_seed']} targets per condition/seed).

|Condition|Nested512 input|Full2048 input|
|---|---:|---:|
{table}

{contrasts}

Mixed training tests a compromise between context lengths, not an architecture change. Paired intervals use5000 document-cluster resamples,seed24, averaging seeds first and preserving document windows. These intervals condition on a deterministic convenience cohort, not population generalization. Seeds share data. The0.02-bpb screen is descriptive, not formal equivalence. No equal-FLOP, equal-time, convergence, frontier-quality, AGI or profitable-service claim follows.

## Frozen protocol and controls

Only mixed is newly trained:250 updates/seed, alternating full2048 on zero-based even steps and independently reshaped4×512 x/y on odd steps,125 each, starting from the same v19 full Transformer adapt500 checkpoint. Fixed model RNG seed240000+seed. Saved v23 context2048 and reset512 step250 controls are reused without retraining. The exact v23 ledger (seed23112026), initial weights, optimizer, schedule, number of updates and targets are matched. Each arm/seed has512000 continuation target presentations; only1536000 new targets and750 new updates are computed here. Context/reset structure changes actual FLOPs and timing. The protocol references field mentions frozen local/GRU as historical background only; neither is evaluated in this v24 run. Original v19 training and v23 control training are inherited costs, not free training or new v24 work.

Fresh AdamW lr.001,betas.9/.95,weight decay.01,clip1,warmup25 and the exact v23 cosine schedule. All three final step250 checkpoints retained privately; no intermediate checkpoint selection. Protocol/data/input/source hashes were frozen before tests and training. See protocol.json,train_starts.npy and manifests. Seven pretraining tests passed; audit reconstitutes all24 score rows from per-target losses,verifies3 final hashes/750updates and long-context causal/optimized equivalence errors ({a['final2048_causal_future_suffix_error']},{a['final2048_optimized_logits_error']}).

## Data

Pinned public ungated FineWeb-Edu sample-10BT train rows800–899 used only as held-out evaluation,revision87f09149ef4734204d70ed1d046ddc9ca3f2b8f9. First4097 UTF8 bytes/document,no joins,normalization or content filtering.108 nonoverlapping2048 windows; nested512 uses final512 observed bytes,effective target context385–512 versus1921–2048. Viewer revision is checked fail-closed; corpus/doc hashes retained. Pinned WikiText2 raw train provenance is unchanged. Both pinned cards were reread. See DATA_CARD.md for ODC-BY and WikiText CC-BY-SA3.0 metadata/4.0 prose inconsistency; corpus/checkpoints are not redistributed. Zero exact full-document matches to v15–v23 manifests,within-cohort duplicates or complete documents in WikiText train. Near duplicates and passage overlap remain unmeasured. Cohort is not a random representative sample.

## Measured resources and inference

New mixed training {s['training_cpu_seconds']:.3f} CPU seconds; full train/score main {s['main_cpu_seconds']:.3f} CPU seconds; scoring alone {sum(x['cpu_seconds'] for x in r['scoring']):.3f}; timing phase including warmups {t['cpu_seconds']:.3f}. Startup,preparation and small auxiliary work are not all separately metered,so an exact inclusive CPU total is not asserted. Main hard limit820 CPU seconds,timing hard limit80; actual measured work far below900. One CPU thread,float32 weights,float64 scoring normalization,shared host without exclusivity.

Inference is batch1 on first8 eligible documents,3seeds,three trained arms and two contexts. Fixed randomized job order(seed240024),one full warmup per measured document. Optimized prefill computes only final output,then32 identical observed-byte inputs each produce a full256-way next-byte distribution. This is teacher-forced cached continuation,not sampled generation. All architecture shapes are identical; small rate differences cannot be attributed causally to trained weights. Timing is descriptive,one repeat/doc,no concurrency,quantization or production latency guarantee.

|Arm|Context|Prefill byte/s CPU|Observed decode byte/s CPU|Prefill KV bytes|After32 KV bytes|
|---|---:|---:|---:|---:|---:|
{timing_table}

Parameters {t['rows'][0]['parameters']}; actual FP32 weight tensor bytes {t['rows'][0]['fp32_weight_bytes']}. State is dense per-layer K/V float32,not total process RSS. Wall-time rates and individual measurements are retained in timing_summary.json/timing.json. Byte units must not be converted into commercial tokens without measured tokenization and system workloads.

## Economic boundary

Requested rates remain US$0.05/M uncached input,US$0.001/M cached input and US$0.10/M billable output with30% margin on net revenue. Conditional output-only hardware allowance US$0.07/M implies3968.25 billable output tokens/s per US$1/hour at100% productive utilization,divided by utilization otherwise,before other costs. This CPU byte-language-model test supplies neither commercial-token conversion nor actual all-in $/useful-token evidence. Frontier useful-quality parity remains independent. The <=US$500/month full average/heavy workload subscription and separate GBP200/GBP20 stretch tiers remain unproven. All-in costs must include prefill/decode,caching,hidden reasoning/tools,bandwidth/storage,idle/bursts/failover,payment/tax/support and training amortization; cost<=70% net revenue,not break-even. Parent report owns current primary pricing and economic scenario recheck.

## Reproduce and deliver

Prepare sibling v19 original checkpoints,v23 fixed control checkpoints,v14 training manifest and v8 train bytes according to pinned instructions; checkpoints are not included. Dependencies:requirements.txt and local PYTHONPATH=/tmp/frontier-v10-packages:/tmp/frontier-v8-packages. Run prepare_data.py,check_data.py,run.py --freeze,unittest test_v24,run.py,summarize.py,audit_results.py,timing_distribution.py,write_report.py. Never overwrite historical results. Protocol hashes are written before training. timing_amendment.json documents a pre-timing source-review correction adding softmax to produce full probabilities; the original frozen timing.py is preserved. No failed training/scoring runs occurred; the pre-timing softmax amendment changes only measurement completeness, not training, model selection or quality results.

Corpus rows.json,checkpoints and __pycache__ are excluded from the deliverable. Loss arrays are scalar losses; train_starts.npy contains offsets,not text. Reproduction requires lawful dataset access and inherited checkpoint training. See artifact_manifest.json for deliverable hashes.
'''
(P/'README.md').write_text(text)
files=[f for f in P.iterdir() if f.is_file() and f.name not in ('rows.json','artifact_manifest.json')]
(P/'artifact_manifest.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)},indent=2)+'\n')
