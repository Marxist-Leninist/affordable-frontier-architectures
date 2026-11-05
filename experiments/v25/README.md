# v25: recomputing neural outputs after distant-prefix replacement

Frozen real-text specificity experiment,2026-11-05. No new training,paid compute or production changes.

## Results

The context-trained and mixed Transformers show no clear aggregate advantage from original over unrelated distant content on this cohort: donor-minus-original means are +0.000117 and +0.002129 bpb,with recipient-cluster intervals spanning zero. Their original2048 contexts still lose to real suffix512 by0.088155 and0.410967 bpb. This does not show useful distant-context prediction after the training repair. Individual target predictions do change substantially(max target-loss changes12.09/9.65 bpb;reset51213.10 bpb),so the result does not prove the models only respond to length.

Means across3 seeds;lower bits/byte is better. All conditions score identical final128 targets across107 windows from68 documents (13696 targets/condition/seed).

|Frozen arm|Original2048|Unrelated prefix2048|Real suffix512|
|---|---:|---:|---:|
|context2048|3.421446|3.421562|3.333290|
|mixed|3.617220|3.619349|3.206253|
|reset512|4.424515|4.430864|3.080105|
|local|3.018915|3.018915|3.018915|
|gru|2.956427|2.956365|2.956192|

The full Transformer outputs are recomputed on both original and donor inputs. This differs from v20,where only explicit copy memory was perturbed while neural outputs stayed fixed. No hybrid mixture is evaluated or retuned here.

Donor-minus-original positive values indicate original distant content predicts better than unrelated distant content. Original-minus-suffix positive values mean the longer original context still hurts. Both comparisons are needed.

- context2048, donor2048 minus original2048: +0.000117 bpb; paired document bootstrap95% [-0.006700815146774465, 0.006436679290332669]; per-seed [0.0012874425419968638, 0.00015269160447584085, -0.0010900433529537068].
- context2048, original2048 minus suffix512: +0.088155 bpb; paired document bootstrap95% [0.07696853235406215, 0.0993741729839947]; per-seed [0.1003383612135984, 0.07974778551007233, 0.08437969464093831].
- mixed, donor2048 minus original2048: +0.002129 bpb; paired document bootstrap95% [-0.007577991725976888, 0.011552078795300926]; per-seed [0.009891947598803075, 0.00647816690730485, -0.009983155893989348].
- mixed, original2048 minus suffix512: +0.410967 bpb; paired document bootstrap95% [0.38666095326469374, 0.43545625647570113]; per-seed [0.33451491682408285, 0.49514254417288495, 0.4032428474583054].
- reset512, donor2048 minus original2048: +0.006349 bpb; paired document bootstrap95% [-0.015960960372875352, 0.029054955281595535]; per-seed [0.0017857491820718607, 0.015000575670093497, 0.0022610279361412466].
- reset512, original2048 minus suffix512: +1.344410 bpb; paired document bootstrap95% [1.3009131381298242, 1.3888272912253357]; per-seed [1.4963284560301906, 1.1985182218461912, 1.3383827723123907].

Local original/donor losses are structurally identical;GRU distant perturbation effects are tiny. These control implementations have different inherited training histories.

Interpretation must not turn content sensitivity into a claim that long context helps overall. A near-zero or uncertain donor effect also does not prove the model uses only sequence length. The intervention preserves lengths and RoPE coordinates but introduces an artificial unrelated-text boundary. It cannot isolate every mechanism or establish distant reasoning.

## Fixed protocol and data

Fresh pinned FineWeb-Edu sample-10BT rows900–999,first4097 UTF8 bytes/document,no joins or content selection.100 documents total282587 capped bytes;68 sufficiently long documents contribute107 windows. Each2049-byte window replaces first1408 bytes with first1408 bytes of the deterministic next eligible different-full-document-hash row,cycling ascending order. Final641 bytes remain identical:513 genuine preceding bytes before earliest target plus all128 targets. Full paths have effective1921–2048 context; suffix512 has385–512.

Frozen weights: v23 context2048/reset512 continuation,v24 mixed continuation,v19 local/GRU. Three seeds1–3,all completed conditions retained. No test selection,training,validation tuning or copy calibration. The full/mixed/reset histories share initial Transformer and250-update ledger but differ in context and compute;local/GRU have no matched new continuation.

Protocol/source/input/data hashes were locked before scoring. Initial preflight failed a too-strict optimized-local float32 parity tolerance:1.2636184692382812e-05 observed against1e-5. Before any corpus scores,we preserved the initial protocol/test/failure and amended only optimized parity to inherited v24 threshold1e-4. Causal remains1e-5 and local donor invariance1e-10. See preflight_amendment.json. All4 corrected preflight tests passed. Audit reconstructs45 aggregate score rows from scalar target losses and checks hashes,donor pairing,causal future invariance and full-head parity.

Bootstrap uses5000 paired recipient-document clusters,seed25,averaging seeds first. Intervals condition on this convenience cohort and do not account fully for cyclic donor cross-document dependence. Seeds share documents;not independent corpus replications. No population guarantee,formal equivalence,convergence,equal-FLOP architecture win or frontier-quality claim.

Both pinned source cards were retrieved. No exact full-document matches to v15–v24 manifests or complete evaluation documents in WikiText train were found;near-duplicate/passage contamination is unmeasured. See DATA_CARD.md,data_audit.json and independent audit for provenance and limits.

## End-to-end original-path inference

Batch1,one CPU thread,float32 weights,first8 eligible documents,3seeds,one full warmup followed by one measured repetition per job;randomized arm/seed/context order seed250025. Optimized prefill and32 observed updates produce33 complete256-way softmax distributions. Timings include full outputs and softmax;not sampled generation,commercial tokens,concurrent service or production throughput. For the local arm,the prefill byte-rate numerator is supplied context length(512 or2048),but the optimized path processes only the final255 structurally reachable input bytes for one output. This is not actual processed-byte throughput or equal long-context capability. Host is shared. Different weights do not cause interpretable architecture-speed changes.

|Arm/context|Prefill bytes/CPU s|Observed decode bytes/CPU s|Prefill tensor state bytes|After32 state bytes|Total wall p50/p95 ms|
|---|---:|---:|---:|---:|---:|
|context2048_512|76969|1392|524288|557056|29.328/35.103|
|context2048_2048|41390|1049|2097152|2129920|80.164/90.824|
|mixed_512|76841|1500|524288|557056|25.971/33.699|
|mixed_2048|43217|1091|2097152|2129920|74.561/86.672|
|reset512_512|73853|1425|524288|557056|29.950/33.106|
|reset512_2048|41373|1061|2097152|2129920|80.258/95.775|
|local_512|133568|1391|130064|130064|27.552/30.714|
|local_2048|554070|1453|130064|130064|25.368/31.421|
|gru_512|35076|3463|512|512|22.868/31.405|
|gru_2048|34952|3363|512|512|64.630/88.418|

Main scoring used 116.951 CPU seconds;timing including warmups used 22.556. Startup,preparation,tests and reporting are not all individually metered;no exact inclusive CPU total is claimed. Measured work is far below900 seconds;separate main/timing hard limits700/100. Tensor state excludes weights,temporaries,Python overhead and peak RSS.24 timing samples per variant. No byte-to-commercial-token conversion is assumed.

## Reproduce

Prepare sibling v19,v23,v24 checkpoints from their pinned instructions and v8/train.txt for contamination checks. Original weights and corpus are not included. Dependencies requirements.txt;local PYTHONPATH=/tmp/frontier-v10-packages:/tmp/frontier-v8-packages. Run prepare_data.py,check_data.py,run.py --freeze,unittest test_v25,run.py,summarize.py,timing.py,audit_results.py,write_report.py. Preserve initial amendment records;do not overwrite historical results. Source and scalar losses are deliverable;rows.json,weights and caches excluded. Artifact manifest hashes all deliverables except itself.

## Economic boundary

No AGI/frontier capability,service profitability or30% margin is established. Byte-level CPU loss and observed-byte update speed cannot establish US$0.05/M uncached input,US$0.001/M cached input orUS$0.10/M output economics. The <=US$500/month same useful average/heavy full-month workload target and separate GBP200/GBP20 tiers remain unproven. All-in cost includes actual useful quality,prefill/decode,cached work,hidden reasoning/tools,storage/network,idle/bursts/failover,payment/tax/support and amortized training. Parent report owns current pricing and private archival.
