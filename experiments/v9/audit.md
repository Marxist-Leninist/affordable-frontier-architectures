# v9 delayed-recall results and audit

All 12 preregistered runs completed. All tuning and final scores are preserved in results.json and run.log.

## Interpretation

Both models achieved 100% held-out accuracy at every trained length (8/16/32/64) in all three final seeds. The vanilla RNN retained 100% accuracy at unseen lengths128/256 in every seed. The tiny sinusoidal-position Transformer averaged95.6055% at128 and58.8216% at256. These are context/positional extrapolation conditions for the Transformer, not an isolated test of memory decay.

The remembered target carries only3bits (one of8symbols), independent of distractor length. Perfect256-position recall does not show retention of many facts, long text, broad intelligence, or superiority over typical modern Transformers. No positions above256 were tested. Parameter matching and input-token/query budgets do not match computation or optimize all architecture-specific hyperparameters.

## Final results

|Model|Seed|Length|Accuracy|Query CE (nats)|
|---|---:|---:|---:|---:|
|transformer|1|8|100.000000%|0.000403223|
|transformer|1|16|100.000000%|0.000414543|
|transformer|1|32|100.000000%|0.000394765|
|transformer|1|64|100.000000%|0.000407529|
|transformer|1|128|100.000000%|0.000484323|
|transformer|1|256|37.500000%|2.940927400|
|recurrent|1|8|100.000000%|0.000722493|
|recurrent|1|16|100.000000%|0.000728022|
|recurrent|1|32|100.000000%|0.000811973|
|recurrent|1|64|100.000000%|0.000856408|
|recurrent|1|128|100.000000%|0.000882574|
|recurrent|1|256|100.000000%|0.000919062|
|transformer|2|8|100.000000%|0.000376229|
|transformer|2|16|100.000000%|0.000390779|
|transformer|2|32|100.000000%|0.000390919|
|transformer|2|64|100.000000%|0.000392523|
|transformer|2|128|100.000000%|0.001423299|
|transformer|2|256|62.597656%|2.043987395|
|recurrent|2|8|100.000000%|0.001941286|
|recurrent|2|16|100.000000%|0.001924590|
|recurrent|2|32|100.000000%|0.001925514|
|recurrent|2|64|100.000000%|0.001924507|
|recurrent|2|128|100.000000%|0.001925422|
|recurrent|2|256|100.000000%|0.001925258|
|transformer|3|8|100.000000%|0.000595358|
|transformer|3|16|100.000000%|0.000572950|
|transformer|3|32|100.000000%|0.000574054|
|transformer|3|64|100.000000%|0.000574083|
|transformer|3|128|86.816406%|0.411353506|
|transformer|3|256|76.367188%|1.004022444|
|recurrent|3|8|100.000000%|0.000873322|
|recurrent|3|16|100.000000%|0.000840245|
|recurrent|3|32|100.000000%|0.000844803|
|recurrent|3|64|100.000000%|0.000845144|
|recurrent|3|128|100.000000%|0.000842957|
|recurrent|3|256|100.000000%|0.000843785|

## Tuning

Selection used mean validation CE across trained lengths only. Transformer selected0.01; RNN selected0.003.

|Model|LR|Mean validation CE|
|---|---:|---:|
|transformer|0.001|0.008650253709|
|transformer|0.003|0.001796107731|
|transformer|0.01|0.000454490511|
|recurrent|0.001|0.006902314131|
|recurrent|0.003|0.001255531157|
|recurrent|0.01|0.539393028943|

## Audit

- Tested masked numerical gradients in every parameter tensor; maximum absolute errors: Transformer1.643e-10, RNN6.649e-11.
- Exact zero direct output gradients at all nonquery positions, with intact backpropagation through earlier hidden states.
- Tested all-masked rejection, causal behavior, batch invariance, direct query CE, and balanced held-out targets.
- Verified paired training-data hashes across models/LRs for each seed.
- Verified six tuning and six final runs, each192,000 input tokens and6,400 supervised queries. Each length has1,600 training examples per run.
- Verified source hashes against final artifacts.
- Parameter counts: Transformer13,728 (width32), RNN13,736 (width101).
- Chance is12.5% when guessing among the8valid target symbols. Full output softmax has17symbols.
- No extra trials, parameter updates, paid compute, or production actions were run.
