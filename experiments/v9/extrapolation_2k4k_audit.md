# 2k/4k delayed-recall extension audit

All six original v9 final models were deterministically reconstructed, with byte-for-byte parameter SHA256 matches before evaluation. Selected learning rates unchanged: Transformer0.01, RNN0.003. No retuning or extra candidate trials. Original results/protocol hashes verified unchanged.

At both2048 and4096 vocabulary symbols, every RNN seed scored1024/1024 (100%) and every Transformer seed128/1024 (12.5%). Each evaluation has128cases per target, and identical input hashes across all six models.

This remembers one of eight symbols, only3bits, regardless of delay. It does not establish retention of many facts or long text, nor superiority over typical Transformers. The Transformer extrapolates absolute sinusoidal positions/context well beyond its training maximum64.

## Exact-forward checks

The single Transformer block computes Q at the final query, K/V at all positions, and its final-position residual/FFN/norm. All input keys are causally available to the final query. Position indices remain0..length-1, including4095. Earlier-position FFN outputs cannot influence final outputs in this single-block architecture. This optimization would need reconsideration for multiple blocks. The RNN retains only its current state during inference.

Random and every trained model passed optimized-versus-full-reference checks at8/64/128/256. Largest probability difference3.3306690738754696e-16. Stable query CE also agrees within2e-12.

## All results

|Model|Seed|Length symbols|Correct/1024|CE nats|Evaluation seconds|
|---|---:|---:|---:|---:|---:|
|transformer|1|2048|128|7.601120577515|1.324023|
|transformer|1|4096|128|7.111603985173|2.687182|
|recurrent|1|2048|1024|0.002841323850|2.225167|
|recurrent|1|4096|1024|0.006145316191|4.989215|
|transformer|2|2048|128|5.627917999191|1.586339|
|transformer|2|4096|128|6.856113456347|2.909813|
|recurrent|2|2048|1024|0.001924604745|2.035178|
|recurrent|2|4096|1024|0.001924285184|3.838752|
|transformer|3|2048|128|10.007652005015|1.227192|
|transformer|3|4096|128|10.484855829329|3.028799|
|recurrent|3|2048|1024|0.000844332427|2.423249|
|recurrent|3|4096|1024|0.000844460068|4.872006|

## Memory estimates

Batch16. Main Transformer z/a/K/V arrays plus attention score/probability arrays total34MiB at2048 and68MiB at4096. These are array-size estimates, not measured process peak RSS. Extra temporary norm/matrix operations, positional arrays, inputs and Python overhead are not included. A conservative working-array allowance of twelve B*T*D float64 arrays plus two attention arrays is98MiB at2048 and196MiB at4096; this allowance is an estimate, not an observed peak or formal allocator bound.

RNN two state-sized float64 arrays total25,856bytes; allowing six such temporaries gives77,568bytes. Full stored x/y evaluation arrays add32MiB at2048 or64MiB at4096 for either model. Model parameters and small output arrays add more. Data generation occurs separately and can temporarily allocate extra integer arrays.

Walltimes describe this implementation only; no inference-speed comparison is warranted. No quadratic attention matrix was allocated for long evaluations. No resource failure or evaluation reduction occurred.

Artifacts: extrapolation_2k4k_protocol.json, extrapolation_2k4k_results.json, extrapolation_2k4k_tests.json, extrapolation_2k4k.py, extrapolation_2k4k.log.
