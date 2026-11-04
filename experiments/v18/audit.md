# v18 independent prefix-pruning audit

## Static correctness

The attention mask retains keys satisfying `qpos - 128 < kpos <= qpos`: 128 positions including the current token. Two serial local blocks therefore give output position t an inclusive input receptive field [t-254, t]. Residual paths and pointwise normalization/MLPs cannot enlarge this field. For a 2,048-token input, the final 128 outputs are positions 1920–2047, whose combined receptive field is [1666, 2047], exactly 382 inputs. Their next-byte targets are raw positions 1921–2048. The current score slicing uses these identical targets for both paths. Nonoverlapping input windows share one boundary byte as the preceding window's final target and the next input window's first input; their scored target tails do not overlap.

The initial reset-position experiment is not numerically exact at the declared gate. Its results must be retained as a failed control. RoPE satisfies common-shift invariance in real arithmetic, but the float32 angle construction, trigonometry and dot products need not be invariant. Resetting suffix coordinates from 1666–2047 to 0–381 changes those computations, and the trained network may amplify discrepancies. Mathematical relative-position invariance is insufficient evidence of the required numerical parity.

Initializing each local layer with zero-length K/V and an absolute counter of 1666 instead preserves original RoPE coordinates. The zero-length caches do not claim nonexistent prefix state. Early suffix hidden states may differ, but outputs become equivalent once all dependency paths lie inside the retained suffix. Chunk boundaries also change relative to the full input, so small arithmetic discrepancies remain possible and require the original fixed gate.

Both paths apply the vocabulary head to the same final128 hidden positions only. This avoids comparison with an unnecessarily expensive full-head baseline. The unchanged local attention policy is an inference-time intervention on frozen full-attention Transformer weights, not a learned long-range architecture. Full Transformer and GRU pruning controls are not mathematically protected and must not be described as equivalent quality merely because average loss changes happen to be small.

## Executed independent checks

Five tests in `test_pruning.py` passed in 1.286 seconds on 2026-11-04. `audit_tests.txt` records the complete output. CPU inference finished and clearance was sent at approximately 20:23 UTC, before subsequent statistics/timing work. This is scheduling coordination, not proof of host-wide exclusivity.

- A discrete dependency-set computation reproduces the exact [1666,2047] union independently of attention tensors, and direct index arithmetic checks next-byte target alignment.
- Three distributed eligible real documents and all three SHA256-verified frozen seeds compare full chunk128 prefill with `local_prefill` pruning. Maximum sampled logit error is 1.53e-5. Independent float64 log-softmax/gather window losses pass the unchanged 2e-6 tolerance. Every model weight remains bit-identical before and after evaluation.
- Final retained layer caches agree within stated numerical tolerances, with equal allocated state size and absolute counters of 2048. Feeding 32 future observed bytes as a chunk gives maximum logit error 9.54e-6, and both counters advance to 2080.
- An ordinary full-head monolithic model forward is an independent execution-path reference for pruned prefill using monolithic and 37/128/511 chunks. All pass. Poisoning every input outside the retained dependency field leaves scored outputs equivalent.
- For the actual timing workload, pruning to 255 inputs for the final one output and performing 32 single-token observed updates was tested on all three real documents and all three seeds. Across these 33-output probes, maximum logit error is 7.63e-6 and maximum per-probe mean loss difference is 1.25e-7 bits/byte. This supplemental 33-target mean check is distinct from the predeclared 128-target scoring gate.
- The sampled reset-position control has max logit discrepancy 2.78e-4, while preserving absolute positions reduces it to 1.53e-5. This directly reproduces the floating-point failure mechanism without claiming that float32 trigonometry alone is the only contributor.

An initially added **stronger, unplanned per-single-token loss assertion** failed at 2.633217548e-6 bits/byte for seed3. Its log is preserved in `audit_extra_per_token_failure.txt`. The protocol promises a 128-target window loss tolerance, not this per-token tolerance. The final test preserves and reports this diagnostic rather than asserting it as a protocol gate. No original gate was weakened. Logit errors and both the original 128-target means and supplemental 33-target means pass. This distinction is necessary to avoid describing all individual token losses as within 2e-6.

## Full scoring artifact verification

A standard-library audit independently recomputed the means and maximum absolute per-window loss deltas of all nine model/seed score groups, verified 98 unique windows per group, checked the dataset response hash and confirmed all three repaired local gates pass while all three reset-position gates fail. `audit_final.json` records these findings and SHA256 identities of audited code/test files. The full repaired scoring maxima are 2.527236938e-5 logits and 2.820750606e-7 bits/byte per window, below the original fixed limits. Full Transformer and GRU controls fail equivalence, as expected for paths without the structural guarantee.

The independent inference audit samples three real documents, not all 98 windows a second time. Full-score arithmetic checks do not independently reproduce every stored logit. No model training, tuning, paid compute, external writes or production actions were performed by the auditor. The audit did not edit `run.py`.

## Interpretation and limits

The corrected method is structurally exact in real arithmetic and numerically equivalent under the declared tested tolerance, not bit-identical. Repair was informed by failures on this same sample, so the repaired result is not a fresh held-out confirmation. The pruning helper is scoped to this local architecture and inference workflow; arbitrary block types, global normalization, other positional schemes and training use have not been validated.

The final-one benchmark may discard more inputs than the 128-output quality workload (255 versus382), but both are derived from the same dependency rule; the explicit continuation test bridges that difference. None of these checks establishes general long-context task quality, service-level latency, paid-hour utilization, commercial token economics or a speedup until the separate timing results are reviewed.

## Reproduce

From this directory with cached frozen checkpoints and input snapshot:

    V14_DIR=../v14 python -m unittest -v test_pruning

Run numerical tests outside timing workloads. Source identities are recorded in `audit_final.json`.

## Final timing audit (20:25 UTC)

`audit_timing.py` performs a standard-library-only independent audit and writes `audit_timing.json`. All 576 retained samples form exactly 288 unique full/pruned pairs: three lengths, three seeds, eight shared documents and four retained repetitions. The stored sample ordering exactly matches the declared randomized variant order, including RNG advances during both discarded warmups. Every sample satisfies `end_ms = prefill_ms + decode_ms`, has finite positive timings and reports 130,064 retained state bytes. Equal state allocation does not imply bit-identical state values; numerical state agreement was tested with tolerances above.

The script independently reproduces all summary medians, favorable-pair counts and linearly interpolated pooled-sample p95 values in `timings.json` and `validation.json`:

- Prefix2048: 95/96 end-to-end pairs favor pruning; median paired end speedup2.1463921 and prefill speedup8.5894451. Pooled end p50 is58.8716265→26.3496875ms; p95 is72.9203967→36.1632657ms.
- Prefix512: median paired end speedup1.1587499;81/96 favorable; pooled end p50 is30.1909840→26.3964835ms.
- Prefix128: median paired end ratio1.0017749;49/96 favorable; pooled end p50 is24.0626430→24.0098020ms. This is a no-pruning control and does not support a meaningful speedup claim.

These are pooled medians across all96 probes per variant/length, not medians of seed medians. The median paired ratio is also a different statistic from the ratio of separately pooled medians. Repeated observations share documents and seeds and must not be presented as96 independent workload samples or as a confidence interval.

Source review confirms prepared input tensors are outside timers, both variants use the same chunk128 prefill and final-one vocabulary head, and both compute one prefix softmax plus32 observed-update softmaxes. The state-allocation inspection occurs after the final timer. Variants are randomized within each document/repetition pair; length/seed order is fixed, and host interference is not ruled out. This measures teacher-forced local CPU work rather than sampled generation or serving latency. No inference or timing was repeated for this final audit. No blocking discrepancy was found.
