# Independent v14 source and bounded-test audit

2026-11-04. This audit covers the frozen experiment implementation, local synthetic/model tests, and final-artifact validation added after fitting completed. It does not certify repository publication or production performance. The final validation section below distinguishes sampled neural rescoring from full statistical rescoring.

## Useful finding

No critical target-alignment, reset, or test-selection defect was found. Ten tests pass in `audit_tests.txt`: seven new independent v14 checks and three inherited model tests rerun here. Source hashes and the independently regenerated complete training ledger agree with the saved protocol. The tested bounded claim is that this code makes a validation-locked, paired-target comparison. No statistical equivalence, convergence, frontier capability, AGI, or dollar-cost success is established by these checks.

## Alignment, data, and selection

- Neural evaluation predicts byte i+1 from observed bytes through i, resets state independently in each complete 128-byte window, and excludes source byte zero plus incomplete trailing targets. Synthetic logits establish the expected loss and target alignment, including a 33-window batch boundary.
- Statistical evaluation uses precisely the same target positions. A spy verifies the suffix context byte-by-byte and its reset at each window boundary. Training counts the same sampled windows as neural fitting, including repeats/overlaps. A synthetic ledger verifies that every window starts afresh. N-gram probabilities normalize, and evaluation does not alter count tables.
- The entire 4,000-by-8 ledger is independently regenerated from seed 14112026 and compared exactly; its SHA256 and all three protocol source hashes match. This is 4,096,000 training-target presentations per run, not distinct corpus bytes. The inherited `ngram.py` docstring says 1,024,000 because it originated in v12; v14's entrypoint is `run.py`, which supplies the larger correct ledger.
- Actual selection statements are tested by extracting their AST without running fitting. Pilot LR selection uses final (4,000-step) validation only. Gate selection uses the **first recorded checkpoint** at or below each threshold, including exact equality and nonmonotonic curves. Missing crossings remain null. Test evaluation covers the unique gate-selected checkpoints plus the final checkpoint, not test-selected winners.
- Fixed validation gates are 2.5, 2.35, and 2.25 bits/byte. Sharing a threshold does not guarantee matching achieved loss: either model may overshoot. Report actual test gaps and the prespecified .02-bpb descriptive screen; do not infer statistical equivalence or tune checkpoint choice on test gaps. The same held-out test corpus has been inspected in earlier experiments; this is exploratory continuation, not untouched-benchmark confirmation.
- Existing causality, cached-versus-full forwarding, batch invariance, parameter matching, finite nonzero gradients, and save/reload tests pass for both neural architectures.

## Timing semantics that must remain explicit

Both timing paths consume the same eight deterministic corpus prefixes and 32 observed continuation bytes, with five repetitions per prefix and two discarded warmups. Synthetic spies verify all calls and contexts. Each sequence computes **33 next-byte distributions**: one after prefill and 32 after cached updates. It is a teacher-forced replay, not sampled generation of exactly 32 output bytes. The final distribution predicts the byte after the 160-byte observed sequence.

The p95 fields summarize 24 retained local sequence probes. In particular, `decode_step_p95_ms` is the p95 of **sequence-average** cached-step times, not p95 individual inter-token latency. No queue, network, tokenizer, sampling, tool, retry, or service overhead is included. Host interference and fixed family/timing order remain uncontrolled. Source comments place timing after all fitting/scoring in this process; this does not establish whole-host isolation.

Quality scoring resets at 128 bytes; latency starts with a 128-byte prefix and grows cache to 160 bytes. These are related but different workloads, and neither is a long-context quality test. Neural prefill computes logits for all prefix positions even though only the last distribution is used; a last-position-head implementation could improve prefill timing. The finite-order statistical model only needs the final suffix for prefill, legitimately avoiding processing the whole prefix when no prefix likelihood is requested.

## Fairness, costs, and remaining gaps

The modern Transformer and residual gated GRU are parameter-matched within approximately .2% and receive identical sampled targets, initialization-seed choices, and LR-grid sizes. The statistical family shares targets but not parameter count, RAM, FLOPs, or training work. Its three order choices share a maximum-order table. This is not an equal-total-resource architecture contest.

The experiment expands the upper LR bound and uses four times v12's target presentations. It does not establish that either baseline is converged or optimally trained. Comparison with v12 is confounded by a fresh ledger, changed warmup, changed cosine horizon, and selected LR; early v14 checkpoints are not replicas of v12's completed schedule. Any training-adequacy conclusion should be based mainly on within-v14 curves, with validation/test and seed spread shown explicitly.

Training seconds exclude validation, checkpoint copying/serialization, test scoring, and latency probes. Report total wall time and all pilot/final fitting costs, not only the winning checkpoint's fitting time. A first gate crossing reports training progress inside a longer run; because each run continues to 4,000 steps and LR is selected using full pilot runs, it is not a realized early-stopping end-to-end compute saving. Serialized n-gram bytes are not live RAM. Neural state tensors exclude interpreter, allocator, temporary activation, and other runtime overhead.

No CPU byte rate can be substituted for commercial subword-token throughput. No measured hourly cost, real arrival trace, production utilization, heavy-user usage ledger, full overhead allocation, or frontier quality reference exists here. The external 30%-margin cost thresholds remain separate requirements, not measurements that this experiment satisfies.

## Reproduce audit

From `experiments/v14`, with dependencies installed and `protocol.json` plus `train_starts.npy` produced by the experiment:

    python -m unittest -v test_v14 test_models

The new checks do not train models or rescore the full corpus. They completed in 0.170 seconds in the initial run. Source changes require rerunning the affected tests and reconciling the protocol hashes rather than preserving a stale pass.


## Final-artifact validation, 09:03 UTC

**Pass for the reported bounded results; no new blocking issue found.** `audit_final.py` and `audit_final.json` record independent verification performed after `results.json` was complete. This run took 16.38 seconds and did not repeat latency measurement or modify frozen source.

- All dataset split hashes, frozen source hashes, protocol/selection hashes, and all nine evaluated checkpoint hashes and byte sizes match. The exact LR choices are .01 for both families, the middle of the tested grid. All final seeds, validation gate crossings, selected checkpoint sets, and summary pair gaps agree with independent recomputation.
- Each of all nine evaluated neural checkpoints was loaded with `weights_only=True` and rescored on 20 distributed test windows, including first/last windows and a batching boundary. The audit constructs windows independently and computes float64 logsumexp/gather losses rather than invoking `run.evaluate`. Maximum difference from the saved per-window float32 losses is **3.61e-7 bits/byte**. The mean of every saved complete block array also matches its reported full-test score within **9.79e-9 bits/byte**. This is sampled fresh neural rescoring plus full saved-array aggregation, not a fresh full-corpus rescore of every neural model.
- The statistical model was rebuilt from the verified entire training ledger. A separately written sequential context/target loop rescored all **1,291,904 test targets**, using the shared `Ngram.prob` routine. It exactly reproduces **2.226091549414156 bpb**. Synthetic tests separately check that shared routine's normalization and reset behavior.
- Summary means, sample standard deviations, and total pilot-plus-final fitting seconds match independent calculations. Final mean test bpb is **2.3596957572 Transformer**, **2.2387842759 GRU**, **2.2260915494 statistical**. Every final neural validation curve improves at each recorded step; this is continued-learning evidence, not convergence evidence.
- All three prespecified neural matched-quality screens fail exactly as reported. At the 2.5 gate, all GRU-minus-Transformer test gaps exceed .02 bpb; at 2.35, all absolute gaps exceed .02 in the opposite direction; at 2.25, Transformer does not cross. Lower measured GRU latency is therefore **not an equal-quality speedup under this protocol**.
- Final `final_tests.txt` shows 17 passing checks, comprising the 10 model/source checks above plus seven economics checks. The memory audit's object-graph method and result were inspected: 54,023,171 bytes of deduplicated Python objects versus 4,506,831 serialized bytes. The method correctly keeps these separate from process RSS and total runtime memory; this audit did not perform a second memory traversal.

The README correctly preserves all timing caveats, the missed matched-quality gate, unknown CPU dollar allocation, prior test exposure, and the absence of a second-corpus or frontier-quality validation. No claimed commercial-token economic pass follows from these bytes-based results.

To repeat the final audit after regenerating checkpoint and per-window intermediates:

    WIKITEXT_DATA=. python audit_final.py
