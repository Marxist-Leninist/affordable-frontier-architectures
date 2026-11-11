# Prospective runtime-only sweep amendment

The original sweep stopped with blocked_no_fit before opening held-out data. Its synthetic timing receipts and all original frozen source files remain unchanged. The original 8-window candidate plus 128-window deeper checks failed the conservative budget. No quality score was used to choose this amendment.

The revised plan retains all 101 counts (0–100), all six matched-step checkpoints, and training repetition r=2. Every count uses the same eight windows per domain. Additional larger checks use 64 windows per domain only at preregistered counts 0, 2 and 100. This is a small exploratory sample, not a validated optimum or a substitute for the completed standard held-out comparison.

The budget is still 7,200 seconds combined across the original attempt and this continuation, not another 7,200 seconds. Actual elapsed_debited_seconds from the original selection receipt is subtracted; new process setup/imports, verification, model loading and all evaluation also count. Per-checkpoint evaluation remains capped at 1,200 seconds, with a conservative predicted limit of 1,050 seconds, margin 1.5 and global reserve 600 seconds.

For each checkpoint, reuse the verified synthetic envelope a+br. Revised prediction is T(N)=(N/4)*(101a+5050b)+16*(3a+102b), N=8. The second term is two domains ×64 windows / batch8 across counts0,2,100. New setup elapsed is added before the final fit check. No synthetic recalibration or held-out performance-based selection is performed by bounded_sweep.py.

All original source hashes, new prospective source hashes, synthetic-input identity, original failed decision, bound calibration receipt hashes, checkpoint identities, paired initialization/batch hashes and all eighteen completed standard checkpoint evaluations are verified before held-out access. Amendment and selection receipts are written first. Full forward calls are independent; seed/topology priority rotates. Missing, capped and nonfinite statuses remain explicit. Output bounded_sweep_results cannot overwrite the original sweep_results or an existing bounded output.

## Reproduction

First reproduce the original source/checkpoints and original synthetic calibration through its existing gated sweep command. If that original scope fits on different hardware, this no-fit amendment does not apply. If it blocks_no_fit without held-out access, bind those new receipts using the standard-library helper:

    python bounded_runtime_plan.py --prepare-receipts sweep_results --output bounded_calibration_receipts.json

Then, after explicit authorization and completed original execution gates:

    ../frontier_cpu_runtime_20261110/venv/bin/python bounded_sweep.py --go-after-nat-complete --calibration-dir sweep_results --calibration-receipts bounded_calibration_receipts.json

The bound receipt manifest is an input identity record, not a hardcoded elapsed-time constant. A fresh reproduction uses its own measured calibration elapsed. Model execution is not part of amendment preparation. Keep original blocked receipts in private provenance; release only a reviewed allowlist, excluding corpus/cache files, prelaunch_history, private_backups and internal receipts. Source/data licenses and model-card limitations are unchanged.
