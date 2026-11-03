# Independent held-out curriculum replication audit

## Frozen protocol

The final stage preserves the completed validation-only curriculum pilot and its learning-rate selection: Transformer 0.003 and vanilla RNN 0.001. I independently verified the pilot selection file hash, paired-training file hash, protocol hash, and every recorded source hash. The full 6,000-step curriculum and 11,129,856 input-token presentations per run are identical to the audited pilot. Every run receives 48,000 supervised queries. Three fresh initialization seeds, 1/2/3, are used per architecture, with fresh Adam and no checkpoint continuation.

The original schedule/data are shared across all rates, models, and seeds. These three seeds therefore quantify initialization variability, not new training datasets. The final stage consumes 66,779,136 training-input presentations and 288,000 supervised queries across six runs. Pilot and previous failed experiments must be included separately when discussing total research cost.

The driver completes all 6,000 steps, saves and reload-checks final weights before evaluation, scores validation and then the reserved test data, and retains all seeds. It does not retune against test outcomes or choose a best epoch. Validation/test evaluation calls cannot alter the already selected learning rate. All failures remain reportable.

## Test separation and task

The pilot reserved test seed base 820000 without scoring it. The final runner uses that base, independently of training base 920000 and validation base 410000. I regenerated all nine test cells without running model inference and independently verified dictionary-oracle answers, 256 cases per cell, and exact balance of query-key and target marginals (16 examples per class). Dataset hashes are saved in independent_test_data.json. Distinct RNG seeds define separately generated samples, not a new task distribution.

There are four/eight/sixteen randomly valued key-value facts and one queried four-bit target. The lengths 512, 2048, and 4096 count synthetic vocabulary symbols, not text bytes. Length 512 is within the training range; 2048/4096 are extrapolation. The task-specific Transformer uses adjacent-token local binding followed by global attention, with no absolute positions. It retains sequence-length-dependent representations, unlike the RNN's fixed hidden state. This is not an equal-inference-memory comparison or a frontier benchmark.

## Final verification

All six final models completed 6,000 steps with the frozen selected rates. I independently verified checkpoint file SHA256s, loaded ordered tensor SHA256s, architecture/step/seed/rate metadata, and recorded exact reload assertions. Current source hashes match the protocol. All nine test-cell dataset hashes and counts match the independently regenerated data, and all 54 seed/cell accuracies match saved predictions against the generated labels. See independent_verified_summary.json for independently computed means, seed accuracies, and reported mean CE. No training or full model evaluation was duplicated by this audit. Validation mastery is not substituted for held-out accuracy, and a failed RNN adequacy gate limits any interpretation as a pure memory-capacity comparison. A successful result can establish only task-specific trained retrieval under this curriculum and compute budget, not general intelligence, economics, or broad architecture superiority.


Verified test accuracy means (Transformer; RNN):
- L512_K4: 100.0000%; 6.2500%.
- L512_K8: 100.0000%; 6.7708%.
- L512_K16: 100.0000%; 5.5990%.
- L2048_K4: 100.0000%; 6.3802%.
- L2048_K8: 100.0000%; 6.3802%.
- L2048_K16: 100.0000%; 4.9479%.
- L4096_K4: 100.0000%; 5.8594%.
- L4096_K8: 100.0000%; 6.9010%.
- L4096_K16: 100.0000%; 5.7292%.
