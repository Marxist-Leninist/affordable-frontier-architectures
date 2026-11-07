# Independent v32 audit

All assertions passed. Reconstructed all validation/evaluation copy distributions, calibration objectives, saved mixture losses and means. Verified provenance hashes and exact overlap exclusions. All six checkpoints passed bounded full/chunked/pruned and neural causal parity.

{
  "local": {
    "delta_bpb": -0.0334204121424995,
    "paired_document_bootstrap_95": [
      -0.05475881557287652,
      -0.018113909546989212
    ],
    "improvement_supported": true
  },
  "gru": {
    "delta_bpb": -0.006682862717357709,
    "paired_document_bootstrap_95": [
      -0.022546216824131084,
      0.005626990281431657
    ],
    "improvement_supported": false
  }
}

Limitations: No cryptographic chronology: files bind protocol/selection/evaluation but evaluated_test flag is self-report; reviewed execution paths never fit on test. Exact full-document and 128-byte target dedup only; near-duplicate and semantic overlap unmeasured. Convenience cohort; two unadjusted primary comparisons; no frontier quality or service cost claim.
