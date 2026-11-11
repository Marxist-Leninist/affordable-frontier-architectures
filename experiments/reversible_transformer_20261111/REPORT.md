# Reversible Transformer pilot

Complete training protocol: True. Correctness gates passed: True.

This is a tiny, fixed-protocol CPU byte-language study. It tests hidden-state inversion, backpropagation equivalence, memory diagnostics and held-out prediction. It is not an inverse from generated tokens to prompts and does not measure general intelligence.

## Held-out results

All metrics below are byte cross entropy converted to bits per byte; lower is better. Three fixed seeds; final checkpoints, no test/validation selection.

### equal_data

| Arm | Runs | WikiText test BPB mean ± sample SD | FineWeb BPB mean ± sample SD | Mean updates | Mean training CPU sec |
|---|---:|---:|---:|---:|---:|
| ordinary | 3 | 3.67373 ± 0.00638 | 3.92983 ± 0.01245 | 500.0 | 6.376 |
| reversible_graph | 3 | 3.65118 ± 0.01325 | 3.91634 ± 0.01418 | 500.0 | 6.312 |
| reversible_inverse | 3 | 3.65118 ± 0.01325 | 3.91634 ± 0.01418 | 500.0 | 8.357 |

Paired test difference reversible_graph minus ordinary: -0.022558 BPB (seed differences: 11: -0.004759, 22: -0.026805, 33: -0.036109). Negative favors the first arm.

Paired test difference reversible_inverse minus reversible_graph: -0.000000 BPB (seed differences: 11: -0.000002, 22: +0.000000, 33: +0.000001). Negative favors the first arm.

### equal_cpu

| Arm | Runs | WikiText test BPB mean ± sample SD | FineWeb BPB mean ± sample SD | Mean updates | Mean training CPU sec |
|---|---:|---:|---:|---:|---:|
| ordinary | 3 | 3.59282 ± 0.01602 | 3.87683 ± 0.03725 | 657.3 | 8.366 |
| reversible_graph | 3 | 3.59570 ± 0.03625 | 3.85698 ± 0.02881 | 660.0 | 8.364 |
| reversible_inverse | 3 | 3.65118 ± 0.01325 | 3.91634 ± 0.01418 | 500.0 | 8.357 |

Paired test difference reversible_graph minus ordinary: +0.002879 BPB (seed differences: 11: +0.009435, 22: +0.037514, 33: -0.038312). Negative favors the first arm.

Paired test difference reversible_inverse minus reversible_graph: +0.055477 BPB (seed differences: 11: +0.044445, 22: +0.039036, 33: +0.082950). Negative favors the first arm.

## Learning sanity check

| Seed | Arm | Initial test BPB | Equal-data test BPB |
|---|---|---:|---:|
| 11 | ordinary | 7.92697 | 3.66777 |
| 11 | reversible_graph | 7.93933 | 3.66302 |
| 11 | reversible_inverse | 7.93933 | 3.66301 |
| 22 | ordinary | 8.00997 | 3.68046 |
| 22 | reversible_graph | 8.01068 | 3.65365 |
| 22 | reversible_inverse | 8.01068 | 3.65365 |
| 33 | ordinary | 8.01733 | 3.67297 |
| 33 | reversible_graph | 7.99697 | 3.63686 |
| 33 | reversible_inverse | 7.99697 | 3.63686 |

## Reconstruction and gradient parity

| Dtype | Depth | Inverse(forward) max abs | Forward(inverse) max abs | All gradient relative L2 |
|---|---:|---:|---:|---:|
| torch.float32 | 1 | 1.19e-07 | 1.19e-07 | 6.9e-08 |
| torch.float32 | 4 | 3.58e-07 | 2.38e-07 | 1.63e-07 |
| torch.float32 | 12 | 7.49e-07 | 9.54e-07 | 2.15e-07 |
| torch.float64 | 1 | 4.44e-16 | 2.22e-16 | 1.15e-16 |
| torch.float64 | 4 | 6.66e-16 | 5.55e-16 | 2.94e-16 |
| torch.float64 | 12 | 1.28e-15 | 1.11e-15 | 3.79e-16 |

The tied embedding/readout end-to-end model also passed paired logits and all-parameter gradient checks at initialization and after five AdamW updates. Future-input perturbations caused exactly zero changes to earlier logits. Float errors are measured, not described as bit-exact.

## Memory and execution diagnostic

| Arm | Depth | Retained forward nonparameter MiB | Whole-process peak RSS MiB | Median update CPU ms |
|---|---:|---:|---:|---:|
| ordinary | 1 | 1.840 | 320.49 | 4.590 |
| ordinary | 12 | 14.387 | 334.95 | 35.855 |
| ordinary | 4 | 5.262 | 321.51 | 12.594 |
| reversible_graph | 1 | 1.840 | 317.16 | 4.251 |
| reversible_graph | 12 | 14.387 | 337.99 | 35.437 |
| reversible_graph | 4 | 5.262 | 324.94 | 12.416 |
| reversible_inverse | 1 | 0.887 | 317.72 | 5.119 |
| reversible_inverse | 12 | 0.887 | 325.13 | 47.775 |
| reversible_inverse | 4 | 0.887 | 318.62 | 16.061 |

Saved tensors are retained forward graph storage, not whole-step peak memory. The inverse method temporarily stores each recomputed block during backward. Whole-process RSS includes Python/Torch, model, gradients, optimizer and allocator behavior. These CPU results do not establish GPU speed or memory savings.

## Matching and limits

- All arms have identical parameter names and tensor dimensions; actual counts are in correctness.json. Ordinary state width48 differs from two48-wide reversible streams. Embedding duplication and average readout are not bijective.
- Every block has the same forward attention/MLP matrix shapes. Inverse-based differentiation recomputes those functions; equal data is not equal FLOPs. Equal-CPU continuations use more training data when faster and are reported separately.
- Training CPU includes batch sampling, forward/backward, clipping and optimizer. Evaluation/IO/imports are excluded from that metric but included in the execution receipt cap. Shared-runtime interference can affect time comparisons.
- Context64, byte vocabulary256, four small layers, only256000 target-byte exposures at the equal-data endpoint. This is a bounded pilot, not frontier-scale evidence.
- Official WikiText splits are disjoint by dataset construction; exact selected-window overlap is recorded below. That is not broad semantic decontamination. Prior user research has already used this benchmark family.
- FineWeb is out-of-domain frozen evaluation only,100 preselected documents. No corpus text is redistributed.
- Three seeds do not justify strong statistical or generalization claims. No sample-quality or intelligence score is inferred from NLL.
- No scaling proposal to600B follows from this test. Robust larger-scale replication, longer contexts, GPU profiles, mixed-precision stability, downstream tasks and competitive tuned controls would be required.

## Data provenance and overlap

WikiText revision: b08601e04326c79dfdd32d625aee71d232d685c3. License verified via Hugging Face card: CC-BY-SA3.0 / GFDL tags. Source byte hashes and frozen selections are in protocol_frozen.json.
- validation: 128 windows, 4096 scored bytes, 42 documents; 0 exact full-window matches in training, 0 duplicate selected windows.
- test: 256 windows, 8192 scored bytes, 52 documents; 0 exact full-window matches in training, 0 duplicate selected windows.
- fineweb: 256 windows, 8192 scored bytes, 87 documents; 0 exact full-window matches in training, 0 duplicate selected windows.

## Reproduction

Use an existing official/reputable CPU PyTorch2.6 runtime. Run checks.py, study.py freeze, each memory_probe.py arm/depth, then study.py run and report.py. The completed run was bounded by CPU/wall/RSS caps recorded in execution_receipt.json. Original corpus bytes are not bundled; download the pinned official sources using the provenance manifests.

Read INVERSE_ATLAS.md for the operation-level mathematics, assumptions and prior art.
