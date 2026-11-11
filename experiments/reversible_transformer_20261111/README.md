# Completed result, 11 November 2026

The numerical study completed successfully. This is the completed fixed-protocol study. See PUBLICATION_AUDIT.json for the public-file selection and original hashes.

## What happened
- Genuine hidden-state inversion passed: worst FP32 round-trip error across tested depths1,4,12 was9.54×10⁻⁷. At12 layers, full input/parameter gradient relative L2 error was2.15×10⁻⁷.
- All three models learned real held-out text. Initial test BPB was roughly8; trained equal-data BPB was roughly3.65–3.67.
- Reconstructing activations preserved the learning result of the identical architecture: paired test BPB differed by at most1.78×10⁻⁶ across three seeds after500 updates.
- It did not supply a better learning algorithm. Its mean equal-data training CPU was8.357s versus6.312s for ordinary autograd on the same reversible architecture:32.4% extra CPU.
- Retained forward autograd storage fell83.1% at four layers and93.8% at12 layers. This metric excludes parameters and optimizer and is not whole-step peak or GPU memory. Whole-process peak RSS in the fresh four-layer probes was318.62MiB inverse versus324.94MiB stored-graph reversible.

## Held-out prediction
Lower bits-per-byte is better. Mean of three fixed seeds.

| Endpoint | Ordinary Transformer | Reversible, stored-graph BP | Reversible, inverse-based BP |
|---|---:|---:|---:|
| Same500 updates, WikiText test | 3.67373 | 3.65118 | 3.65118 |
| Same measured training CPU, WikiText test | 3.59282 | 3.59570 | 3.65118 |
| Same500 updates, FineWeb out-of-domain | 3.92983 | 3.91634 | 3.91634 |
| Same measured training CPU, FineWeb out-of-domain | 3.87683 | 3.85698 | 3.91634 |

The reversible architecture had a small equal-data advantage over the ordinary model, but changed hidden-state geometry and residual scaling are confounds. Its equal-time WikiText difference was inconsistent across seeds. Inverse-based backpropagation was worse than both faster alternatives at equal CPU because it consumed more computation per update. Equal-time models saw additional training data; this is a separate budget comparison, not a claim of equal data and equal time simultaneously.

## Scope
Each primary model has91,296 parameters, four48-wide attention/MLP blocks, byte vocabulary256 and context64. Reversible states have two48-wide streams. Each equal-data run used256,000 training target-byte exposures. WikiText test evaluation scored8,192 bytes from52 articles; FineWeb scored8,192 bytes from87 documents. All selected windows were unique and had no exact full-window match in training. Three seeds and these small evaluation sets do not establish broad language or intelligence gains.

The experiment took109.915747 child-process CPU seconds and110.857168 wall seconds, under600/900-second caps. All subprocesses exited0 and were reaped, stop=null, no GPU, no new rental, no production changes.

## Decision
Keep reversible execution as a memory-versus-compute option, not a demonstrated intelligence or learning breakthrough. There is no evidence here to justify a600B run. A next study would need a tuned stronger baseline, residual-scale/state-geometry controls, longer sequences, more training and held-out data, GPU/mixed-precision stability tests, and downstream task evaluation.

See REPORT.md for all seeds and standard deviations, protocol_frozen.json for hashes and selections, correctness.json for numerical errors, and execution_receipt.json for the actual resource accounting.

## Reproduce
Use Python3.12, PyTorch2.6.0 CPU and NumPy. No corpus text is redistributed. Use the pinned WikiText and FineWeb manifests; prepare_data.py can download verified original source bytes with PyArrow. Set REV_WIKITEXT_DIR to the verified WikiText text directory and REV_FINEWEB_ROWS to the verified FineWeb rows JSON. The latter viewer response is not a durable archive: stop if its frozen hash does not match.

Run checks.py; compute_accounting.py; study.py freeze; memory_probe.py for each ordinary/reversible_graph/reversible_inverse mode and depth1,4,12; study.py run; report.py. Set CPU thread counts to1 and use an otherwise idle machine. Freezing creates a new source/hash record before the reproduced run; the published protocol_frozen.json records the completed original run. No additional research computation is required to inspect existing results.

Checkpoints contain model parameters, AdamW optimizer tensors, generator state and step count, not corpus text. They are small91,296-parameter research models, not chat models. All18 checkpoints preserve the three seeds, three arms and equal-data/equal-CPU endpoints. Load with torch.load(..., map_location="cpu", weights_only=True). They belong to the existing repository license; publication does not alter any license, permission or visibility.

No conclusion about human or general intelligence follows from this small byte-language benchmark.
