# Standard topology reporting

This reporter is preparation for the completed standard-checkpoint comparison. It performs no training or inference and imports no model/data-loader module. It reads JSON receipts and hashes checkpoint bytes without deserializing them. It never opens raw corpus files or writes outside a new directory under `reporting/`.

Run only once all standard evaluation jobs finish:

```sh
python reporting/report_topology.py \
  --input . \
  --output reporting/topology_final
```

Run from the experiment directory, or use absolute paths. A nonzero exit status and a `blocked` JSON message mean no comparison was produced. Incomplete evaluations cannot be overridden. The output directory must not already exist. A report is created only after the following checks all pass:

The experiment's original prepared `README.md` is not a reporting input and is not included in input provenance. It may be omitted or replaced with a public README without changing report output.

- All six `result.json` files have status `completed` and record three evaluated checkpoints each.
- The frozen source manifest still matches source bytes and every training receipt, and the protocol/topology/counts match this experiment.
- All 18 checkpoint files exist and match their recorded SHA-256 hashes. Every evaluation entry matches the corresponding training checkpoint metadata.
- Same-seed initialization hashes and all equal-step batch-stream hashes match between arms.
- Evaluation files contain exactly three checkpoint entries and all three domains with the expected 256/10,059/2,134 windows, finite metrics and complete per-window losses/document IDs.
- Evaluation manifests, selection hashes and ordered document IDs are identical across every arm, seed and checkpoint. Aggregate losses/bits-per-byte agree with the saved per-window values. Accuracy denominators are feasible.
- Input receipt/source hashes remain unchanged through collection. Identity checks compare saved receipts rather than reconstructing training or rereading raw corpus text.

The output includes `REPORT.html`, `REPORT.md`, 12 standalone SVG figures, and CSV/JSON tables for all components, paired differences, summaries, checkpoint timing, timing summaries and full-run timing. Source/checkpoint/input identities are in `provenance.json`; output hashes are in `artifact_hashes.json`. These files contain no raw corpus text. Full precision is retained in machine-readable outputs; percentages and paired percentage-point differences are labeled in the human report.

Matched 2,000-update, 300-training-CPU-second and 900-training-CPU-second results are separate. Each domain is separate. Means, sample seed SD and observed seed range are descriptive for three fitted pairs, never a confidence interval over architectures. No bootstrap, composite score, outcome-dependent checkpoint selection or sweep inference is performed. Timing sums update work, not full process wall time. Existing WikiText/FineWeb cohorts are explicitly described as reused.

## Fixture tests and figure QA

```sh
python -m unittest discover -s reporting -p test_report_topology.py -v
```

Tests create synthetic JSON and fake checkpoint byte files only inside a temporary reporting subdirectory, then remove it. No PyTorch, source model imports, corpus reads, training or inference are involved. Tests cover fail-closed partial states, identities, finite/consistent metrics, deterministic generation, valid SVG XML and output confinement. The fixture values are not experimental results and must not be published as such.

For visual QA, render the generated SVGs in a browser or SVG renderer. Images use explicit dimensions/view boxes, descriptive titles and text labels. Any retained fixture preview must be clearly marked as synthetic. After real generation, inspect `REPORT.html`, the paired-difference figure and the three exact-match panels, including the all-zero case.

`report_sweep.py`, its template/tests, frozen protocol, training source and `source_hashes.json` are unchanged. The 0–100 inference-count sweep is a separate report from r=2-trained checkpoints and may use different evaluation cohorts; do not combine its scores with this standard comparison.
