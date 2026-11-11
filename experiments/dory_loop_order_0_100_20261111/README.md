# Dory-inspired recurrent-order NAT: split versus whole, inference counts 0–100

This is an independently implemented tiny non-autoregressive (NAT) motif, not an NVIDIA Dory checkpoint or faithful architecture replication. Six models (two orders × three paired seeds) each store 764,304 parameters. All were trained at r=2. A 96-byte prefix conditions 32 blank suffix slots; all 32 output byte distributions are produced in parallel. Hidden-state recurrence does not feed predicted suffix bytes back into later outputs, perform diffusion, or perform adaptation.

The model has 8 input blocks, 20 unique recurrent-middle blocks and 8 output blocks. Split runs A repeated r times, then B repeated r times; whole runs (A,B) repeated r times. At r=0 only the middle is skipped: the 16 outer attention blocks still mix the prefix. Total layer applications are 16+20r, including 56 at the trained r=2 and 2,016 at r=100. Every other count is inference extrapolation outside the trained loop count, not a newly trained or validated model. Reinjection is omitted equally in both arms.

## Actual completion and reports

All six standard evaluation receipts are completed; all 18 full checkpoint hashes match training receipts and the 12 frozen source/protocol files are unchanged. Standard evaluation uses 256 validation, 10,059 WikiText-test and 2,134 FineWeb windows per checkpoint. Matched 2,000-update, 300-training-CPU-second and 900-training-CPU-second results are separate comparisons, not interchangeable estimands. Summed training update CPU time is 5401.134206 seconds; it excludes evaluation and other overhead.

The original sweep recorded blocked_no_fit before any held-out scoring; its three unchanged synthetic receipts are retained. A separately frozen runtime-only amendment is finalized with 1248 planned cells, 8 windows/domain at all 101 counts and 64 windows/domain only at 0,2,100. Recorded cell statuses are {"completed": 1248}. Finalized does not mean every cell succeeded. Its combined runtime debit is 3343.732987 seconds, including 81.068509 seconds from the original attempt. The original and amended attempts share the same planned 7,200-second budget rather than a new allocation; the report records any cap or overshoot.

- [Standard topology report](dory_loop_compare_20261111/reporting/topology_final/REPORT.md) and [HTML](dory_loop_compare_20261111/reporting/topology_final/REPORT.html).
- [Bounded inference-sweep report](dory_loop_compare_20261111/reporting/bounded_sweep_final/REPORT.md), [original blocked-attempt disclosure](dory_loop_compare_20261111/reporting/sweep_final/BLOCKED.md), and [frozen amendment](dory_loop_compare_20261111/BOUNDED_AMENDMENT.md).
- [Frozen training protocol](dory_loop_compare_20261111/protocol.json), [sweep protocol](dory_loop_compare_20261111/sweep_protocol.json), [file manifest](FILE_MANIFEST.json), [full-checkpoint manifest](CHECKPOINT_MANIFEST.json), [provenance](PROVENANCE.json), and [rights notices](THIRD_PARTY_NOTICES.md).

Quality remains limited: all 18 standard checkpoints have WikiText-test byte accuracy 18.956283% and validation accuracy 18.994141%, exactly matching the corresponding always-space target-frequency controls. The [post-hoc data-only diagnostic](dory_loop_compare_20261111/target_frequency_diagnostic.json) verifies the same scored cohort hashes. It does not measure model prediction distributions, so this equality does not prove every model output is a space. All standard checkpoints have zero exact 32-byte completions on each evaluated domain. No practical text-generation or intelligence gain is established.

## Main findings from this completed run

At the same 2,000 training updates, split versus whole averaged 4.57376 versus 4.57228 bits/byte on WikiText and 4.58043 versus 4.57926 on FineWeb. The small paired differences change sign across seeds, so this does not establish a clear practical topology winner.

On the separately reported 64-window/domain inference checks, increasing the count from 2 to 100 changed three-seed mean bits/byte as follows (lower is better):

| Topology | WikiText, 2 to 100 | FineWeb, 2 to 100 |
| --- | --- | --- |
| Split | 4.56664 to 4.57834 | 4.52922 to 4.53482 |
| Whole | 4.56447 to 4.58015 | 4.52880 to 4.53816 |

Those are mean degradations, not unanimous seed outcomes: one split/FineWeb seed improved. Paired seed ranges and every trace remain in the machine-readable reports. Byte accuracy was unchanged between 2 and 100, and exact 32-byte matches remained zero. Mean paired forward-wall ratios across the four topology/domain groups ranged from 35.57x to 36.78x. These times sum forward work over each 64-window cohort, not robust request latency. Thus extra inference repetitions did not establish useful generation gains in this r=2-trained tiny NAT prototype. They do not establish how NVIDIA's pretrained, differently trained Dory model behaves.

Use the actual reports for scores, paired differences, all seed traces, coverage and timing limitations. Three fitted seeds support descriptive comparisons, not universal architecture rankings. Sweep cohort sizes differ by phase and are never blended. The amendment was selected using already recorded synthetic timing, not held-out quality. The original attempted scope included 128-window checks at seven counts but was blocked without scoring; the amended scored scope is 8 windows at all counts plus 64 at only 0,2,100. Missing/capped/nonfinite cells are not zero scores. Forward runtime is summed over a cohort, not standalone request latency. No best loop count is declared validated. There is no 600B-scale, GPU-speed, cost-saving, or general-intelligence claim.

## Bundle layout and historical readiness dependency

- `dory_loop_compare_20261111/`: all 12 frozen source/protocol files, their source_hashes.json, six runs with training/evaluation/result JSON and three original full checkpoints each, paired checks and correctness/completion log.
- `dory_loop_compare_20261111/bounded_*`, BOUNDED_AMENDMENT.md and test_bounded_runtime_plan.py: separately frozen amendment sources/protocol, source hashes, calibration binding and static research audits. Original source identities are unchanged.
- `dory_loop_compare_20261111/target_frequency_diagnostic.py` and `.json`: post-hoc data-only target-frequency control, with source/cohort hashes and no text/token arrays. The source requires locally regenerated caches; the packager does not execute it.
- `dory_loop_compare_20261111/reporting/`: stdlib-only report generators, tests/templates/docs, actual final reports and SVG plots. Synthetic QA fixtures are excluded.
- `dory_loop_compare_20261111/sweep_results/`: the original three synthetic-only calibration, no-fit selection and blocker receipts, preserved unchanged.
- `dory_loop_compare_20261111/bounded_sweep_results/`: actual amended numerical per-window losses, document-group/selection indices, amendment/standard-evaluation provenance, cohorts and finalized status. No corpus strings or token arrays.
- `frontier_research_v8/` and `frontier_research_v31/`: pinned public dataset download scripts and manifests only.
- `nat_long_20261111/`: five unchanged, already-public historical NAT completion/evaluation/timing/summary receipts. These are intentionally retained because frozen control.py requires those sibling paths. They are not new Dory results. Exact copied hashes and the earlier public evaluation-redaction provenance are in PROVENANCE.json. No historical NAT checkpoint or full experiment is needed.
- `audit_training_pairs.py` and `test_audit_training_pairs.py`: stdlib-only helper and synthetic fixture tests. The helper derives a fresh-run training-pair audit from actual receipts/checkpoint hashes and refuses any existing output. Its hash is in PROVENANCE.json.

The nested original prepared README is intentionally excluded. Protocol status strings are frozen historical planning text; current completion is established by actual result/status artifacts, not by rewriting the protocol. The repository's existing license and permissions remain unchanged. An offline staging manifest is not a publication receipt.

## Reproduce from scratch

Use Linux, Python 3.12, the official PyTorch 2.6.0 CPU wheel (measured 2.6.0+cpu), NumPy 2.3.5 and PyArrow 19.0.1. Report generation uses only Python's standard library; a browser or SVG renderer is optional for viewing figures. No Matplotlib is required. Install torch from its official CPU wheel index and the other versions in requirements.txt. CPU affinity ID 1 must exist. Changing affinity/hardware affects timing, and editing any frozen source/protocol requires a newly identified experiment rather than silently updating archived hashes.

1. Work in a fresh sibling-layout copy. Retain the 12 source/protocol files and source_hashes.json, separately frozen bounded sources and bounded_source_hashes.json, target_frequency_diagnostic.py, reporting sources, dataset script/manifest directories, public pair-audit helper, and the five *actual historical public* NAT receipts at sibling `nat_long_20261111`. Do not copy runs, caches, sweep_results, bounded_sweep_results, bounded_calibration_receipts.json, final reports, diagnostic result JSON, old log/summary/audits, or locks into this fresh execution copy. Do not fabricate readiness receipts or patch out control.py's gates. Keep the archived release untouched.
2. From `frontier_research_v8`, run `python prepare_data.py`. From `frontier_research_v31`, run its `python prepare_data.py`. Read the rights notices first. Source bytes are fetched directly, hashed and checked; no corpus is bundled. Exact FineWeb retrieval depends on the pinned viewer snapshot still being available and fails closed on changes.
3. From `dory_loop_compare_20261111`, run `python run_all.py --go-after-nat-complete > pipeline.log 2>&1`. This executes the correctness check, six serial fits, then all six standard evaluations; it rejects existing run directories. Run no competing training/timing job. The explicit flag and historical receipts are required; this command itself authorizes no upload or payment. To continue only an interrupted evaluation after inspecting completed fit receipts, use `python train.py --go-after-nat-complete --evaluate-only --kind split --seed 1` with the actual intended kind/seed, serially. Never restart existing fits blindly.
4. Once every result is completed, run `python summarize.py`, then `python ../audit_training_pairs.py --input . --write`. The extra completed_pair_checks.json in this archive is a historical audit receipt, not an automatically generated output of run_all.py. The helper derives a new audit from the new run before using report_topology.py. Do not carry the archived audit into a new experiment as evidence. Keep the helper at the bundle root when making the fresh sibling-layout copy.
5. After standard evaluation and other timing have finished, run `python sweep.py --go-after-nat-complete`. It refuses an existing sweep_results directory, trains nothing and retains explicit capped/nonfinite/missing states or a pre-heldout blocker. Its synthetic calibration can block before loading held-out data. Do not rerun just to replace unfavorable outcomes. The archived run blocked_no_fit. Only if the fresh original attempt likewise blocks_no_fit before held-out access, bind its own new synthetic receipts with `python bounded_runtime_plan.py --prepare-receipts sweep_results --output bounded_calibration_receipts.json`, then run `python bounded_sweep.py --go-after-nat-complete --calibration-dir sweep_results --calibration-receipts bounded_calibration_receipts.json`. This retains the original attempt's actual elapsed debit and original total budget. If the original scope fits on different hardware, this no-fit amendment does not apply. Use your configured Python environment; the environment-specific interpreter path retained in the frozen amendment document is historical, not an additional bundled dependency.
6. Optionally reproduce the post-hoc data-only control after local evaluation caches exist with `python target_frequency_diagnostic.py` in the fresh experiment directory. It reads licensed local data and numerical receipts, does no model inference, and writes a new diagnostic. Never run it merely to overwrite the archived result.
7. Generate reports from completed numerical artifacts with the commands below. Do not confuse output from tests with results. No model or dataset is loaded by the report generators.

## Regenerate reports from the archived release (no model compute)

From the bundle root, run:

    python dory_loop_compare_20261111/reporting/report_topology.py --input dory_loop_compare_20261111 --output dory_loop_compare_20261111/reporting/topology_regenerated
    python dory_loop_compare_20261111/reporting/report_bounded_sweep.py --input dory_loop_compare_20261111/bounded_sweep_results --output dory_loop_compare_20261111/reporting/bounded_sweep_regenerated

The topology reporter hashes the original full checkpoint files without deserializing them. The bounded reporter requires finalized amended status, matching cohort/amendment provenance and the retained original synthetic-only receipts. Its CLI is documented in its bundled template/source. The original report_sweep.py is retained for the unchanged original protocol but cannot produce a numerical report from the archived original no-fit blocker. Report outputs must be new directories. Tests are `python -m unittest discover -s dory_loop_compare_20261111/reporting -p 'test_report_*.py'`; they use synthetic fixtures, never experimental models/corpora.

## Checkpoint/security and rights scope

The 18 `.pt` files are the original full checkpoints, including optimizer and NumPy/Torch RNG state, copied without model-only conversion so frozen hashes/loaders remain valid. Verify hashes first and load only trusted files. Frozen evaluation scripts use `torch.load(..., weights_only=False, map_location='cpu')`; loading untrusted pickle checkpoints can execute code. The packager uses zipfile/pickletools string/opcode inspection without deserialization and does not guarantee arbitrary pickle safety. No NVIDIA pretrained weights, raw source text, data caches, private source, credentials or private backup artifacts are included.

Licensing caveats, especially the unresolved WikiText license-metadata inconsistency and FineWeb underlying-text rights, are preserved in THIRD_PARTY_NOTICES.md. No new license or permission is granted by this bundle.

## Derive a training-pair audit for a new run

From the bundle root, run `python audit_training_pairs.py --input dory_loop_compare_20261111 --check` to verify the new run without writing, then use `--write` to create completed_pair_checks.json once. The helper verifies current frozen source/protocol identities, six complete training receipts, all 18 checkpoint SHA-256s, paired initialization and every equal-step batch-stream hash. It reads no models, corpus or held-out scores and never creates/replaces historical NAT readiness receipts. It refuses an existing pair audit; inspect that file rather than overwriting it.

Run helper fixture tests from the bundle root with `python -m unittest test_audit_training_pairs`. These tests use synthetic receipts and tiny fake byte files, never experimental checkpoint tensors, models or corpora.
