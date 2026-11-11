# Bounded recurrent-order NAT inference sweep

This report concerns a **separate, prospective runtime-only scope amendment**, not successful execution of the original sweep plan. The original synthetic calibration stopped with `blocked_no_fit` before loading held-out data. Its original seven-count, 128-window/domain extra cohort **did not run**. No original-plan quality results are implied or substituted.

All six matched-2,000-update models were trained at r=2 only. This is exploratory inference-count extrapolation, not 101 training runs or a validated search for a best count. The scope amendment changed no model weights or training protocol and used no held-out quality outcomes to choose counts, windows or checkpoints.

## Original block and retained budget

The original plan tested runtime candidates of 32, 16 and 8 windows/domain across every count, each with an additional 128-window/domain cohort at counts 0, 1, 2, 4, 8, 32 and 100. None satisfied the prespecified synthetic runtime bounds. For its smallest candidate, the predicted global bound was **{{ORIGINAL_SMALLEST_CANDIDATE_GLOBAL_BOUND_SECONDS}} seconds**; the largest margined per-checkpoint prediction was **{{ORIGINAL_SMALLEST_CANDIDATE_MAX_MARGINED_CHECKPOINT_SECONDS}} seconds**, compared with a 1,050-second prediction allowance inside the 1,200-second per-checkpoint cap. These were runtime predictions, not completed evaluation durations.

The original **{{PRIOR_ELAPSED_DEBITED_SECONDS}} seconds** already spent are retained as a debit, including calibration and original setup. The amendment does **not** get a fresh 7,200-second allocation. It reuses the original SHA-256-bound synthetic timings without rerunning calibration. New setup debited **{{NEW_SETUP_DEBITED_SECONDS}} seconds** before held-out loading. The revised predicted global bound was **{{AMENDED_PREDICTED_GLOBAL_BOUND_SECONDS}} seconds** and the largest margined per-checkpoint prediction was **{{AMENDED_MAX_MARGINED_CHECKPOINT_SECONDS}} seconds**. The same hard 7,200-second combined budget and 1,200-second cumulative evaluation budget per checkpoint apply. Prediction includes a 1.5 safety factor and 600-second global reserve; this is a budget rule, not a guarantee of coverage.

Final recorded new execution time: **{{FINAL_NEW_ELAPSED_SECONDS}} seconds**. Combined prior-plus-new time: **{{FINAL_COMBINED_ELAPSED_SECONDS}} seconds**. Recorded combined-cap overshoot: **{{FINAL_COMBINED_CAP_OVERSHOOT_SECONDS}} seconds**. Global cap flag: **{{GLOBAL_CAPPED}}**. Individual checkpoint cumulative wall times and any observed overshoot appear in checkpoint_runtime.csv/json. A full forward call can overrun the time at which a cap was reached; do not conceal recorded overshoot.

## Amended cohorts and finalized coverage

- All-101 phase: **8 fixed evenly spaced windows per domain**, at every integer count r=0…100.
- Separate larger-cohort checks: **64 fixed evenly spaced windows per domain**, only at **r=0, 2 and 100**.
- Domains: WikiText-2 test and the existing FineWeb cohort. The underlying cohorts were reused from earlier studies, not newly collected held-out evidence for the wider research program.
- Models: split and whole topologies, seeds 1, 2 and 3, all using the same six matched-update checkpoints as the original planned sweep.

The amendment preserves all 101 counts and all six models, while reducing only the extra-cohort scope. Windows are selected without quality outcomes. The 8-window and 64-window phases are different cohorts; never blend their scores or assume every intermediate count received 64-window evaluation. Each evaluated cohort covers at least four documents. Eight windows/domain is extremely small and may not represent the source distribution.

The finalized ledger contains **{{CELLS}} planned count/seed/domain/phase cells** and **{{GROUPS}} arm/domain/count/phase summary groups**. Status totals: {{STATUSES}}. Complete three-seed groups: **{{COMPLETE_GROUPS}} / {{GROUPS}}**. Finalized means execution accounting ended, not that every planned cell completed.

Every planned cell is retained in cells.csv/json. Missing files/cells, capped cells, nonfinite logits/losses and malformed metrics are explicit and have null quality/timing metrics, never zero placeholder scores. The original cell status is preserved separately. Nonfinite status is instability, not ordinary missingness. Source finalization counts are checked against a complete available cell ledger. Deleted/incomplete files remain missing rather than silently dropping their planned cells.

Plots show all available seed traces and gaps for missing/invalid cells. Connecting lines are visual guides, not measurements at intervening selected counts. Means, sample seed SD and observed range use completed seeds only; completion counts distinguish partial groups. Partial-group means may be biased by missingness and should not be compared as complete results. Paired cells and paired summaries report split-minus-whole differences only where both seed-matched cells completed. Negative bits/byte differences favor split; positive accuracy differences favor split. These are descriptions of the fitted seeds, not architecture-population confidence intervals or significance tests. No document bootstrap, composite score, best-count selection or quality claim from unfinished execution is produced.

## Architecture and task interpretation

Split executes I A^r B^r O, where A and B are distinct ordered ten-layer middle groups. Whole executes I (AB)^r O. Both store **764,304 parameters** across **8 input + 20 middle + 8 output unique attention blocks**, embeddings and head, with width 48 and FFN width 117. Training r=2 uses **56 total layer applications**. General inference depth is **16+20r**: r=100 means 2,016 applications, not 2,016 distinct learned layers.

Zero means zero middle loops. All **16 outer attention layers remain active**, with **353,424 invoked module parameters**, retaining prefix conditioning. At positive counts all 764,304 parameters participate. Invoked counts refer to entire module tensors, not individual embedding rows. At r=1 the layer schedules coincide, but independently trained arm weights may differ. Reinjection is omitted in both arms.

The byte NAT objective is a **96-byte prefix and 32 fully blank output slots**. One parallel network call predicts all suffix bytes, with no gold suffix inputs, predicted-token feedback, teacher forcing or adaptive parameter updates during inference. Additional loops add hidden-state computation, not autoregressive token decoding or diffusion denoising. Training used a common **fixed base learning rate 0.0005**, 100-step warmup then constant rate, without separate architecture tuning.

This independent motif uses no NVIDIA checkpoint and does not replicate Dory normalization, weights, pretraining or implementation. It provides no general-intelligence claim, no 600B-model equivalence, no GPU performance forecast and no reliable large-model scaling conclusion. Many tested counts are far outside the r=2 training setting, and repeated exploratory comparisons do not validate a favorable count.

## Timing and reproducibility

Forward CPU/wall metrics are **sums of independent forward calls over the relevant cohort**. They exclude metric computation, loading, synthetic calibration and I/O. There is no cross-count latent reuse. These timings are not repeated warmed request-latency benchmarks. Nominal layer-operation counts match between arms at a fixed count, but not between different counts or between the 8- and 64-window phases. Actual runtime need not match.

After authorized execution is finalized, from the experiment root:

    python reporting/report_bounded_sweep.py --input bounded_sweep_results --output reporting/bounded_sweep_final
    python -m unittest discover -s reporting -p test_report_bounded_sweep.py -v

The default original-calibration directory is sweep_results; --original-calibration may specify its verified relocated copy. The reporter validates the finalized status before opening scores. It checks the original twelve frozen sources and separately frozen amendment sources, three original synthetic receipts, the original no-fit decision, retained runtime debit, amendment selection and cohort identities, matched-update checkpoint identities and paired document order. Receipt hashes of the earlier standard evaluations are retained; this reporter does not reload models, rehash checkpoint bytes, rerun those evaluations or open corpus text.

provenance.json contains input/source/checkpoint/cohort hashes and budget receipts. cells/summary and paired_cells/paired_summary are CSV/JSON; checkpoint_runtime records cumulative checkpoint evaluation times. artifact_hashes.json binds generated outputs. The original report_sweep.py remains unchanged and supplies only the SVG drawing function. Public outputs omit raw text, token arrays, per-window loss/group arrays, arbitrary manifest metadata, credentials and absolute local paths. Publication is a separate action and this generator uploads nothing. Exclude synthetic fixture reports and QA previews from research-result publication.

Architectural motivation: [NVIDIA's official Dory model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/blob/main/README.md). Existing code/data provenance and licenses still apply; this report assigns no new license and implies no NVIDIA endorsement.

## Per-seed plots

{{PLOTS}}
