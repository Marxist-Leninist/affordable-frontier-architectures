# NAT comparison report template (not experimental results)

Status: final output is gated on all planned fits and evaluations. No provisional numerical conclusion.

1. Scope: 765,568-parameter Transformer versus 765,184-parameter optimized bidirectional Mamba-1; FP32 CPU, three layers, 96-byte prefix and fully blanked 32-byte suffix, one forward pass. Not diffusion.
2. Matched-data quality: [three-seed WikiText and FineWeb suffix bits/byte, byte accuracy, exact-match]. 16,000 updates, identical per-seed data order; report paired hash verification.
3. Matched CPU-time quality: [first update crossing 900 CPU seconds, actual overshoot, updates and metrics for every seed]. Missing checkpoints stay unavailable.
4. Uncertainty: [conditional paired-document 95% intervals] separately from [across-seed sample SD and individual differences]. No interpolated held-out test curve or generic significance claim.
5. Learning curves: every recorded validation observation for all seeds versus updates, training CPU time and wall time.
6. Timing: concurrent-run CPU/wall/end-to-end, compiler/cache load, separate adaptation time; update timing excludes batch generation/hash, logging, validation, saving and compilation, and is not end-to-end wall; serial whole-model training and inference p50/p95. Combined timing-process RSS is not per-model memory.
7. Synthetic tasks: [copy, reverse, sort, prefix cumulative sum modulo 10] after separate 2,000-step adaptation. Not zero-shot or general intelligence.
8. Optimization provenance: custom linear-work CPU affine scan; block microbenchmark improvements are not whole-model or GPU speedups. Architecture reference: Mamba commit e9594ce1c732d97440f0332fdc43170a2294dbfa, Apache-2.0; independent implementation, private AGILLM source not redistributed.
9. Data and limitations: pinned sources and cohort selection; exact-window exclusions and residual target overlap; FineWeb was used in prior research but not tuning here. Corpus and generated text excerpts excluded. Previously inspected WikiText metadata reports CC-BY-SA-3.0/GFDL while other licensing prose links CC-BY-SA-4.0: retain this inconsistency and original-source rights; never redistribute the corpus. FineWeb dataset card ODC-BY, third-party underlying rights may apply.
10. Selection caveats: both LR winners are the lowest grid point .0005. Primary objective is text-only, unlike pilot mixed-task training. Three seeds and bounded search do not prove convergence, intelligence, GPU speed or 600B-scale benefit.

Run only after all results exist:

    python generate_report_long.py --check-only
    python generate_report_long.py

Outputs: report/REPORT.md; learning_curves.png; quality_vs_time.png; paired_uncertainty.png; standalone_timing.png; task_adaptation.png; report_input_hashes.json.
The generator reads JSON metadata/results and frozen source bytes for integrity checks only; it does not import model/data code, open checkpoints/corpora, evaluate models, publish or upload.

11. Static partial forward MAC accounting at B=1,L=128,D=128,three layers,output32: Transformer 103,088,128 matrix MACs; bidirectional Mamba 86,788,096 matrix/convolution MACs; separately 3,145,728 Mamba state elements updated. Omits normalization, activations, RoPE, softmax, scan and pointwise work; not equal FLOPs or evidence Mamba does less total work. Matching parameters is not matching compute. Both dense arms invoke every layer module, with indexed embedding rows and no expert router. See compute_accounting_long.json and measured runtime.
