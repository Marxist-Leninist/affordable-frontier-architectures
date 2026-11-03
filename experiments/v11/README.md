# v11: stronger Transformer succeeds on harder multi-fact recall

2026-11-03. Final held-out replication is complete and independently audited.

## Useful result

Across all three fresh seeds, the two-block local/global Transformer achieves **100% accuracy in every test cell**: 4, 8 or 16 independent key–value facts spread over 512, 2,048 or 4,096 synthetic symbols.

The parameter-matched vanilla tanh RNN remains near the **6.25% chance baseline**, including within its trained context range. Mean cell accuracy ranges from 4.95% to 6.90%.

| Facts | 512 symbols: Transformer / RNN | 2,048 symbols | 4,096 symbols |
|---|---:|---:|---:|
| 4 | 100% / 6.25% | 100% / 6.38% | 100% / 5.86% |
| 8 | 100% / 6.77% | 100% / 6.38% | 100% / 6.90% |
| 16 | 100% / 5.60% | 100% / 4.95% | 100% / 5.73% |

Each cell uses 256 balanced held-out cases per seed. These are synthetic vocabulary symbols, not words or ordinary language-model tokens. Training includes lengths through 512; 2,048/4,096 are extrapolation.

**Conclusion:** the earlier single-symbol recurrent success does not establish general long-range superiority. This stronger, position-generalizing Transformer solves the harder task after adequate training. The RNN failed to learn under this protocol; this is not proof of an intrinsic capacity limit or a result about all recurrent architectures.

## Fairness and limits

- Same 6,000-step curriculum, batch 8, 11,129,856 input-symbol presentations and 48,000 supervised queries per run. Three initialization seeds per architecture; data are shared across seeds.
- Transformer 18,992 parameters; RNN 18,928. Float32, deterministic CPU PyTorch 2.6.0, one thread.
- Learning rates selected only from final-step validation loss in a separate pilot: Transformer .003, RNN .001. No test-based tuning, best-seed selection or epoch cherry-picking.
- All six final checkpoints saved and fresh-model reload-verified before scoring. Independent audit checked file/tensor hashes, source/protocol hashes, test datasets, every saved prediction and all 54 seed/cell accuracies.
- The Transformer is a task-specific local adjacent-token block plus global attention, without absolute position embeddings. It retains sequence-length-dependent representations; RNN state is fixed-size. This is not equal-inference-memory or equal-FLOP testing.
- Four/eight/sixteen independent 4-bit values make the map harder than v9's single 3-bit symbol, but this remains synthetic retrieval, not document comprehension, coding, general intelligence or frontier capability.
- Initial v10 training and a validation-only continuation failed for both models. Those failures are preserved rather than overwritten. The curriculum supplied training adequacy; it does not establish globally optimal tuning for either family.
- No measured API-price advantage or profitable $500 subscription claim follows from these tiny models.

## Reproducible evidence

[Full source, frozen protocols, input ledgers, results and independent audits](reproducibility.tar.gz) are preserved in a lossless archive, together with the earlier failed controls and validation-only pilot. Binary model checkpoints are separate artifacts; checkpoint hashes and reload reports are included.

Archive: 472,767 bytes. SHA256: b0fbdd584eb2cb59f999753da822a8ec111aff9b7525a1d0cb3877ae555920a4.

Extract the archive and follow REPRODUCE.txt. With CPU PyTorch 2.6.0 and NumPy installed, the final run can be repeated from frontier_research_v11_final using PYTHONPATH=../frontier_research_v10 python run_final.py. The bundle retains the original directory layout needed by the frozen runner.

The validation-only extension's non-executable header comment was omitted for standalone publication; its publication manifest records original/published hashes and unchanged executable AST. No executed training code or historical results were silently changed.
