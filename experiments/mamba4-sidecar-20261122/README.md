# Unofficial Mamba-4 idea: bounded, discretionary exact memory
This is a small public research pilot, not a Mamba-team release, official Mamba-3 kernel result, or natural-language benchmark.

The measured core is a learned selective diagonal recurrent proxy h_t = sigmoid(g(x_t))*h_(t−1) + (1−sigmoid(g(x_t)))*u(x_t), with query-conditioned learned output head. For every seed the constant base goes unchanged into all retrieval arms; references store recent explicit writes from the synthetic generator, never tokens from assistant/user memory. This is not a Mamba-3 implementation, and the improvement is **sidecar exact lookup**, not improved latent recall. An exact record's returned value always replaces its corresponding classifier's guess only on a hit validated in the current epoch. A bounded FIFO may miss or forget older events. The intentionally unsafe epoch-free ablation is not eligible. Observation at input contains only actual encoded record fields/fillers/resets and a key query, never gold answer, scenario family, hidden state labels or answers. The gate sees classifier confidence, margin, entropy and modeled header signals, not the result from probing the ring. Stored records are key, value, epoch, local time. Per-run sequence lengths are <=128 and within uint16 maxima. Do not silently extrapolate this fixed fixture to unrestricted sequence counters.

## Results
3 seeds, 1,200 disjoint test episodes per seed per split (3,600 ID and 3,600 distribution shift). Training 960 matched proxy updates on length 32. Gate training and decision cutoff fit on split-independent 624 fit/624 dev episodes per seed, length 48. Test ID uses length 48 and new RNG; OOD length 128, additional/shifted distractors, independently generated. Six episode families are balanced: local, far, overwrite, absent, stale, evicted. Constructed record values and query are synthetic. Gates were calibrated to ~40% development read rate, not test labels. Secondary offline equal 40% test calls identifies discrimination but is not an executable streaming promise (requires batch score ranks).

| Method | ID accuracy | ID read | OOD accuracy | OOD read |
| --- | ---: | ---: | ---: | ---: |
| Recurrent 48 | 69.78% | 0 | 69.19% | 0 |
| Recurrent 56 | 71.11% | 0 | 69.53% | 0 |
| Confidence | 77.61% | 41.94% | 77.47% | 36.78% |
| Heuristic | 77.47% | 41.83% | 77.06% | 36.28% |
| Learned | 78.17% | 39.56% | 80.42% | 36.28% |
| Always exact | 79.39% | 100% | 82.17% | 100% |

Normalized utility = 100*(accuracy−0.025*read_rate)/best utility in split; 0.025 is a declared hypothetical probe cost, **not measured savings**. ID: learned100 > always99.63 > confidence99.20 > heuristic99.03 > wide92.14 > recurrent90.41. OOD: always100 > learned99.80 > confidence96.09 > heuristic95.58 > wide87.27 > recurrent86.85. At exactly 40% calls, shifted learned80.94% > confidence78.17% > heuristic77.86% > random74.28%. The learned gate recovers 404/467 shifted improvements with 1306/3600 probes; 838 of those probes found no current record. Even exact policy cannot undo base false positives on misses (115/3600 shifted, 195/3600 ID); a safety-critical consumer should not interpret a model-only guess as verified retrieval. On ID the unsafe epoch-free arm makes 272 false-present outputs vs 195 safe; safe sidecar returned zero incorrect hits over all test episodes.

## Bytes, speed and scope
Proxy state 192 raw bytes per user stream, sidecar and header 31 additional bytes. Width56 recurrence-only state 224 (>=223 for original state+archive, excludes shared weights and transient activations), trained on identical stream/updates; wider model also has more parameters. Shared gate fitted stats+weights 224 bytes, not per-stream. Python runtime object overhead is not measured. State claims do not include checkpoint, tool or wrapper runtime memory.

Actual matched prepared-state NumPy/Python median OOD overhead on this CPU: always probe ~0.562 µs/query across seed medians versus learned feature+gate+requested probes ~0.782 µs. ID medians ~0.558 vs ~0.737. Model inference, sidecar writes, feature availability overhead outside the measured wrapper (no remote access), and serialization are not included in those query overhead numbers. Core+archive setup separately in results. **The gate reduces probes but was slower on this tiny local table.** For an always-resident four-entry ring, prefer reading it every time if maximizing tested accuracy/speed. To justify a learned gate use a truly expensive lookup setting with measured wall-clock gains and mixed absence/cost tests.

## Verification
After install-free NumPy run, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pilot.py --verify` and separately `python wide_control.py --verify` independently reload SHA-bound exact checkpoints and replay complete non-timing metrics. `python timing.py` checks output accuracies and alternates arms across 15 host timings per split/seed.

## Checkpoints and evaluation
File `model_17.npz` is the default example (lowest numbered seed, not test-selected). All three proxy and three width-control checkpoints are included; every result reports all three seeds. These NPZ files contain only finite numeric NumPy arrays as shown in MODEL_CARD.md. Load using `numpy.load(path,allow_pickle=False)`; the included verifier checks file SHA-256 and reconstructed held-out metrics. Hashes are pinned in SHA256SUMS. This release has no weights or code from the upstream Mamba repository, any P057/P059 model, external datasets, or harvested text.
