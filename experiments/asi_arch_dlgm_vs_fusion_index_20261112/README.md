# DLGM #458 vs Fusion #879: one combined index (12 November 2026)

Two ASI-Arch candidates compared on seven lines of evidence, including both under the AGILLM training objective, and combined into one number each. The weights and the scoring rule were frozen (`index_spec_v1.json`, sha256 `fda0f7ad167e19c11d51a5e091af9ad8ebbd59c6bf6bca5d16967602a8af0140`) at 2026-11-12T01:59:02Z, before the Fusion run under this objective had produced a result.

Live page (rebuilds itself from the synced runs): https://zqx.lat/asi-arch/dlgm-vs-fusion/

## Result

Fusion #879 is on top: 85.5 to 74.3, a lead of 11.2 points. It wins 86% of 20,000 random re-weightings of the same evidence.

| | DLGM #458 | Fusion #879 |
|---|---:|---:|
| **Index (0 to 100)** | **74.3** | **85.5** |

**Read this with the result.** The frozen formula scores the two main own-objective lines with each candidate in its faithful form. Only a cut-down form fits at 560 layers. The same index with other forms on those two lines:

| Form on the two own-objective lines | DLGM gain (nats) | Fusion gain (nats) | Seeds | Index DLGM | Index Fusion |
|---|---:|---:|---|---:|---:|
| low-rank form | +0.117 | +0.102 | 3 and 3 | 80.6 | 80.7 |
| low-rank gate, per-channel kernels | +0.120 | +0.112 | 3 and 3 | 80.6 | 83.0 |

## The seven lines

| Evidence | Weight | DLGM | Fusion | Gain over the shared baseline |
|---|---:|---:|---:|---|
| Your objective, same number of training steps | 35% | 81.9 | 100.0 | simulator, faithful form: DLGM +0.1108, Fusion +0.1352 (nats) |
| Your objective, long context | 5% | 100.0 | 93.9 | simulator, faithful form: DLGM +0.2495, Fusion +0.2343 (nats) |
| Your objective, same simulator box time | 10% | 0.0 | 0.0 | simulator, low-rank form: DLGM -0.0470, Fusion -0.1505 (nats) |
| Your matched small-model run (v65 R6) | 15% | 100.0 | 81.0 | wikitext: DLGM +0.0677, Fusion +0.0587; fineweb: DLGM +0.0847, Fusion +0.0637 (nats per byte) |
| ASI-Arch paper: benchmark tasks | 15% | 70.6 | 97.3 | 20M parameters: DLGM +1.1400, Fusion +2.7700; 340M parameters: DLGM +2.0300, Fusion +1.9200 (accuracy points) |
| ASI-Arch paper: training loss | 10% | 100.0 | 90.6 | 20M parameters: DLGM +0.1778, Fusion +0.1445; 340M parameters: DLGM +0.0721, Fusion +0.0720 (nats) |
| Serving readiness | 10% | 50.0 | 100.0 | rubric: DLGM 50, Fusion 100 (rubric) |

Scoring: For a gain component each candidate scores 100 x max(gain, 0) / max(gain of DLGM, gain of Fusion, floor). The better one gets 100 when its gain clears the noise floor, the other gets its share of that gain, and a candidate that does not beat the shared baseline gets 0. Where a component has sub-experiments they are scored separately and averaged. Index = weighted sum of the component scores (0 to 100). The name on top is the higher index. A margin under 3 points is called level in words.

## Under the AGILLM objective (simulator `agillm_mini.py`, 1,200 steps, held-out loss, lower is better)

| Run | Seeds | AR | SAT | NAT | Gain, equal steps (nats) | 4x context | ms per step | Gain, equal simulator time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Live block (base, base_r2, base_r3) | 7 | 5.2429 | 5.4642 | 5.8446 | 0 | -0.316 | 202 | 0 |
| `dlgm458` | 3 | 5.0965 | 5.3806 | 5.8133 | +0.111 +/- 0.007 | -0.067 | 461 (x2.37) | -0.003 |
| `dlgm458_lr` | 3 | 5.0901 | 5.3797 | 5.8043 | +0.117 +/- 0.002 | -0.089 | 543 (x2.79) | -0.047 |
| `dlgm458_lrk` | 3 | 5.0851 | 5.3723 | 5.8081 | +0.120 +/- 0.011 | -0.071 | 412 (x2.12) | +0.013 |
| `fusion879` | 3 | 5.0718 | 5.3597 | 5.7861 | +0.135 +/- 0.004 | -0.082 | 370 (x1.90) | +0.032 |
| `fusion879_lr` | 3 | 5.1074 | 5.3908 | 5.8154 | +0.102 +/- 0.013 | -0.118 | 744 (x3.82) | -0.150 |
| `fusion879_lrk` | 3 | 5.0983 | 5.3830 | 5.8000 | +0.112 +/- 0.004 | -0.088 | 393 (x2.02) | +0.008 |

Gain = live block minus candidate on 0.6 AR + 0.2 SAT + 0.2 NAT; +/- is one standard error across seeds. "4x context" is the change in loss on the last 256 tokens when the 768 tokens before them are also visible. "Equal simulator time" uses the site method (step cost = 0.75 x bytes streamed + 0.25 x measured ms per step against the base run of the same day).

## How firm

- Random weightings (20000 draws, uniform on the simplex): DLGM ahead in 14%, Fusion in 86%.
- Leave one line out: without "Your objective, same number of training steps": Fusion #879 on top (70.1 vs 77.7); without "Your objective, long context": Fusion #879 on top (72.9 vs 85.0); without "Your objective, same simulator box time": Fusion #879 on top (82.5 vs 95.0); without "Your matched small-model run (v65 R6)": Fusion #879 on top (69.7 vs 86.3); without "ASI-Arch paper: benchmark tasks": Fusion #879 on top (74.9 vs 83.4); without "ASI-Arch paper: training loss": Fusion #879 on top (71.4 vs 84.9); without "Serving readiness": Fusion #879 on top (77.0 vs 83.9).
- The matched small run (v65 R6) is a statistical tie at 3 paired seeds; the paper rows are single runs; serving readiness is a rubric.
- Fusion retention is inert at this run length (upstream init 0.99996; see `fusion_state.json`), so the delta memory behaves like the plain delta rule of DLGM here.
- Both main ports use the block-wide output norm instead of the per-head norm of the originals; `fusion879_hn.py` and `dlgm458_hn.py` restore a per-head norm for both as a robustness check (results in `runs/` when present).
- `dlgm458.py` computes its memory in chunks of 16, Fusion needs 32: same mathematics for DLGM, different step time, so the speed columns are not strictly like for like.
- "Equal time" is the simulator cost model at width 256. At width 8192 an operation count puts either mixer at about 6M multiply-adds per token against about 800M for the binary matrices of a block and about 34M for attention over 2,048 tokens; neither mixer has been timed in the real trainer.

## Files

- `plugins/fusion879.py` (and the byte-identical `fusion879_lr.py`; the file name selects the low-rank form): ASI-Arch #879 (delta_net_psfr) ported to the simulator block with the conventions of `plugins/dlgm458.py`.
- `test_fusion879_equiv.py`, `fidelity_test_result.json`: the port against the original #879 class with copied weights (float64 on CPU, float32 on the GPU with the real FLA 0.2.2 kernels).
- `index_spec_v1.json`: the frozen formula. `build_index.py`: builds the page and `index.json` from the runs. `index.json`, `page/index.html`: snapshot at packaging time.
- `runs/`: every raw run JSON used (25 files). `census560.json`: storage bill of each form at 380 and 560 blocks.
- `agillm_mini.py`: the simulator as run (sha256 prefix `dc5c3c28ddbf1853`, frozen since 9 Nov).

## Reproduce

```
python3 -I agillm_mini.py --plugin plugins/fusion879.py --seed 1 --out fusion879__s1.json      # needs the simulator data set (train.npy, test.npy)
python3 test_fusion879_equiv.py                                                              # needs flash-linear-attention 0.2.2, upstream data/106.json, a GPU for the float32 phase
```

Upstream sources: GAIR-NLP/ASI-Arch at commit 3113c51b (data/106.json, entries #458 and #879); paper arXiv 2507.18074v2, Table 9.

Packaged 2026-11-12 04:08 UTC by Claude for Scott.
