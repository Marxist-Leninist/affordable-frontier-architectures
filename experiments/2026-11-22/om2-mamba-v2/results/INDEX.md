| rank | model | family | **combined** | quality (bpb) | exact recall | training speed (s/step) | decode (ms/token) | prefill | state memory at 256 tokens | training memory | bits gained per train minute |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mamba-2 | Mamba | **115.1** | 115 (2.178) | 108 (56%) | 58 (0.38) | 104 (2.3) | 62 | 716 (143 KiB) | 73 (1049 MiB) | 0.534 |
| 2 | Our Mamba v2, positive decay | Mamba (ours) | **112.6** | 116 (2.168) | 107 (56%) | 59 (0.38) | 87 (2.7) | 62 | 716 (143 KiB) | 73 (1049 MiB) | 0.468 |
| 3 | Our Mamba v2 | Mamba (ours) | **109.7** | 116 (2.168) | 108 (56%) | 57 (0.40) | 79 (2.9) | 57 | 716 (143 KiB) | 69 (1105 MiB) | 0.450 |
| 4 | v2 + placebo sidecar | Mamba (ours) + sidecar | **108.7** | 116 (2.163) | 107 (56%) | 56 (0.40) | 76 (3.0) | 56 | 716 (143 KiB) | 69 (1107 MiB) | 0.442 |
| 5 | v2 + state-gated exact store ("Mamba-4" idea) | Mamba (ours) + exact store | **101.6** | 117 (2.157) | 108 (56%) | 55 (0.41) | 75 (3.1) | 55 | 378 (271 KiB) | 68 (1113 MiB) | 0.438 |
| 6 | v2 + always-on exact store | Mamba (ours) + exact store | **101.0** | 117 (2.153) | 108 (56%) | 55 (0.41) | 71 (3.2) | 55 | 378 (271 KiB) | 69 (1099 MiB) | 0.437 |
| 7 | Transformer | Transformer | **100.0** | 100 (2.379) | 100 (52%) | 100 (0.22) | 100 (2.3) | 100 | 100 (1024 KiB) | 100 (761 MiB) | 0.762 |
| 8 | Our Mamba v1 | Mamba (ours) | **84.1** | 117 (2.151) | 108 (56%) | 6 (3.77) | 123 (1.9) | 18 | 2327 (44 KiB) | 22 (3488 MiB) | 0.047 |

Each column is an index with the Transformer = 100 (higher is better), raw value in brackets. Combined = weighted geometric mean: quality 35%, recall 15%, train_speed 15%, decode_speed 15%, state_memory 10%, prefill_speed 5%, train_memory 5%. Rows with a quality index below 80 are shown but not ranked. Speed and memory: one machine (Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz), one thread, every arm in its own process.
