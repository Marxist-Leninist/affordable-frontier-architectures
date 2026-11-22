# OM2 results

| rank | arm | params | test bpb (macro) | article | fiction | dialogue | quality index | train s | bpb gained / train min | decode tok/s | decode p95 ms | combined |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Our Mamba v1 | 474,752 | 2.1509 | 2.142 | 2.156 | 2.154 | 117.1 | 8123 | 0.0440 | 230.1 | 5.42 | 39.6 |
| 2 | v2 + always-on exact store | 504,289 | 2.1525 | 2.151 | 2.156 | 2.150 | 117.0 | 1486 | 0.2402 | 135.0 | 9.14 | 58.4 |
| 3 | v2 + state-gated exact store | 504,289 | 2.1569 | 2.156 | 2.160 | 2.154 | 116.6 | 1466 | 0.2432 | 170.1 | 8.05 | 63.3 |
| 4 | v2 + placebo sidecar (no lookup) | 504,289 | 2.1628 | 2.153 | 2.174 | 2.162 | 116.2 | 1126 | 0.3162 | 323.0 | 3.21 | 85.4 |
| 5 | Our Mamba v2 | 471,264 | 2.1676 | 2.152 | 2.183 | 2.168 | 115.8 | 1030 | 0.3452 | 275.5 | 3.88 | 83.4 |
| 6 | Our Mamba v2, positive decay only | 471,264 | 2.1681 | 2.163 | 2.173 | 2.168 | 115.7 | 997 | 0.3566 | 319.2 | 3.39 | 88.5 |
| 7 | Mamba-2 (HF definition and init, our kernel) | 471,008 | 2.1782 | 2.164 | 2.188 | 2.182 | 114.9 | 1316 | 0.3114 | 210.0 | 6.56 | 70.0 |
| 8 | Transformer | 476,288 | 2.3788 | 2.388 | 2.350 | 2.398 | 100.0 | 608 | 0.5616 | 324.5 | 3.52 | 100.0 |

## Exact recall under distractors (second occurrence, bytes 5-32 of a repeated 32-byte span)

| arm | gap 16 | gap 64 | gap 128 | gap 176 |
|---|---|---|---|---|
| Transformer | 50.7% (gain -0.04) | 52.5% (gain -0.05) | 52.8% (gain -0.07) | 52.4% (gain -0.05) |
| Mamba-2 (HF definition and init, our kernel) | 54.4% (gain -0.04) | 56.0% (gain -0.07) | 56.3% (gain -0.08) | 57.3% (gain -0.08) |
| Our Mamba v1 | 54.6% (gain -0.07) | 55.6% (gain -0.09) | 56.9% (gain -0.11) | 57.2% (gain -0.10) |
| Our Mamba v2 | 55.1% (gain -0.05) | 55.8% (gain -0.09) | 56.0% (gain -0.11) | 57.1% (gain -0.09) |
| Our Mamba v2, positive decay only | 54.8% (gain -0.03) | 56.0% (gain -0.07) | 55.8% (gain -0.10) | 56.3% (gain -0.09) |
| v2 + placebo sidecar (no lookup) | 54.2% (gain -0.05) | 56.0% (gain -0.08) | 56.2% (gain -0.10) | 57.0% (gain -0.10) |
| v2 + always-on exact store | 54.6% (gain -0.05) | 55.9% (gain -0.06) | 56.9% (gain -0.08) | 57.1% (gain -0.08) |
| v2 + state-gated exact store | 54.5% (gain -0.06) | 56.2% (gain -0.07) | 56.0% (gain -0.08) | 57.2% (gain -0.07) |

## Paired contrasts (test macro bpb, negative = first arm better; document bootstrap, descriptive)

- om2 minus om1: +0.0168 [+0.0127, +0.0209], per seed +0.0150, +0.0185
- om2 minus m2: -0.0106 [-0.0145, -0.0070], per seed -0.0186, -0.0026
- om2 minus om2pos: -0.0005 [-0.0042, +0.0033], per seed +0.0009, -0.0018
- om2pos minus m2: -0.0101 [-0.0136, -0.0066], per seed -0.0195, -0.0007
- om2 minus tf: -0.2111 [-0.2189, -0.2034], per seed -0.2103, -0.2120
- m2 minus om1: +0.0274 [+0.0236, +0.0312], per seed +0.0336, +0.0211
- m2 minus tf: -0.2005 [-0.2093, -0.1921], per seed -0.1916, -0.2094
- om1 minus tf: -0.2279 [-0.2359, -0.2195], per seed -0.2253, -0.2305
- om2g minus om2a: +0.0044 [+0.0008, +0.0080], per seed +0.0036, +0.0051
- om2g minus om2p: -0.0059 [-0.0098, -0.0020], per seed -0.0049, -0.0069
- om2a minus om2p: -0.0102 [-0.0140, -0.0066], per seed -0.0085, -0.0120
- om2p minus om2: -0.0049 [-0.0089, -0.0007], per seed -0.0055, -0.0042
- om2g minus om2: -0.0108 [-0.0151, -0.0064], per seed -0.0104, -0.0112
- om2a minus tf: -0.2262 [-0.2339, -0.2187], per seed -0.2242, -0.2282
- om2g minus tf: -0.2219 [-0.2297, -0.2146], per seed -0.2206, -0.2231
- v2 + placebo sidecar (no lookup): soft gate open rate 0.003, mean gate 0.009
- v2 + state-gated exact store: soft gate open rate 0.000, mean gate 0.070; hard-gated test bpb 2.1977 with reads on 0.0% of tokens
