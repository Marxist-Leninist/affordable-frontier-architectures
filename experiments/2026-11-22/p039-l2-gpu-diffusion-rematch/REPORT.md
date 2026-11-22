# P039-L2: matched GPU-scale rematch of P039 (GTX 1660 SUPER)

Cases: 4,768 prompt-240 / suffix-16 completions (article 496, fiction 512, dialogue 3,760) from 303 held-out P045 test documents (article 31, fiction 32, dialogue 240), equal-domain then equal-document averages, 95% CIs from 2000 document bootstrap replicates.

## All models and metrics, best first

| Rank | Model · decoder | Index (attn_ar = 100) | Quality index | Held-out bits/byte [95% CI] | Non-space acc | Exact 16/16 | Space rate | Degenerate | Decode ms (16 bytes, GPU kernel) | Train GPU min | Dev bits gained / train min | GPU min to common dev target | Train-step peak MiB | Params | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | attn_ar · greedy ★ | 100.0 | 100.0 | 1.823 [1.780, 1.873] | 0.1503 | 0.0010 | 0.208 | 0.003 | 6.50 | 14.2 | 0.0291 | 12.8 (2/2 seeds) | 393 | 3,290,880 | pass |
| 2 | hybrid_ar · greedy ★ | 97.5 | 102.2 | 1.784 [1.741, 1.832] | 0.1537 | 0.0020 | 0.209 | 0.003 | 6.18 | 17.2 | 0.0184 | 11.2 (2/2 seeds) | 639 | 3,291,224 | pass |
| – | hybrid_diff · l2r16 | n/a | 69.3 | 2.449 [2.414, 2.485] | 0.0969 | 0.0000 | 0.234 | 0.015 | 34.67 | 25.2 | 0.0271 | not reached (0/2) | 918 | 3,291,224 | fail: bits not below KN, non-space acc not above KN |
| – | hybrid_diff · conf16 ★ | n/a | 68.3 | 2.449 [2.414, 2.485] | 0.0942 | 0.0000 | 0.254 | 0.024 | 34.67 | 25.2 | 0.0271 | not reached (0/2) | 918 | 3,291,224 | fail: bits not below KN, non-space acc not above KN |
| – | attn_diff · l2r16 | n/a | 66.3 | 2.543 [2.510, 2.577] | 0.0922 | 0.0000 | 0.225 | 0.014 | 19.16 | 16.3 | 0.0471 | not reached (0/2) | 393 | 3,290,880 | fail: bits not below KN, non-space acc not above KN |
| – | attn_diff · conf16 ★ | n/a | 64.9 | 2.543 [2.510, 2.577] | 0.0884 | 0.0000 | 0.247 | 0.027 | 19.16 | 16.3 | 0.0471 | not reached (0/2) | 393 | 3,290,880 | fail: bits not below KN, non-space acc not above KN |
| – | hybrid_diff · conf8 | n/a | 62.8 | 2.449 [2.414, 2.485] | 0.0797 | 0.0000 | 0.425 | 0.209 | 17.38 | 25.2 | 0.0271 | not reached (0/2) | 918 | 3,291,224 | fail: bits not below KN, non-space acc not above KN |
| – | attn_diff · conf8 | n/a | 62.5 | 2.543 [2.510, 2.577] | 0.0820 | 0.0000 | 0.340 | 0.088 | 9.62 | 16.3 | 0.0471 | not reached (0/2) | 393 | 3,290,880 | fail: bits not below KN, non-space acc not above KN |
| – | attn_diff · conf4 | n/a | 57.0 | 2.543 [2.510, 2.577] | 0.0682 | 0.0000 | 0.519 | 0.205 | 4.81 | 16.3 | 0.0471 | not reached (0/2) | 393 | 3,290,880 | fail: bits not below KN, non-space acc not above KN |
| – | hybrid_diff · conf4 | n/a | 54.9 | 2.449 [2.414, 2.485] | 0.0609 | 0.0000 | 0.659 | 0.510 | 8.69 | 25.2 | 0.0271 | not reached (0/2) | 918 | 3,291,224 | fail: bits not below KN, non-space acc not above KN, half or more degenerate |
| – | attn_diff · conf2 | n/a | 52.3 | 2.543 [2.510, 2.577] | 0.0574 | 0.0000 | 0.734 | 0.578 | 2.40 | 16.3 | 0.0471 | not reached (0/2) | 393 | 3,290,880 | fail: bits not below KN, non-space acc not above KN, half or more degenerate |
| – | hybrid_diff · conf1 | n/a | 52.2 | 2.449 [2.414, 2.485] | 0.0551 | 0.0000 | 0.838 | 0.910 | 2.15 | 25.2 | 0.0271 | not reached (0/2) | 918 | 3,291,224 | fail: bits not below KN, non-space acc not above KN, half or more degenerate |
| – | hybrid_diff · conf2 | n/a | 51.5 | 2.449 [2.414, 2.485] | 0.0535 | 0.0000 | 0.816 | 0.800 | 4.33 | 25.2 | 0.0271 | not reached (0/2) | 918 | 3,291,224 | fail: bits not below KN, non-space acc not above KN, half or more degenerate |
| – | attn_diff · conf1 | n/a | 48.4 | 2.543 [2.510, 2.577] | 0.0492 | 0.0000 | 0.849 | 0.940 | 1.20 | 16.3 | 0.0471 | not reached (0/2) | 393 | 3,290,880 | fail: bits not below KN, non-space acc not above KN, half or more degenerate |
| ref | KN byte 9-gram · greedy | n/a | n/a | 2.040 [1.996, 2.088] | 0.1410 | 0.0015 | 0.210 | 0.004 | n/a | n/a | n/a | n/a | n/a | n/a | reference |
| ref | always a space · - | n/a | n/a | n/a | 0.0000 | 0.0000 | 1.000 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | reference |

How to read: one row per network and decoder, best first. Ranked by the Index. Index = weighted geometric mean of each metric's ratio to attn_ar greedy, inverted where lower is better (likelihood 0.25, nonspace acc 0.25, latency b1 0.25, train time 0.25); 100 = attn_ar greedy, higher is better. Only rows that pass the quality gate are ranked: held-out bits/byte below the KN byte n-gram (2.040), non-space accuracy above KN greedy (0.1410), and fewer than half of the completions degenerate (one byte 12 or more times in 16). So a fast model that outputs spaces cannot rank. The gate and the weights were set before any network was scored on the test set. Quality index = held-out likelihood and non-space accuracy only, equal weights.

Held-out bits/byte: exact next-byte NLL for AR; for diffusion, the exact left-to-right NLL of the same network (one masked byte revealed per step). Decode ms = summed GPU kernel time (torch.profiler) for one 16-byte completion after a 240-byte prompt, batch 1, NVIDIA GeForce GTX 1660 SUPER, median of three rotated rounds; it leaves out Python and kernel-launch overhead, which weighs most on cached AR (eager wall-clock times are in timing.json under the b1_* keys). AR reuses its KV/state cache, diffusion re-reads the whole window on every pass (confN = N passes, most confident bytes first; l2r16 = 16 passes, left to right). Train GPU min = 20,000 steps × the median over three rounds of the mean GPU kernel time per training step. Dev scores are exact NLL for AR and an 8-sample NELBO (an upper bound) for diffusion, checked every 2,000 steps. Dev bits gained / train min = drop in dev bits/byte from the first dev check to the best one, per GPU minute between them; it rewards a model that starts worse, so it is not comparable between AR and diffusion. GPU min to common dev target = GPU minutes until the dev curve first reaches the worst AR run's best dev bits (mean over seeds). Train-step peak MiB = peak allocated memory of one training step in the timing pass; the full runs record their own peaks, which include dev evaluation, in state.json. ★ = decoder set before any network was scored on the test set.

Timing clean (no other GPU processes seen while timing): yes.

## Detail by arm and decoder

| Arm | Params | Suffix bits/byte (lower better) | Decoder | Non-space acc | Byte acc | Exact 16/16 | Space rate | Degenerate |
|---|---|---|---|---|---|---|---|---|
| attn_ar | 3,290,880 | 1.823 [1.780, 1.873] | greedy | 0.1503 | 0.1906 | 0.0010 | 0.208 | 0.003 |
| attn_diff | 3,290,880 | 2.543 [2.510, 2.577] (NELBO 2.595) | conf1 | 0.0492 | 0.2067 | 0.0000 | 0.849 | 0.940 |
| attn_diff | 3,290,880 | 2.543 [2.510, 2.577] (NELBO 2.595) | conf2 | 0.0574 | 0.1947 | 0.0000 | 0.734 | 0.578 |
| attn_diff | 3,290,880 | 2.543 [2.510, 2.577] (NELBO 2.595) | conf4 | 0.0682 | 0.1648 | 0.0000 | 0.519 | 0.205 |
| attn_diff | 3,290,880 | 2.543 [2.510, 2.577] (NELBO 2.595) | conf8 | 0.0820 | 0.1438 | 0.0000 | 0.340 | 0.088 |
| attn_diff | 3,290,880 | 2.543 [2.510, 2.577] (NELBO 2.595) | conf16 | 0.0884 | 0.1321 | 0.0000 | 0.247 | 0.027 |
| attn_diff | 3,290,880 | 2.543 [2.510, 2.577] (NELBO 2.595) | l2r16 | 0.0922 | 0.1322 | 0.0000 | 0.225 | 0.014 |
| hybrid_ar | 3,291,224 | 1.784 [1.741, 1.832] | greedy | 0.1537 | 0.1939 | 0.0020 | 0.209 | 0.003 |
| hybrid_diff | 3,291,224 | 2.449 [2.414, 2.485] (NELBO 2.511) | conf1 | 0.0551 | 0.2106 | 0.0000 | 0.838 | 0.910 |
| hybrid_diff | 3,291,224 | 2.449 [2.414, 2.485] (NELBO 2.511) | conf2 | 0.0535 | 0.2045 | 0.0000 | 0.816 | 0.800 |
| hybrid_diff | 3,291,224 | 2.449 [2.414, 2.485] (NELBO 2.511) | conf4 | 0.0609 | 0.1825 | 0.0000 | 0.659 | 0.510 |
| hybrid_diff | 3,291,224 | 2.449 [2.414, 2.485] (NELBO 2.511) | conf8 | 0.0797 | 0.1588 | 0.0000 | 0.425 | 0.209 |
| hybrid_diff | 3,291,224 | 2.449 [2.414, 2.485] (NELBO 2.511) | conf16 | 0.0942 | 0.1404 | 0.0000 | 0.254 | 0.024 |
| hybrid_diff | 3,291,224 | 2.449 [2.414, 2.485] (NELBO 2.511) | l2r16 | 0.0969 | 0.1370 | 0.0000 | 0.234 | 0.015 |

Baselines: KN byte 9-gram (order chosen on dev) suffix bits 2.040; all_space: non-space 0.0000, byte 0.1755, exact 0.0000; kn9_greedy: non-space 0.1410, byte 0.1795, exact 0.0015

Aggregate index (attn_ar = 100; weights {'likelihood': 0.25, 'nonspace_acc': 0.25, 'latency_b1': 0.25, 'train_time': 0.25}):

- attn_ar (greedy): aggregate 100.0, quality-only 100.0, 0.0291 dev bits gained per clean training minute, clean training 14.2 min, ratios likelihood 1.000, nonspace_acc 1.000, latency_b1 1.000, train_time 1.000
- attn_diff (conf16): aggregate not ranked (gate failed: likelihood_beats_kn, nonspace_beats_kn_greedy), quality-only 64.9, 0.0471 dev bits gained per clean training minute, clean training 16.3 min, ratios likelihood 0.717, nonspace_acc 0.588, latency_b1 0.339, train_time 0.870
- hybrid_ar (greedy): aggregate 97.5, quality-only 102.2, 0.0184 dev bits gained per clean training minute, clean training 17.2 min, ratios likelihood 1.022, nonspace_acc 1.022, latency_b1 1.051, train_time 0.824
- hybrid_diff (conf16): aggregate not ranked (gate failed: likelihood_beats_kn, nonspace_beats_kn_greedy), quality-only 68.3, 0.0271 dev bits gained per clean training minute, clean training 25.2 min, ratios likelihood 0.744, nonspace_acc 0.627, latency_b1 0.187, train_time 0.563

Paired differences vs attn_ar (arm minus attn_ar):

- attn_diff: bits +0.7200 [+0.6910, +0.7472]; conf16_vs_greedy:nonspace_acc -0.0619 [-0.0679, -0.0556]; conf16_vs_greedy:exact -0.0010 [-0.0014, -0.0005]; conf16_vs_greedy:byte_acc -0.0585 [-0.0641, -0.0523]
- hybrid_ar: bits -0.0391 [-0.0484, -0.0297]; greedy_vs_greedy:nonspace_acc +0.0034 [-0.0006, +0.0073]; greedy_vs_greedy:exact +0.0011 [+0.0002, +0.0022]; greedy_vs_greedy:byte_acc +0.0033 [-0.0009, +0.0073]
- hybrid_diff: bits +0.6258 [+0.5970, +0.6540]; conf16_vs_greedy:nonspace_acc -0.0561 [-0.0621, -0.0501]; conf16_vs_greedy:exact -0.0010 [-0.0014, -0.0005]; conf16_vs_greedy:byte_acc -0.0502 [-0.0565, -0.0435]
