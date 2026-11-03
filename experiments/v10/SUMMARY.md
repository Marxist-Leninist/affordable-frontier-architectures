# v10 completed result

Training tokens per run: 1,394,688. Learning rates selected by validation CE: {'transformer': 0.01, 'rnn': 0.001}

Chance is6.25%. Accuracy percentages below are mean [min,max] across3 initialization seeds on the same256 held-out cases per cell.

## transformer
- L512_K4: 6.25% [6.25, 6.25], CE2.7719
- L512_K8: 6.25% [6.25, 6.25], CE2.7709
- L512_K16: 6.25% [6.25, 6.25], CE2.7738
- L2048_K4: 6.25% [6.25, 6.25], CE2.7672
- L2048_K8: 6.25% [6.25, 6.25], CE2.7720
- L2048_K16: 6.25% [6.25, 6.25], CE2.7763
- L4096_K4: 6.25% [6.25, 6.25], CE2.7701
- L4096_K8: 6.25% [6.25, 6.25], CE2.7749
- L4096_K16: 6.25% [6.25, 6.25], CE2.7756
## rnn
- L512_K4: 6.38% [4.69, 7.42], CE2.7997
- L512_K8: 5.34% [5.08, 5.47], CE2.8007
- L512_K16: 5.86% [4.30, 7.42], CE2.7883
- L2048_K4: 4.56% [2.73, 6.25], CE2.7931
- L2048_K8: 6.51% [5.86, 7.03], CE2.8065
- L2048_K16: 5.47% [4.69, 6.64], CE2.7858
- L4096_K4: 7.16% [6.64, 7.81], CE2.7863
- L4096_K8: 7.68% [6.25, 10.16], CE2.8067
- L4096_K16: 5.21% [4.69, 5.86], CE2.7975

All12 checkpoints saved, exact reload parity, file SHA256 reverified after completion.

Both architectures failed to learn robust retrieval at this budget. These results do not rank memory capacity or support an architectural advantage. No-position architecture avoids the earlier absolute-position failure mode, but does not itself solve optimization.