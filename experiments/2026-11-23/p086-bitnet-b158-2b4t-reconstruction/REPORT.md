# P086 new-context held-out evidence

A matched mini pretraining comparison on one GTX 1660 SUPER, with no Microsoft model bytes and no chat/SFT/DPO experiment. Lower BPB is better.

| Same topology and training exposure | Test BPB | Fit seconds | Eligible quality × fit index |
|---|---:|---:|---:|
| FP32 | 2.22367 | 49.24 | 100.00 |
| Ternary W / tokenwise A8 adaptation | 2.23090 | 55.07 | 94.40 |

Per-seed BitNet minus FP32 loss +0.00468/+0.00978 BPB; both met +0.15. Pooled difference +0.00723 BPB, fixed-seed paired 32×8-window block interval +0.00066 to +0.01409. This rules out a superiority claim while showing matched small-test quality within the predeclared tolerance. Training used 4,915,200 targets per seed/arm (2,400 B16×T128 updates), the same fixed indexed exposures and staged schedule. One preflight-rejected candidate window matched train exactly; the locked corrected selection excludes it before scoring. See protocol for limits, fresh held-out byte starts, and why these are not independent held-out articles.

Each arm used 6,407,936 bytes of FP32 masters plus 12,816,056 bytes of Adam state (19,223,992 total persistent tensor bytes). Whole fit allocator peaks were 289,142,784–327,654,912 bytes (FP32) and 318,535,680 bytes (both ternary seeds). Separate ternary inference artifacts contain 387,072 code bytes, 112 scale bytes and 214,784 high-precision other bytes (601,968 tensor bytes, 623,045 on disk). Exact original CUDA reconstruction/serialization check applies to its tested inputs; this is 10.64× lower tensor payload than FP32, not native execution speed. Separate fresh CPU verifier showed max logit differences 0.05821 and 0.03626 from re-quantizing the masters on CPU versus consuming the exact archived ternary packs on its synthetic probes. This does not contradict exact historical CUDA replay. Shared inverse-scale comparisons explicitly test that separate issue. No serving timings, deployment, broader generalization, efficiency-per-watt, or memory peak at inference was measured.

Original measured train source SHA-256: 6393ed7d4f787f4c6e2b016fb69f6f0ab7db5e2f78a857fb609a8c88d6aafdf3. Release copy changes data path and CUDA-device selection only. The reported aggregate is SHA-256 3ccf13c0a886d0796d2426d2e2472071501106a63e0d010c84b424aa806eec89 and can be regenerated from the included numeric evidence. The run used no vendor binaries or pretrained weights.
