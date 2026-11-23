# P086: 2025 Microsoft BitNet b1.58 clean-room pretraining pilot

This independent mini implementation is based on Microsoft's [2025 BitNet b1.58 2B4T technical report](https://arxiv.org/html/2504.12285v1#S2), publicly documented configuration, and compatible model semantics. No upstream source code or Microsoft weights are redistributed. See [the protocol](PROTOCOL.md) for the pinned upstream references, adaptation decisions and release limitations.

Unlike [P085's 2023 W1A8 study](../p085-original-bitnet-v1/), this uses ternary `−1, 0, +1` projection weights, tokenwise A8, a matched grouped-query Transformer with gated squared ReLU and RMSNorm, tied embeddings, rotary position encoding, and a two-stage pretraining schedule with a training-only prose filter. Both P086 arms use the same architecture and schedule. The four-layer, 1,601,984-parameter experiment uses WikiText2 bytes, not the original vocabulary/corpus. It does not reproduce Microsoft's 2B weights, original full training, supervised fine-tuning, or preference tuning. Runtimes use PyTorch fake quantization; a packed inference export measures storage, not native low-bit speed or training memory.

## Measured comparison

Both arms ran two seeds from scratch, 2,400 updates and 4,915,200 paired byte targets per seed and arm. The matched FP32 control ranked **1st (100.00)** with 2.22367 test bits per byte and 49.24 s mean fit. BitNet b1.58 ranked **2nd (94.40)** with 2.23090 test bits per byte and 55.07 s mean fit. Both passed the preregistered quality gate (within +0.15 BPB per seed); the paired fixed-seed block interval for the ternary arm's BPB penalty was [0.00066, 0.01409]. These data do not give a common P085–P086 leaderboard, as architecture, schedule and held-out windows differ. See [REPORT.md](REPORT.md) and [summary.json](summary.json).

The custom ternary export stored 601,968 B of tensor payload versus 6,407,936 B for the dense model, a 10.64× reduction. Both used 19,223,992 B of persistent trainable master weights plus Adam states/counters. Packed projection codes, tokenwise activation codes, FP32 computation/accumulation, persistent training state, and measured whole-GPU allocation are distinct. No native kernel, inference speed or serving-memory win was measured.

## Reproduce and verify

Use Python with NumPy and PyTorch, plus a CUDA-enabled build for the measured fit. The pinned-data helper requires Hugging Face datasets and checks all split hashes. There is no raw dataset, checkpoint, credentials, private log or vendor code in this GitHub directory.

```sh
sha256sum -c SHA256SUMS
python train.py --qa-cpu --out cpu-qa
python prepare_wikitext.py --out wikitext2
python train.py --data wikitext2 --out p086-run
python summary.py p086-run > reproduced-summary.json
```

Obtain the study's `weights/` and `evidence/` from the [immutable Hugging Face release](https://huggingface.co/ScAGI/affordable-frontier-architectures/tree/56578d74a0aaa72761f8a1f931aedf49723096eb/experiments/2026-11-23/p086-bitnet-b158-2b4t-reconstruction) and run `python verify.py --root /path/to/downloaded-experiment` for CPU replay of the published snapshots. Inspect PyTorch snapshots with `weights_only=True`. Final master files do not retain the final Adam states; exact next-update restore is a separate engineering test. WikiText2 and third-party tools remain subject to their own terms.

The original CUDA pack replay matched checked logits exactly. On the separate CPU probe, recomputing inverse scales from FP32 masters differed from using archived GPU scales by up to 0.05821/0.03626 logit units for the two seeds, despite zero code mismatches. Sharing the CPU-native scales eliminated those differences bit-for-bit on that probe. This is a cross-device rounding effect, not checkpoint corruption; no universal bitwise parity is claimed.
