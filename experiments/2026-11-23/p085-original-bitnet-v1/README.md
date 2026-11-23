# P085: 2023 BitNet v1 training study
This independent, from-scratch mini implementation follows [BitNet v1 (Wang et al., 2023)](https://arxiv.org/html/2310.11453v1), sections 2.1–2.2 and Appendix A. It is **binary W1A8**, not the later ternary b1.58 recipe, not an official checkpoint, and not a reproduction of the paper's model scale or corpus. Training uses ordinary PyTorch FP32 master weights, FP32 AdamW and fake quantization; the report separates those costs from actual packed inference storage. No Microsoft source code or checkpoint is included.

On the GTX 1660-class experiment card, the 1.903M-parameter mini averaged 3.40707 bits per byte across two seeds versus 2.37099 for its topology-matched FP32 control. It did not qualify on the predeclared quality threshold. Read the [report](REPORT.md), [protocol](PROTOCOL.md) and [machine-readable summary](summary.json) for the full comparison. This negative mini-scale result does not evaluate native low-bit kernels or establish a general quality limit.

## Reproduce

Use Python with NumPy and PyTorch. The GPU run needs a CUDA-capable PyTorch build; datasets is needed only for fetching the pinned WikiText2 release. No dataset or checkpoint is bundled here.

```sh
python train.py --qa-cpu --out cpu-qa
python prepare_wikitext.py --out wikitext2
python train.py --data wikitext2 --out p085-run
python summary.py p085-run > reproduced-summary.json
python beta_diag.py p085-run > reproduced-beta-diagnostic.json
```

The CPU QA checks the forward scalar oracle, inference causality, the straight-through backward pass and serialized optimizer continuation. The final training checkpoints contain weights, not the final Adam moments. The separate beta diagnostic measures reconstruction error in weight space only. Instructions in the protocol describe the measured run, not a request to reproduce it on production hardware.

## Verify the archived checkpoints

The study's four master-only training snapshots, two inference exports and frozen receipts are hosted separately with the [immutable Hugging Face experiment](https://huggingface.co/ScAGI/affordable-frontier-architectures/tree/74d57fd6bf9637e6606506c9f028f341c17d1071/experiments/2026-11-23/p085-original-bitnet-v1). Download its `evidence/` and `weights/` subdirectories intact into a fresh directory, then run `python verify.py --root /path/to/downloaded-experiment`. The original CUDA check recovered exact logits on its tested inputs. The independent CPU check recovers exact signs and logits when both sides use the archived beta values; newly reduced FP32 beta on a different device can move a rounded activation. The small checkpoints are our research outputs, not Microsoft weights.

WikiText2 and third-party software retain their original terms. The training implementation here is independent; it does not vendor BitNet code or confer rights to the upstream paper, dataset or software.
