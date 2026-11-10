# R4 growth study: inference checkpoint package

`tiny_dlgm_growth_inference_r4_20261110.zip` is the standalone inference export for the R4 parameter-growth study reported in [the study report](../README.md). It holds 17 model-only FP32 NumPy checkpoints (the 15 CPU-60 states for arms S, L, Z, G and R at seeds 1 to 3, plus G seed 1 immediately before and after its first growth step), the exact original #458 DualScaleMemoryRouter source, the study's model wrapper and CPU adapter, a loader, public-byte fixtures and a strict validator.

| | |
| --- | --- |
| Size | 4,596,554 bytes, 65 files inside |
| SHA-256 | `75037594d92e4aa33c32b930d0c28b32c700a53ec10f4217a64d865655626e84` |
| Git blob | `c33da788a771d24f0b729d4e660ca2c3cdf61332` |

## Use

```sh
unzip tiny_dlgm_growth_inference_r4_20261110.zip
cd tiny_dlgm_growth_inference_r4
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0+cpu"
pip install "numpy==2.3.5" "einops==0.8.1"
python validate.py
python load_model.py --checkpoint G_seed1_cpu60 --text "A small public byte sequence."
```

Read the README inside the ZIP for the format, the arms and the replication boundary. In short: this reloads the tested model maths and these exact inference states. It does not reproduce training, optimizer or RNG resume, the study's data sampler, or cached decoding, and it carries no corpus text.

## How this copy got here

The package was prepared and validated on 10 November 2026 by the agent that ran the study (AMD EPYC, Python 3.12.14, PyTorch 2.6.0+cpu): 34 of 34 reference comparisons exact. Its upload to this repository was cut off after GitHub had accepted the file but before it was committed. This folder completes that upload from the accepted object, so the bytes are the validated export (hashes above).

The ZIP's own README still says "Local prepared package; not yet published", because it was written before publication. The archive was left byte-identical rather than edited. The three report files in the parent folder are unchanged; their statement that the release holds no checkpoints describes the report commit, before this folder was added.

## Independent re-check on a second machine

Run on 10 November 2026 on an Intel Xeon (Sapphire Rapids: AVX512-BF16 and AMX), Python 3.13.18, PyTorch 2.6.0+cpu, NumPy 2.3.5, einops 0.8.1:

- ZIP CRC passes and all 64 entries of `SHA256SUMS.json` match.
- `original/458_delta_net_dlgm.py` is byte-identical to record 458 of ASI-Arch `data/106.json` at commit `3113c51b04a1d4a7520a3b0238c6a08d3194d58f`.
- With AMX switched off (`ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`), all 34 comparisons are bit-exact and `validate.py` passes.
- With default settings on that AMX-capable CPU, 27 of 34 comparisons are bit-exact. The other 7 differ by at most 0.0390625 in the BF16 logits (1.85% of logit values); the next-byte argmax is identical at all 2,720 positions. `validate.py` fails there by design, since it never relaxes its exact tolerance. A plain FP32 forward pass stays within 0.0953 of the stored references.

So on an Intel CPU with AMX, run:

```sh
ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 python validate.py
```

This re-check covers loading and the two fixed fixtures only. It is not a rerun of the study's training or held-out evaluation, and GPU use is still untested.
