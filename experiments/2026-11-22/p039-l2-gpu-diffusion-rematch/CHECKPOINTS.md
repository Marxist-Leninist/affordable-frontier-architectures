# Checkpoints

The ten best-dev checkpoints (safetensors, float32, no metadata) are on Hugging Face: [https://huggingface.co/ScAGI/affordable-frontier-architectures/tree/main/experiments/2026-11-22/p039-l2-gpu-diffusion-rematch](https://huggingface.co/ScAGI/affordable-frontier-architectures/tree/main/experiments/2026-11-22/p039-l2-gpu-diffusion-rematch)
under `checkpoints/`. They are not on GitHub. Final-step checkpoints and optimizer states are not published. The two
v1 diffusion runs were stopped near update 2,600 and are not published either.

| file | bytes | sha256 | best update | best dev bits/byte |
|---|---|---|---|---|
| `checkpoints/v1/attn_ar_17/best.safetensors` | 13,430,264 | `f3973f3488034c4af1b262d47b48e997c32f44472b4a263afa85d023e947addb` | 20,000 | 1.7007 |
| `checkpoints/v1/attn_ar_43/best.safetensors` | 13,430,264 | `6aa09c9954a3d36664d8a8339cb5f954920724f4ec0d103f3689dfe5dd4413d8` | 20,000 | 1.7030 |
| `checkpoints/v2/attn_ar_17/best.safetensors` | 13,168,032 | `64f60ccb1569e82c5bd7c8fd1eb23e9fa9530c070d0bc78e73ccf0c8040a7f14` | 18,000 | 1.7450 |
| `checkpoints/v2/attn_ar_43/best.safetensors` | 13,168,032 | `1c7e385c3556888fc429dd823823f03f05384386ecd9add56e97cc6dce78ab5c` | 18,000 | 1.7485 |
| `checkpoints/v2/attn_diff_17/best.safetensors` | 13,168,032 | `414bfc869e52ccf1120211719759aeed7037bac517404739ce63ccfd090f6f4d` | 20,000 | 2.5567 |
| `checkpoints/v2/attn_diff_43/best.safetensors` | 13,168,032 | `0924fa11092152b224bf427141936413379a03a30ee3bc80bd71102736efd126` | 20,000 | 2.5541 |
| `checkpoints/v2/hybrid_ar_17/best.safetensors` | 13,170,352 | `471df5a8877bff0a4a914ecd794273fbe60a97269038cbc8c56cf380ad6298c7` | 20,000 | 1.7100 |
| `checkpoints/v2/hybrid_ar_43/best.safetensors` | 13,170,352 | `0a6ec3f4fcd73d45cbc396d54aa7b5eaa9bf24c6bd943f8e1874d0a49716e9d8` | 20,000 | 1.7246 |
| `checkpoints/v2/hybrid_diff_17/best.safetensors` | 13,170,352 | `1cce6462390e8018c6e8cac353f65c5158660cdc6d75014b1e4f2e87dc59b681` | 20,000 | 2.4583 |
| `checkpoints/v2/hybrid_diff_43/best.safetensors` | 13,170,352 | `0dc32a71763fcf1413b87825b46ff871bb3deafcc0be5080c71381fbcbac0511` | 20,000 | 2.4676 |

Dev bits/byte are exact NLL for AR runs and an 8-sample NELBO for diffusion runs.

Data credits: WikiText-2 (Merity et al., 2016; Wikipedia text, CC BY-SA 3.0), Project Gutenberg books (public domain in the US) and OpenAssistant Conversations (Köpf et al., 2023; Apache-2.0). The models were trained on this text; none of it is included here.
