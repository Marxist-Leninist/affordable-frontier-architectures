# Reconstruct-and-continue independent Diogenes harness

See MODEL_CARD.md and LICENSE for exact attribution and scope. Source code
doesn't execute an HF model script, install software, load pickles, require
credentials, upload data, or make paid API calls. CPU-only in measured run.
Use official PyTorch, safetensors, NumPy, pyarrow, and Hugging Face datasets.
The stable original SHA is checked before update. All outputs are safetensors
model states; they do not include optimizer, RNG or data resume state.

1. Put the official pinned config and original model.safetensors in MODEL.
2. For the actual Hugging Face **network-stream** route, run
   python stream_fixture.py results/cached-reference live_frozen
   with an empty dedicated HF_HOME. This uses streaming=True on only the three
   immutable official public Parquet shard URLs and freezes selected raw byte
   windows privately (no copies of raw windows are included here). It requires
   exact fixture hash parity against the provided metadata-only cached run.
3. Run CUDA_VISIBLE_DEVICES='' python pilot.py MODEL live_frozen RUN
4. Start a new Python process:
   CUDA_VISIBLE_DEVICES='' python pilot.py MODEL live_frozen RUN --replay

The alternative local-data route passes DATA containing exact pinned
train.parquet, validation.parquet, test.parquet, SHA checked independently.
That route is local row streaming, not live HF network streaming.

## Published artifacts
- [Verified winning checkpoint and upstream licence](https://huggingface.co/ScAGI/affordable-frontier-architectures/tree/main/experiments/2026-11-22/diogenes-reconstructed-trainer)
- [Model card with measured pilot results](MODEL_CARD.md)
- [Exact live-stream evaluation](results/live-stream/test.json) and
  [fresh reload](results/live-stream/fresh_replay.json)
