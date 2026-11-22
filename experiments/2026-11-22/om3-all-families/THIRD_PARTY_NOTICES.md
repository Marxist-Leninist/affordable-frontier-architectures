# Third-party notices

The eight arms in `arms/` were written for this study from the papers cited in each file. Where a paper leaves a detail open, the file follows the convention of the authors' public reference implementation and names it: google-deepmind/recurrentgemma (Apache-2.0), fla-org/flash-linear-attention (MIT), NX-AI/xlstm (Apache-2.0), microsoft/torchscale (MIT), BlinkDL/RWKV-LM (Apache-2.0), HazyResearch/safari (Apache-2.0).

- `om2bench.py` is the OM2 harness: `OM2Mixer._ssd` and `segsum` are adapted from the `ssd_minimal` reference in state-spaces/mamba (`mamba_ssm/modules/ssd_minimal.py`, Apache License 2.0, (c) 2024 Albert Gu and Tri Dao); the `m2` and `m2hf` arms build transformers' `Mamba2ForCausalLM` (Apache License 2.0) to copy its definition and initialisation.
- Data (not redistributed): WikiText-2 raw (Merity et al. 2016, derived from Wikipedia, CC BY-SA 3.0); public-domain Project Gutenberg fiction (held-out windows from eBooks #84 and #174); OpenAssistant Conversations oasst1 (Apache License 2.0).
