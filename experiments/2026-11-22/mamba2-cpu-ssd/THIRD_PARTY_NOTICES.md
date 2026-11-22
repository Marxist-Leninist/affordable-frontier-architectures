# Third-party notices

## state-spaces/mamba: `ssd_minimal` reference (Apache License 2.0)

`segsum` and `ssd_chunked` in `mamba2_cpu_ssd.py` are adapted from `mamba_ssm/modules/ssd_minimal.py` in
https://github.com/state-spaces/mamba (checked against tag v2.3.2.post1, commit a14b1df).
Copyright (c) 2024 Albert Gu and Tri Dao. Licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0).
Changes: the head axis is split into (group, head in group) so that C.B^T is computed once per group of heads;
operands are promoted to one dtype with autocast disabled; padding to a whole number of chunks and the final state
returned for the cache follow transformers' torch path.

## Hugging Face transformers: `Mamba2Mixer.torch_forward` (Apache License 2.0)

`fast_torch_forward` in `mamba2_cpu_ssd.py` follows `Mamba2Mixer.torch_forward` in
`transformers/models/mamba2/modeling_mamba2.py` (transformers 4.57.6) outside the SSD core, which is replaced by
`ssd_chunked`. Copyright 2024 state-spaces/mamba2 org and HuggingFace Inc. team. Licensed under the Apache License,
Version 2.0. `FastMamba2Mixer` subclasses transformers' `Mamba2Mixer`.

## Data used to train and evaluate the checkpoints (not redistributed here)

The checkpoints in `checkpoints/` were trained and evaluated on the "P056" corpus, a human-written three-domain byte
corpus assembled for these experiments with a source-entity split and train/test overlap exclusion:

- Encyclopedia articles: WikiText-2 (raw), Merity et al. 2016 (Salesforce), derived from Wikipedia; CC BY-SA 3.0.
- Fiction: public-domain Project Gutenberg eBooks (the held-out test windows come from eBook #84, Mary Shelley's
  "Frankenstein", and #174, Oscar Wilde's "The Picture of Dorian Gray").
- Dialogue: OpenAssistant Conversations (oasst1) message trees; Apache License 2.0.

No corpus text is included in this folder; only model weights and aggregate scores are published.
