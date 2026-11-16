# Frozen weights used in v69

These are the exact six step-800 checkpoints used in the v69 numerical/timing experiment, not new v69 training. Three seeds for each of GGG Gated DeltaNet (165,996 parameters) and AAAwide Transformer (190,848 parameters). FP32 tensor-only state dictionaries, 256-byte vocabulary. Total 4,349,880 bytes; no optimizer state, corpus text or secrets.

SHA-256 and Git blob hashes, sizes and training steps are in weights_manifest.json and match ../protocol.json. Local safe loading used torch.load(..., map_location="cpu", weights_only=True), all entries finite tensors and counts match the model. Raw weights do not run by themselves: model definitions from the prior v43/v56 recovered implementation are required; ../run.py documents that runtime layout. These tiny byte language models are research artifacts, not chat models or frontier systems. v69 changed only inference code, preserving every checkpoint byte.

Training provenance: the prior protocol/source manifest records Salesforce/wikitext, wikitext-2-raw-v1, revision b08601e04326c79dfdd32d625aee71d232d685c3. Byte tokenization and original training/evaluation provenance are retained in ../source_data_manifest.json, ../protocol.json and preceding experiment reports. The evaluation timing cohort came from HuggingFaceFW/fineweb-edu, not new v69 training. No raw source text is included.

## Attribution and licence status

WikiText by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher, https://huggingface.co/datasets/Salesforce/wikitext and https://arxiv.org/abs/1609.07843. Dataset card checked 2026-11-16 labels CC-BY-SA-3.0 and GFDL. FineWeb-Edu dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu, ODC-BY. Those are source dataset licences, not an assertion that an identical licence automatically attaches to model weights. No new model licence or repository licence is assigned here; the user authorized public publication, not a licence change. Please verify applicable permissions before downstream redistribution or commercial use.

Published to the user's existing GitHub repository after explicit approval; no Hugging Face upload or quota deletion was needed.
