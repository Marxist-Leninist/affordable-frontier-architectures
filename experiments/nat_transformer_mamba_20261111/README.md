# Initial NAT Transformer versus Mamba baseline (800 updates)

Completed November 11, 2026. This is the initial 800-update pilot, separate from longer optimization work. See [full report](nat_compare_20261111/REPORT.md).

The tiny Transformer trained in 9.89 seconds versus 90.64 seconds for the unfused bidirectional Mamba-1 CPU reference. Mamba had slightly lower held-out text loss and higher sorting accuracy; Transformer was better on copying and reversing. This is not a fused GPU benchmark or a measure of general intelligence. No 600B change is justified by this pilot.

## Contents and reproduction

Code, protocols, metrics, audits and all 16 small FP32 state-dict checkpoints the original local experiment produced are included, in nat_compare_20261111/checkpoints/ where the scripts load them. All six final 800-update checkpoints, all six near-five-second snapshots and all four tuning checkpoints (learning rates 0.001 and 0.003) are present. The two unselected tuning checkpoints (tune_mamba_0.001.pt and tune_transformer_0.001.pt) were held back from the first checkpoint commit by a publication approval blocker and added afterwards with the repository owner's approval. Each final_* and tune_* file matches checkpoint_sha256 in its result JSON file, and FILE_MANIFEST.json lists the SHA256 of all 16. Final checkpoints are named final_{transformer,mamba}_{1,2,3}.pt; *_time.pt are near-five-second snapshots; tune_* are tuning runs. Each model has approximately 94,000 parameters. They use byte inputs, 32 prefix positions and 16 blank suffix positions, predicting all 16 suffix bytes in one pass. They are research artifacts, not useful conversational models.

Use Python 3.12 and the pinned requirements in nat_compare_20261111/requirements.txt. From frontier_research_v8 run python prepare_data.py. Then follow nat_compare_20261111/REPORT.md. Load checkpoints with torch.load(path, weights_only=True, map_location="cpu") and the NAT class in models.py; no remote code loader is required. Prefer running reproduction in a fresh copy so original results remain intact.

## Public-release audit

This is a sanitized copy, not a modification of the original scientific artifacts. Apart from this publication README, only source_manifest.json removes private source filesystem paths and incidental live-host GPU occupancy. Model code, checkpoints and numerical results are unchanged. Python caches, transfer helpers, and the original aggregate hash file are excluded. FILE_MANIFEST.json gives SHA256 hashes for this public copy. No WikiText corpus/parquet files, private AGILLM source, credentials or personal data are redistributed.

See nat_compare_20261111/THIRD_PARTY_NOTICES.md for Mamba attribution and the source dataset's licensing ambiguity. This publication does not assign a new license to user code or checkpoints and does not change repository licensing, visibility or permissions.
