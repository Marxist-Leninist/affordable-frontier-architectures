# The 18 model checkpoints are not in this folder yet

This folder was published on 11 November 2026 from the experiment's code-and-results package: 141 files, every one matching its SHA-256 in `FILE_MANIFEST.json`.

What is missing: the 18 PyTorch checkpoints listed in `CHECKPOINT_MANIFEST.json` (169,747,950 bytes in total; the manifest gives the path, size and SHA-256 of each). They exist only in the workspace where the six models were trained and had not been uploaded when this folder was published.

What that means for the other files here:

- `README.md` and `PUBLIC_RELEASE_MANIFEST.json` describe the full release, checkpoints included. `PUBLIC_RELEASE_MANIFEST.json` still says `"publication_performed": false`; it was written before publication and is kept unchanged.
- `DELIVERY_NOTE.md` says to obtain the checkpoints from the repository publication. This repository does not hold them yet.
- Everything numerical is here: training, evaluation and result receipts for all six runs, the original and the amended sweep results, the final reports and figures, and all source code.
- Steps that verify checkpoint hashes (for example regenerating the strict topology report) cannot be run from this folder alone until the checkpoints are added.

When the checkpoints are added, each file must match its entry in `CHECKPOINT_MANIFEST.json`, and this note should then be deleted.
