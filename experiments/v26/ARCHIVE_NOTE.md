# Archive and reproduction boundaries

Private source repository receives UTF8 source, reports, scalar JSON summaries, tests and manifests only. Binary losses.npz and subgroup.npy are retained in the private local ZIP, but are excluded from text-only repository upload. rows.json, model checkpoints, Python caches and raw copied dataset cards are excluded from both exports. The artifact manifest may list binary files absent from the text-only repository.

Recreate lawful pinned dataset rows with prepare_data.py, and prepare inherited v19/v23 checkpoints according to their upstream protocols. Recreate subgroup.npy and frozen manifests with run.py --freeze only in a new reproduction directory; do not overwrite original frozen results. Then execute preflight tests and run.py to recreate per-target losses. README provides full commands and limits. Data-card source links and recorded hashes remain available without redistributing cards or corpus.

Scientific source was frozen before scores. Reporting and independent audit scripts were added after freeze and do not modify evaluation. Exact content hashes, errors and pre-score clarifications are retained. Nothing here establishes frontier capability, distant reasoning or production profitability.
