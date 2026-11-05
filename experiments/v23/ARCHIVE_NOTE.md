# Repository reproduction scope

The repository retains tested source, protocols, exact data/checkpoint/source hashes, aggregate numeric results, training logs, six passing test results and independent audits. Reproduce in a workspace using the sibling directory names documented in README. Restore historical v19-v22 source with archives/20261105_v19_v22/extract.py, and lawfully prepare prior WikiText/checkpoints according to their reports.

Corpus rows.json and model checkpoints are intentionally excluded. Regenerate train_starts.npy with run.py --freeze and losses.npz with the bounded run.py procedure; the full scalar-loss array remains in the complete local research bundle, not this text-focused repository delivery. Run summarize.py and audit_results.py after regeneration. The original artifact manifest identifies the full run and may include binaries absent here. No claim that installing dependencies alone recreates inherited training is made.
