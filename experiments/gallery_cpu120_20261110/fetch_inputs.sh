#!/bin/bash
# Fetch the pinned public inputs of this experiment and check their hashes.
# None of them is redistributed in this repository: the gallery programs belong to the ASI-Arch project and
# the text belongs to its datasets. Usage: bash fetch_inputs.sh [directory, default "inputs"]
set -euo pipefail
OUT="${1:-inputs}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT/asiarch" "$OUT/hf"
curl -fsSL -o "$OUT/asiarch/106.json" \
  "https://raw.githubusercontent.com/GAIR-NLP/ASI-Arch/3113c51b04a1d4a7520a3b0238c6a08d3194d58f/data/106.json"
for split in train validation test; do
  curl -fsSL -o "$OUT/hf/wt2_$split.parquet" \
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/$split-00000-of-00001.parquet"
done
# The same request the v44 study recorded (experiments/asi_arch_psfr_cache_v1/v44_inputs/fineweb_request.json).
# It returned byte-identical content on 8 and on 10 November 2026; if the service ever answers differently,
# the hash check below fails rather than letting different text through.
curl -fsSL -o "$OUT/hf/fineweb_rows.json" \
  "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW%2Ffineweb-edu&config=sample-10BT&split=train&offset=2500&length=100"
(cd "$OUT" && sha256sum -c "$HERE/INPUTS.sha256")
