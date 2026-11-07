#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-reproduced_run}"
mkdir "$OUT"
cp models.py run.py run_one.py checks.py resume_remaining.sh download_data.py source_data_manifest.json "$OUT/"
cd "$OUT"
python download_data.py
python checks.py
python run.py --freeze
bash resume_remaining.sh
echo "Completed raw endpoints and checkpoints in $OUT. Equal-data endpoints reproduce exactly on the recorded CPU software stack; timing-budget endpoints vary with hardware."
