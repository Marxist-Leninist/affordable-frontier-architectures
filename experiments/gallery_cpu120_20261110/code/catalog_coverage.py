"""Report exact catalog and screen coverage; does not run gallery code."""
import collections
import hashlib
import json
from pathlib import Path
import sys

gallery_path, experiment, out = map(Path, sys.argv[1:4])
raw = gallery_path.read_bytes()
gallery = json.loads(raw)
runs = [json.loads(p.read_text()) for folder in ("results", "results_confirm")
        for p in (experiment / folder).glob("*.json")]
by_id = collections.defaultdict(list)
for r in runs:
    if str(r.get("arm", "")) != "transformer" and "wikitext_ce_nats" in r:
        by_id[str(r["arm"])].append(r)
rows = [{"index": r["index"], "name": r["name"],
         "program_sha256": hashlib.sha256(r["program"].encode()).hexdigest(),
         "completed_runs": len(by_id[str(r["index"])]),
         "seeds": sorted({x["seed"] for x in by_id[str(r["index"])]})}
        for r in sorted(gallery, key=lambda r: r["index"])]
report = {
    "schema": "asi_arch.catalog_coverage.v1",
    "gallery_sha256": hashlib.sha256(raw).hexdigest(),
    "upstream_commit": "3113c51b04a1d4a7520a3b0238c6a08d3194d58f",
    "records": len(rows), "unique_indices": len({r["index"] for r in rows}),
    "unique_names": len({r["name"] for r in rows}),
    "unique_program_hashes": len({r["program_sha256"] for r in rows}),
    "screened_records": sum(r["completed_runs"] > 0 for r in rows),
    "uncovered_indices": [r["index"] for r in rows if not r["completed_runs"]],
    "completed_run_files": len(runs),
    "interpretation": "105 exported source records, not proof of 105 semantically distinct architectures. No canonical subset resolving the historical 103 count is established. Existing runs are the adapted 350k-parameter, 120-CPU-second screen, not new training or full-scale results.",
    "rows": rows}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
