"""Build explicit source-ID coverage. No model execution, network, or queue writes."""
import argparse, collections, datetime, hashlib, json, math
from pathlib import Path

CATALOG_SHA = "aa77e8289971b3922102f000fddf44a6a9a31ca1f3b59c834722ac130f63759c"
SIM_PREFIX = "dc5c3c28ddbf1853"
# Explicit reviewed provenance mapping, not a regex that mistakes local variants for upstream IDs.
UPSTREAM_BY_PLUGIN = {
    "dlgm458": 458, "dlgm458_lr": 458, "dlgm458_lrk": 458, "dlgm458_hn": 458,
    "fusion879": 879, "fusion879_lr": 879, "fusion879_lrk": 879, "fusion879_hn": 879,
    "bamg259": 259,  # Reviewed prepared port; mapping alone never counts as training.
}
def digest(data):
    return hashlib.sha256(data).hexdigest()

def classify_run(record):
    if not isinstance(record, dict):
        return "non_run", ["not_an_object"]
    if record.get("plugin") == "upstream_tests" or record.get("steps") == 0:
        return "operator_check_only", ["no_training_steps"]
    if "plugin" not in record or "seed" not in record:
        return "non_run", ["not_a_seeded_training_record"]
    problems = []
    if record.get("ok") is not True:
        problems.append("not_successful")
    if record.get("steps") != 1200:
        problems.append("not_1200_updates")
    if record.get("objective") != "agillm-fit-v1":
        problems.append("different_or_missing_objective")
    if record.get("sim_sha256") != SIM_PREFIX:
        problems.append("different_or_missing_recorded_simulator_hash")
    if record.get("causality", {}).get("ok") is not True:
        problems.append("causality_not_passed")
    if record.get("skipped") != 0:
        problems.append("skipped_updates_or_unknown")
    final = next((x for x in record.get("curve", []) if x.get("step") == 1200), None)
    if final is None:
        problems.append("missing_fixed_endpoint")
    else:
        if final.get("tokens") != 2457600:
            problems.append("different_token_budget")
        for metric in ("ar", "sat", "nat"):
            value = final.get(metric)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                problems.append("nonfinite_or_missing_" + metric)
    cfg = record.get("cfg", {})
    for key, expected in {"V": 8192, "d": 256, "H": 4, "L": 12, "ctx": 256, "batch": 8}.items():
        if cfg.get(key) != expected:
            problems.append("different_shape_" + key)
    if problems:
        return "other_or_incomplete_training", problems
    return "eligible_agillm_simulator_run", []

def build(catalog, runs, cpu):
    raw = catalog.read_bytes()
    if digest(raw) != CATALOG_SHA:
        raise ValueError("catalog pin changed; review before counting")
    records = json.loads(raw)
    if len({x["index"] for x in records}) != len(records):
        raise ValueError("duplicate upstream IDs")
    observed = []
    for path in sorted(runs.glob("*.json")):
        data = path.read_bytes()
        try:
            row = json.loads(data)
        except (ValueError, UnicodeError):
            observed.append({"file": path.name, "sha256": digest(data), "classification": "unreadable_snapshot"})
            continue
        status, issues = classify_run(row)
        cfg = row.get("cfg") if isinstance(row, dict) else None
        observed.append({
            "file": path.name, "sha256": digest(data), "classification": status, "issues": issues,
            "original_record_sha256": row.get("original_record_sha256") if isinstance(row, dict) else None,
            "plugin": row.get("plugin") if isinstance(row, dict) else None,
            "seed": row.get("seed") if isinstance(row, dict) else None,
            "upstream_id": UPSTREAM_BY_PLUGIN.get(row.get("plugin")) if isinstance(row, dict) else None,
            "steps": row.get("steps") if isinstance(row, dict) else None,
            "config_sha256": digest(json.dumps(cfg, sort_keys=True).encode()) if cfg else None,
        })
    eligible = [r for r in observed if r["classification"] == "eligible_agillm_simulator_run"]
    tested = sorted({r["upstream_id"] for r in eligible if r["upstream_id"] is not None})
    cpu_metrics = json.loads(cpu.read_text())
    cpu_ids = {int(r["arm"]) for r in cpu_metrics["rows"] if str(r["arm"]) != "transformer"}
    coverage = []
    for record in sorted(records, key=lambda x: x["index"]):
        idx = record["index"]
        these = [r for r in eligible if r["upstream_id"] == idx]
        variants = {}
        for plugin in sorted({r["plugin"] for r in these}):
            seeds = sorted({r["seed"] for r in these if r["plugin"] == plugin})
            variants[plugin] = {"eligible_seeds": seeds, "three_seed_pilot_complete": seeds == [1, 2, 3]}
        coverage.append({
            "upstream_id": idx, "name": record.get("name_new", record["name"]), "internal_name": record["name"],
            "program_sha256": digest(record["program"].encode()),
            "agillm_status": "trained_adapted_port" if these else "prepared_not_trained" if idx == 259 else "not_yet_trained",
            "agillm_variants": variants, "agillm_run_files": [r["file"] for r in these],
            "historical_ar_cpu_screen": idx in cpu_ids,
            "next_gate": "no new catalog coverage from repeating this ID" if these else "GPU fidelity and safe admission before pilot" if idx == 259 else "source review, port, correctness and safe admission before pilot",
        })
    remaining = [r["upstream_id"] for r in coverage if r["upstream_id"] not in tested]
    return {
        "schema": "asi_arch.comparable_campaign_coverage.v2",
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_catalog": {"sha256": CATALOG_SHA, "commit": "3113c51b04a1d4a7520a3b0238c6a08d3194d58f",
            "records": len(records), "source_url": "https://github.com/GAIR-NLP/ASI-Arch/blob/3113c51b04a1d4a7520a3b0238c6a08d3194d58f/data/106.json"},
        "scope": {"requested_count": 103, "separately_supplied_103_id_list_found": False,
            "interpretation": "The manifest enumerates all 105 available source records. Exactly 103 remain after removing the two IDs with eligible AGILLM simulator port training. This is an explicit set difference, not an invented 103-total source list."},
        "counts": {"eligible_simulator_runs": len(eligible), "eligible_plugin_variants_including_controls": len({r["plugin"] for r in eligible}),
            "distinct_upstream_ids_with_agillm_training": len(tested), "upstream_ids_remaining": len(remaining),
            "historical_ar_cpu_screened_ids": len(cpu_ids), "historical_ar_cpu_runs": cpu_metrics["jobs_finished"]},
        "tested_upstream_ids": tested, "remaining_upstream_ids": remaining,
        "comparability": {
            "objective": "agillm-fit-v1", "primary_metric": "0.6 AR + 0.2 SAT + 0.2 NAT",
            "updates": 1200, "training_token_targets": 2457600,
            "shape": {"V":8192,"d":256,"H":4,"L":12,"ctx":256,"batch":8},
            "validity": "Successful record, fixed endpoint, finite AR/SAT/NAT, causality passed, zero skipped updates, recorded simulator hash matches.",
            "limits": ["Adapted small simulator, not full-scale results.", "Historical run files store only a 16-character simulator hash, no per-run plugin/data hashes.",
                "Current source/data hashes cannot retroactively prove historical bytes.", "Full, LR, LRK and head-normalized variants are not additional upstream architectures.",
                "Timing and port adaptations differ; do not call these measured full-scale equal-compute results.", "The AR-only byte-level CPU screen has different tokenization, objectives, precision and budgets; losses are not directly comparable."]},
        "plugin_to_upstream_id": UPSTREAM_BY_PLUGIN, "coverage": coverage, "run_evidence": observed,
        "next_experiment": {"candidate_id":259,"candidate_name":"BAMG_MemoryGate","status":"prepared_not_queued",
            "conditions":["Existing owner handoff and idle safe lane","GPU fidelity against pinned source adaptation","Verified serving health and conservative memory admission, fail closed on unknown health",
                "Fixed config, data/source hashes and seed schedule published before launch","One bounded pilot first; no additional rentals or live-training interruption"],
            "not_a_winner_claim":True}
    }
if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("catalog",type=Path);p.add_argument("runs",type=Path);p.add_argument("cpu_metrics",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args(); result=build(a.catalog,a.runs,a.cpu_metrics);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"counts":result["counts"],"tested":result["tested_upstream_ids"],"remaining":len(result["remaining_upstream_ids"])}))
