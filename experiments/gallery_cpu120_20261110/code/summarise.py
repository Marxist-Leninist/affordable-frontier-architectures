"""Turn the per-job result files into METRICS.json and a results table (RESULTS.md).

Usage: summarise.py RESULTS_DIR V44_LEDGER COHORT_SETS OUT_DIR [CONFIRM_DIR]

Comparisons with the Transformer control are paired by held-out window. WikiText windows from one article are
resampled together (cluster bootstrap); FineWeb-Edu windows are one per document. Intervals are conditional on
the seed that was run and on these reused held-out windows.

Programs that were also run on seeds 2 and 3 get a confirmation section. They were picked for that by their
seed-1 result, so seed 1 flatters them; the comparison that counts uses seeds 2 and 3 only, against the same
seeds of the controls (same seed = same initialisation recipe and the same training windows in the same order).
With CONFIRM_DIR, seeds 2 and 3 of candidates and controls alike are read from there: one block of runs made
together after the screen, so that a drift in the machine's speed over the evening cannot favour either side.
The controls' seeds 2 and 3 then exist twice (screen block and confirmation block), which also shows how much a
result moves when the same seed is simply run again.
"""
import glob
import json
import pathlib
import sys

import numpy as np

res_dir, ledger_path, sets_path, out_dir = (pathlib.Path(p) for p in sys.argv[1:5])
confirm_dir = pathlib.Path(sys.argv[5]) if len(sys.argv) > 5 else None
ledger = json.loads(ledger_path.read_text())
sets = json.loads(sets_path.read_text())
seen_before = set(sets["previously_screened_by_chatgpt_campaign"])
wiki_docs = np.array([w["doc"] for w in ledger["wiki"]])
rows = [json.loads(pathlib.Path(f).read_text()) for f in sorted(glob.glob(str(res_dir / "*.json")))]
failed = sorted(pathlib.Path(f).name[:-7] for f in glob.glob(str(res_dir / "*.failed")))
rng = np.random.default_rng(20261110)


def boot_wiki(delta, n=4000):
    docs = np.unique(wiki_docs)
    groups = [delta[wiki_docs == d] for d in docs]
    out = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        out[i] = np.concatenate([groups[j] for j in pick]).mean()
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def boot_plain(delta, n=4000):
    idx = rng.integers(0, len(delta), (n, len(delta)))
    m = delta[idx].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def key(r):
    return (r["arm"], r["seed"])


by = {key(r): r for r in rows}
controls = {}
for arm, label in (("transformer", "Transformer w96 (control)"), ("879", "#879 FusionGatedFIRNet (control)"), ("458", "#458 DualScaleMemoryRouter (control)")):
    rs = [by[(arm, s)] for s in (1, 2, 3) if (arm, s) in by]
    if rs:
        w = np.array([r["wikitext_ce_nats"] for r in rs]); f = np.array([r["fineweb_edu_ce_nats"] for r in rs])
        controls[arm] = {"label": label, "seeds": [r["seed"] for r in rs], "parameters": rs[0]["parameters"],
                         "wikitext_ce": w.tolist(), "fineweb_edu_ce": f.tolist(),
                         "wikitext_mean": float(w.mean()), "fineweb_edu_mean": float(f.mean()),
                         "wikitext_sd": float(w.std(ddof=1)) if len(w) > 1 else None,
                         "fineweb_edu_sd": float(f.std(ddof=1)) if len(f) > 1 else None,
                         "updates": [r["updates"] for r in rs]}

table = []
t_runs = [by[("transformer", s)] for s in (1, 2, 3) if ("transformer", s) in by]
t_ref = None
if t_runs:   # the reference is the Transformer's loss on each window, averaged over its seeds
    t_ref = {"wikitext": np.mean([r["wikitext_window_ce_nats"] for r in t_runs], 0),
             "fineweb_edu": np.mean([r["fineweb_edu_window_ce_nats"] for r in t_runs], 0), "seeds": [r["seed"] for r in t_runs]}
for r in rows:
    if r["seed"] != 1:
        continue
    row = {k: r[k] for k in ("arm", "name", "tag", "parameters", "updates", "byte_targets", "training_cpu_seconds",
                             "byte_targets_per_cpu_second", "wikitext_ce_nats", "fineweb_edu_ce_nats", "peak_rss_mib",
                             "status", "gallery_score", "state_sha256", "source_sha256")}
    row["wikitext_ce_at_60s"] = r["mid_checkpoints"][0]["wikitext_ce_nats"] if r["mid_checkpoints"] else None
    row["fineweb_edu_ce_at_60s"] = r["mid_checkpoints"][0]["fineweb_edu_ce_nats"] if r["mid_checkpoints"] else None
    row["causality_max_abs_diff"] = max(r["audit_before"]["causality_max_abs_diff"], r["audit_after"]["causality_max_abs_diff"])
    row["batch_independence_max_abs_diff"] = max(r["audit_before"]["batch_independence_max_abs_diff"], r["audit_after"]["batch_independence_max_abs_diff"])
    row["audits_pass"] = all(r[a][k] for a in ("audit_before", "audit_after") for k in ("causal", "batch_independent", "finite"))
    row["previously_screened"] = r["arm"] != "transformer" and int(r["arm"]) in seen_before
    row["within_size_budget"] = abs(r["parameters"] - 350_000) <= 525
    if t_ref is not None and r["arm"] != "transformer":
        dw = np.array(r["wikitext_window_ce_nats"]) - t_ref["wikitext"]
        df = np.array(r["fineweb_edu_window_ce_nats"]) - t_ref["fineweb_edu"]
        row["vs_transformer"] = {"reference_seeds": t_ref["seeds"], "wikitext_delta": float(dw.mean()), "wikitext_ci95": boot_wiki(dw),
                                 "fineweb_edu_delta": float(df.mean()), "fineweb_edu_ci95": boot_plain(df)}
    table.append(row)

gallery = [r for r in table if r["arm"] != "transformer"]
ranked = sorted([r for r in gallery if r["audits_pass"] and r["within_size_budget"]],
                key=lambda r: r["wikitext_ce_nats"] + r["fineweb_edu_ce_nats"])
for i, r in enumerate(ranked):
    r["rank_by_sum"] = i + 1
CONTROL_ARMS = ("transformer", "879", "458")
CONFIRM_SEEDS = (2, 3)


def windows(r, name):
    return np.array(r[f"{name}_window_ce_nats"])


confirm_rows = rows if confirm_dir is None else [json.loads(pathlib.Path(f).read_text()) for f in sorted(glob.glob(str(confirm_dir / "*.json")))]
cby = {key(r): r for r in confirm_rows}      # seeds 2 and 3 come from here, for candidates and controls alike
confirmation = {}
for arm in sorted({r["arm"] for r in confirm_rows}, key=lambda a: (a != "transformer", int(a) if a != "transformer" else 0)):
    if (arm, 1) not in by or not all((arm, sd) in cby for sd in CONFIRM_SEEDS):
        continue
    rs = {1: by[(arm, 1)], **{sd: cby[(arm, sd)] for sd in CONFIRM_SEEDS}}
    entry = {"name": rs[1]["name"], "parameters": rs[1]["parameters"], "is_control": arm in CONTROL_ARMS,
             "per_seed": {str(sd): {"wikitext_ce": rs[sd]["wikitext_ce_nats"], "fineweb_edu_ce": rs[sd]["fineweb_edu_ce_nats"],
                                    "updates": rs[sd]["updates"]} for sd in sorted(rs)},
             "selection_seed_1": {"wikitext_ce": rs[1]["wikitext_ce_nats"], "fineweb_edu_ce": rs[1]["fineweb_edu_ce_nats"]},
             "confirmation_mean": {"seeds": list(CONFIRM_SEEDS),
                                   "wikitext_ce": float(np.mean([rs[sd]["wikitext_ce_nats"] for sd in CONFIRM_SEEDS])),
                                   "fineweb_edu_ce": float(np.mean([rs[sd]["fineweb_edu_ce_nats"] for sd in CONFIRM_SEEDS]))},
             "three_seed_mean": {"wikitext_ce": float(np.mean([rs[sd]["wikitext_ce_nats"] for sd in (1, 2, 3)])),
                                 "fineweb_edu_ce": float(np.mean([rs[sd]["fineweb_edu_ce_nats"] for sd in (1, 2, 3)])),
                                 "wikitext_sd": float(np.std([rs[sd]["wikitext_ce_nats"] for sd in (1, 2, 3)], ddof=1)),
                                 "fineweb_edu_sd": float(np.std([rs[sd]["fineweb_edu_ce_nats"] for sd in (1, 2, 3)], ddof=1))}}
    if arm not in CONTROL_ARMS:
        entry["versus"] = {}
        for ctl in CONTROL_ARMS:
            if not all((ctl, sd) in cby for sd in CONFIRM_SEEDS):
                continue
            per_seed = {str(sd): {"wikitext_delta": rs[sd]["wikitext_ce_nats"] - cby[(ctl, sd)]["wikitext_ce_nats"],
                                  "fineweb_edu_delta": rs[sd]["fineweb_edu_ce_nats"] - cby[(ctl, sd)]["fineweb_edu_ce_nats"]}
                        for sd in CONFIRM_SEEDS}
            dw = np.mean([windows(rs[sd], "wikitext") - windows(cby[(ctl, sd)], "wikitext") for sd in CONFIRM_SEEDS], 0)
            df = np.mean([windows(rs[sd], "fineweb_edu") - windows(cby[(ctl, sd)], "fineweb_edu") for sd in CONFIRM_SEEDS], 0)
            entry["versus"][ctl] = {
                "per_seed": per_seed, "wikitext_delta": float(dw.mean()), "wikitext_ci95": boot_wiki(dw),
                "fineweb_edu_delta": float(df.mean()), "fineweb_edu_ci95": boot_plain(df),
                "better_on_both_sets_in_every_confirmation_seed": all(v["wikitext_delta"] < 0 and v["fineweb_edu_delta"] < 0 for v in per_seed.values()),
                "worse_on_both_sets_in_every_confirmation_seed": all(v["wikitext_delta"] > 0 and v["fineweb_edu_delta"] > 0 for v in per_seed.values())}
    confirmation[arm] = entry

# the same control, the same seed, run twice: once in the screen block and once in the confirmation block
repeats = []
if confirm_dir is not None:
    for ctl in CONTROL_ARMS:
        for sd in CONFIRM_SEEDS:
            if (ctl, sd) in by and (ctl, sd) in cby:
                a, b = by[(ctl, sd)], cby[(ctl, sd)]
                repeats.append({"arm": ctl, "seed": sd, "updates": [a["updates"], b["updates"]],
                                "wikitext_ce": [a["wikitext_ce_nats"], b["wikitext_ce_nats"]],
                                "fineweb_edu_ce": [a["fineweb_edu_ce_nats"], b["fineweb_edu_ce_nats"]],
                                "wikitext_abs_diff": abs(a["wikitext_ce_nats"] - b["wikitext_ce_nats"]),
                                "fineweb_edu_abs_diff": abs(a["fineweb_edu_ce_nats"] - b["fineweb_edu_ce_nats"])})

metrics = {"cohort": "gallery_cpu120_20261110", "jobs_finished": len(rows) + (len(confirm_rows) if confirm_dir is not None else 0),
           "jobs_failed": failed, "gallery_programs_finished": len(gallery), "controls": controls,
           "confirmation": confirmation, "same_seed_run_twice": repeats, "rows": table,
           "note": "One seed per gallery program; three seeds for the controls. CE in nats per byte, lower is better."}
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "METRICS.json").write_text(json.dumps(metrics, indent=1) + "\n")


def fmt(v, n=4):
    return "" if v is None else f"{v:.{n}f}"


lines = ["# Results table", "",
         "Cross-entropy in nats per byte on the held-out windows after 120 seconds of training CPU (lower is better). "
         "One seed per gallery program. `vs T` is the paired difference, window by window, from the Transformer control "
         "averaged over its three seeds (negative is better than the Transformer); an asterisk means the 95% interval "
         "excludes zero on that set. The interval covers the choice of held-out windows only, not the choice of seed: "
         "use the controls' seed-to-seed spread below as the yardstick for that.", ""]
if controls:
    lines += ["## Controls, three seeds", "", "| Control | Parameters | WikiText CE (mean, sd) | FineWeb-Edu CE (mean, sd) | Updates in 120 s |", "| --- | ---: | ---: | ---: | ---: |"]
    for c in controls.values():
        lines.append(f"| {c['label']} | {c['parameters']:,} | {fmt(c['wikitext_mean'])} ({fmt(c['wikitext_sd'])}) | "
                     f"{fmt(c['fineweb_edu_mean'])} ({fmt(c['fineweb_edu_sd'])}) | {', '.join(str(u) for u in c['updates'])} |")
    lines.append("")
candidates = {a: c for a, c in confirmation.items() if not c["is_control"]}
if candidates:
    def label(a):
        return {"transformer": "Transformer w96", "879": "#879 Fusion", "458": "#458 DLGM"}.get(a, f"#{a} {confirmation[a]['name'].replace('delta_net_', '')}")

    def signed(v, ci):
        star = "*" if (ci[1] < 0 or ci[0] > 0) else ""
        return f"{v:+.4f}{star}"

    lines += ["## Confirmation on seeds 2 and 3", "",
              "The best programs of the seed-1 screen were run again on seeds 2 and 3, in one block together with fresh runs of the "
              "three controls on the same seeds. Seed 1 is what the candidates were picked on, so it is shown but not counted. "
              "Means below are over seeds 2 and 3 of that block only, for the candidates and the controls alike.", "",
              "| Model | Parameters | Seed 1 (selection) WikiText / FineWeb-Edu | Seeds 2-3 mean WikiText | Seeds 2-3 mean FineWeb-Edu | Updates (seeds 1, 2, 3) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    order = sorted(confirmation, key=lambda a: confirmation[a]["confirmation_mean"]["wikitext_ce"] + confirmation[a]["confirmation_mean"]["fineweb_edu_ce"])
    for a in order:
        c = confirmation[a]
        lines.append(f"| {label(a)}{' (control)' if c['is_control'] else ''} | {c['parameters']:,} | {fmt(c['selection_seed_1']['wikitext_ce'])} / {fmt(c['selection_seed_1']['fineweb_edu_ce'])} | "
                     f"{fmt(c['confirmation_mean']['wikitext_ce'])} | {fmt(c['confirmation_mean']['fineweb_edu_ce'])} | "
                     f"{', '.join(str(c['per_seed'][str(sd)]['updates']) for sd in (1, 2, 3))} |")
    lines += ["", "Differences from each control on seeds 2 and 3 (candidate minus control, negative is better; an asterisk means the "
              "95% interval over held-out windows excludes zero; \"every seed\" says whether the sign is the same on both sets in both seeds):", "",
              "| Candidate | vs #458 DLGM WikiText | vs #458 DLGM FineWeb-Edu | every seed | vs #879 Fusion WikiText | vs #879 Fusion FineWeb-Edu | every seed | vs Transformer WikiText | vs Transformer FineWeb-Edu | every seed |",
              "| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for a in [a for a in order if a in candidates]:
        cells = [label(a)]
        for ctl in ("458", "879", "transformer"):
            v = candidates[a]["versus"].get(ctl)
            if not v:
                cells += ["", "", ""]
                continue
            cells += [signed(v["wikitext_delta"], v["wikitext_ci95"]), signed(v["fineweb_edu_delta"], v["fineweb_edu_ci95"]),
                      "better" if v["better_on_both_sets_in_every_confirmation_seed"] else "worse" if v["worse_on_both_sets_in_every_confirmation_seed"] else "mixed"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if repeats:
        lines += ["The same control with the same seed, run twice (once during the screen, once in the confirmation block). The two runs "
                  "start identically and differ only in how many updates fit into 120 CPU-seconds, so this is the size of pure timing noise:", "",
                  "| Control | Seed | Updates (screen, confirmation) | WikiText CE (screen, confirmation) | FineWeb-Edu CE (screen, confirmation) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for r in repeats:
            lines.append(f"| {label(r['arm'])} | {r['seed']} | {r['updates'][0]}, {r['updates'][1]} | {fmt(r['wikitext_ce'][0])}, {fmt(r['wikitext_ce'][1])} | "
                         f"{fmt(r['fineweb_edu_ce'][0])}, {fmt(r['fineweb_edu_ce'][1])} |")
        lines.append("")
lines += ["## Gallery programs, seed 1, best first (sum of the two losses)", "",
          "| Rank | Gallery # | Program | Parameters | Updates | WikiText CE | vs T | FineWeb-Edu CE | vs T | WikiText CE at 60 s | Screened before |",
          "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
for r in ranked:
    v = r.get("vs_transformer") or {}
    sw = "*" if v and (v["wikitext_ci95"][1] < 0 or v["wikitext_ci95"][0] > 0) else ""
    sf = "*" if v and (v["fineweb_edu_ci95"][1] < 0 or v["fineweb_edu_ci95"][0] > 0) else ""
    lines.append(f"| {r['rank_by_sum']} | {r['arm']} | {r['name'].replace('delta_net_', '')} | {r['parameters']:,} | {r['updates']} | "
                 f"{fmt(r['wikitext_ce_nats'])} | {fmt(v.get('wikitext_delta'), 3) + sw if v else ''} | {fmt(r['fineweb_edu_ce_nats'])} | "
                 f"{fmt(v.get('fineweb_edu_delta'), 3) + sf if v else ''} | {fmt(r['wikitext_ce_at_60s'])} | {'yes' if r['previously_screened'] else 'no'} |")
left_out = [r for r in gallery if not (r["audits_pass"] and r["within_size_budget"])]
if left_out or failed:
    lines += ["", "## Not ranked", ""]
    for r in left_out:
        why = [] if r["audits_pass"] else ["an audit failed"]
        if not r["within_size_budget"]:
            why.append(f"cannot be matched to the size budget ({r['parameters']:,} parameters)")
        lines.append(f"- #{r['arm']} {r['name']}: {'; '.join(why)}. WikiText {fmt(r['wikitext_ce_nats'])}, FineWeb-Edu {fmt(r['fineweb_edu_ce_nats'])}.")
    for t in failed:
        lines.append(f"- {t}: the job did not finish (see its .stderr file).")
(out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"jobs": len(rows), "failed": failed, "ranked": len(ranked), "not_ranked": len(left_out),
                  "top5": [(r["arm"], r["name"], round(r["wikitext_ce_nats"], 4), round(r["fineweb_edu_ce_nats"], 4)) for r in ranked[:5]],
                  "controls": {k: (round(v["wikitext_mean"], 4), round(v["fineweb_edu_mean"], 4)) for k, v in controls.items()}}))
