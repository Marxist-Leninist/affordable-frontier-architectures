"""Pick the programs that get a confirmation run, by a rule fixed before the screen had finished.

Usage: make_confirm_queue.py RESULTS_DIR OUT_QUEUE.json [how_many, default 8]
Rule: the gallery programs with the lowest seed-1 sum of the two held-out losses, among those that passed
every audit and fit the size budget, leaving out the two gallery controls (#879, #458), which already have
three seeds. Each chosen program is queued for seeds 2 and 3, and so are the three controls (again), so the
confirmation is one block of runs made together.
"""
import glob
import json
import pathlib
import sys

res = pathlib.Path(sys.argv[1])
k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
rows = [json.loads(pathlib.Path(f).read_text()) for f in glob.glob(str(res / "*_s1.json"))]


def eligible(r):
    audits = all(r[a][c] for a in ("audit_before", "audit_after") for c in ("causal", "batch_independent", "finite"))
    return r["arm"] not in ("transformer", "879", "458") and audits and abs(r["parameters"] - 350_000) <= 525


ranked = sorted((r for r in rows if eligible(r)), key=lambda r: r["wikitext_ce_nats"] + r["fineweb_edu_ce_nats"])
chosen = ranked[:k]
# The three controls are run again in the same block, spread among the candidates, so that both sides of every
# comparison were measured in the same stretch of time on the same machine.
queue = []
for s in (2, 3):
    arms = [int(r["arm"]) for r in chosen]
    controls = ["transformer", 458, 879]
    step = max(1, len(arms) // len(controls))
    block = []
    for i, arm in enumerate(arms):
        if i % step == 0 and controls:
            block.append(controls.pop(0))
        block.append(arm)
    block += controls
    queue += [{"arm": arm, "seed": s, "save_weights": True} for arm in block]
pathlib.Path(sys.argv[2]).write_text(json.dumps(queue, indent=1) + "\n")
print(json.dumps({"seed_1_runs_seen": len(rows), "eligible": len(ranked),
                  "chosen": [(r["arm"], r["name"], round(r["wikitext_ce_nats"] + r["fineweb_edu_ce_nats"], 4)) for r in chosen],
                  "next_not_chosen": [(r["arm"], round(r["wikitext_ce_nats"] + r["fineweb_edu_ce_nats"], 4)) for r in ranked[k:k + 2]]}))
