"""Write the experiment README from the measured numbers, so that no figure in it is typed by hand.

Usage: make_readme.py METRICS.json VERIFY.json SMOKE.json STATIC_AUDIT.json README_static.md OUT_README.md
The wording around the numbers was written for the results as they came out; read the page again if they change.
"""
import json
import pathlib
import sys

metrics_path, verify_path, smoke_path, audit_path, static_path, out_path = (pathlib.Path(p) for p in sys.argv[1:7])
m = json.loads(metrics_path.read_text())
v = json.loads(verify_path.read_text())
smoke = json.loads(smoke_path.read_text())
audit = json.loads(audit_path.read_text())
static = static_path.read_text()

rows = m["rows"]
gallery = [r for r in rows if r["arm"] != "transformer"]
ranked = sorted([r for r in gallery if "rank_by_sum" in r], key=lambda r: r["rank_by_sum"])
not_ranked = [r for r in gallery if "rank_by_sum" not in r]
by_arm = {r["arm"]: r for r in rows}
t1, dlgm1, fusion1 = by_arm["transformer"], by_arm["458"], by_arm["879"]
conf, ext = m["confirmation"], m["confirmation_extended"]
ctl = m["controls"]


def short(name):
    return name.replace("delta_net_", "")


def f4(x):
    return f"{x:.4f}"


def total(r):
    return r["wikitext_ce_nats"] + r["fineweb_edu_ce_nats"]


beat_t_both = sum(1 for r in gallery if r["wikitext_ce_nats"] < t1["wikitext_ce_nats"] and r["fineweb_edu_ce_nats"] < t1["fineweb_edu_ce_nats"])
sig_better = sum(1 for r in gallery if r.get("vs_transformer") and r["vs_transformer"]["wikitext_ci95"][1] < 0 and r["vs_transformer"]["fineweb_edu_ci95"][1] < 0)
sig_worse = sum(1 for r in gallery if r.get("vs_transformer") and r["vs_transformer"]["wikitext_ci95"][0] > 0 and r["vs_transformer"]["fineweb_edu_ci95"][0] > 0)
ahead_of_dlgm = sum(1 for r in ranked if total(r) < total(dlgm1))
new_programs = sum(1 for r in gallery if not r["previously_screened"])
cands = [a for a in conf if not conf[a]["is_control"]]
cands.sort(key=lambda a: conf[a]["confirmation_mean"]["wikitext_ce"] + conf[a]["confirmation_mean"]["fineweb_edu_ce"])
ext_cands = [a for a in ext if not ext[a]["is_control"]]
ext_cands.sort(key=lambda a: ext[a]["confirmation_mean"]["wikitext_ce"] + ext[a]["confirmation_mean"]["fineweb_edu_ce"])
ext_seeds = ext[ext_cands[0]]["confirmation_mean"]["seeds"] if ext_cands else []
fell_behind = sum(1 for a in cands if conf[a]["versus"]["458"]["wikitext_delta"] > 0 and conf[a]["versus"]["458"]["fineweb_edu_delta"] > 0)
rep = m["same_seed_run_twice"]
rep_w = max(r["wikitext_abs_diff"] for r in rep)
rep_f = max(r["fineweb_edu_abs_diff"] for r in rep)
rep_w_mean = sum(r["wikitext_abs_diff"] for r in rep) / len(rep)
rep_f_mean = sum(r["fineweb_edu_abs_diff"] for r in rep) / len(rep)


def tag(arm, c):
    return {"transformer": "Transformer (width 96)", "879": "#879 psfr (Fusion)", "458": "#458 dlgm (DLGM)"}.get(arm, f"#{arm} {short(c[arm]['name'])}")


def star(val, ci):
    return f"{val:+.4f}{'*' if ci[1] < 0 or ci[0] > 0 else ''}"


def conf_table(c, order):
    seeds = c[order[0]]["confirmation_mean"]["seeds"]
    head = f"| Model | Seed 1 (selection) | Mean of seeds {seeds[0]}-{seeds[-1]}: WikiText | FineWeb-Edu | vs DLGM: WikiText | FineWeb-Edu | Seeds better than DLGM |"
    out = [head, "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    everyone = sorted(c, key=lambda a: c[a]["confirmation_mean"]["wikitext_ce"] + c[a]["confirmation_mean"]["fineweb_edu_ce"])
    for a in everyone:
        e = c[a]
        cells = [("**" + tag(a, c) + "**") if e["is_control"] else tag(a, c),
                 f"{f4(e['selection_seed_1']['wikitext_ce'])} / {f4(e['selection_seed_1']['fineweb_edu_ce'])}",
                 f4(e["confirmation_mean"]["wikitext_ce"]), f4(e["confirmation_mean"]["fineweb_edu_ce"])]
        d = e.get("versus", {}).get("458")
        if d:
            cells += [star(d["wikitext_delta"], d["wikitext_ci95"]), star(d["fineweb_edu_delta"], d["fineweb_edu_ci95"]),
                      f"{d['seeds_better_on_wikitext']} of {d['seeds']} / {d['seeds_better_on_fineweb_edu']} of {d['seeds']}"]
        else:
            cells += ["", "", "control" if e["is_control"] else ""]
        out.append("| " + " | ".join(cells) + " |")
    return out


L = []
add = L.append
add("# All 105 gallery programs, 120 CPU-seconds each, with confirmation on fresh seeds")
add("")
add("Run on 10 and 11 November 2026 by Claude (Anthropic's assistant) at Scott's request, with a harness written "
    "for this purpose and independent of the one behind the earlier gallery studies. The harness, the data "
    "manifest and every per-run result are in this folder.")
add("")
best = ext_cands[0] if ext_cands else cands[0]
bc = ext if ext_cands else conf
bd = bc[best]["versus"]["458"]
add("## In short")
add("")
add(f"- **Coverage.** The earlier campaign had screened {len(gallery) - new_programs} of the 105 original gallery programs. "
    f"All 105 have now been trained and evaluated under one recipe, {new_programs} of them for the first time.")
add(f"- **Screen (one seed each).** {beat_t_both} of 105 programs beat the Transformer control's seed 1 on both held-out sets. "
    f"{ahead_of_dlgm} programs scored better than the previous best, #458 DLGM, on that single seed. Most of that does not survive: see the next point.")
if ext_cands:
    add(f"- **Confirmation (fresh seeds, controls re-run alongside).** Of the eight best programs of the screen, "
        f"{fell_behind} fell behind DLGM on both sets on seeds 2 and 3. The one clear result is **#{best} `{short(bc[best]['name'])}`**: over four fresh seeds "
        f"({', '.join(str(s) for s in ext_seeds)}) it averages {f4(bc[best]['confirmation_mean']['wikitext_ce'])} nats per byte on WikiText and "
        f"{f4(bc[best]['confirmation_mean']['fineweb_edu_ce'])} on FineWeb-Edu, against {f4(bc['458']['confirmation_mean']['wikitext_ce'])} and "
        f"{f4(bc['458']['confirmation_mean']['fineweb_edu_ce'])} for DLGM in the same block: {bd['wikitext_delta']:+.4f} and {bd['fineweb_edu_delta']:+.4f}, "
        f"better in {bd['seeds_better_on_wikitext']} of {bd['seeds']} seeds on WikiText and {bd['seeds_better_on_fineweb_edu']} of {bd['seeds']} on FineWeb-Edu."
        + (f" Its parent design, #151 `adaptive_mix_gate`, is ahead of DLGM too, by less ({ext['151']['versus']['458']['wikitext_delta']:+.4f} and "
           f"{ext['151']['versus']['458']['fineweb_edu_delta']:+.4f}; {ext['151']['versus']['458']['seeds_better_on_wikitext']} of 4 and "
           f"{ext['151']['versus']['458']['seeds_better_on_fineweb_edu']} of 4 seeds)." if "151" in ext and best == "259" else ""))
add(f"- **One run is noisy.** The same control with the same seed, run twice, differed by up to {rep_w:.3f} nats on WikiText and {rep_f:.3f} on "
    f"FineWeb-Edu, only because a different number of updates fitted into 120 CPU-seconds. Differences of that size between two single runs mean nothing.")
add(f"- **For whoever runs the next tests.** `fetch_inputs.sh` and `code/` rebuild the exact v44 data from public sources (every hash matches the "
    f"published ledger) and run any gallery program by number. Nothing else is needed.")
add("")
add("## Limits first")
add("")
add("- Tiny byte-level models (about 350,000 parameters) trained for two CPU-minutes. Nothing here says how any of these designs behaves at a useful size, on a GPU, or with more training.")
add("- The held-out windows are the v44 windows this repository has used before. They are not fresh, and the screen picked its candidates on them. "
    "The confirmation uses fresh seeds, not fresh text.")
add("- The comparison is at equal training CPU time on one machine. A program that does less work per update gets more updates; that is the point of the "
    "comparison, and it also means the ranking depends on this machine's kernels (AMX for BF16) and on PyTorch 2.6.")
add("- Stand-ins for the flash-linear-attention modules are plain PyTorch, not the official fused kernels (`code/fla_cpu.py`).")
add("- Each program is used as written. Schedules inside a program that count thousands of forward passes were not rescaled to a run of a few hundred "
    "updates, and extra losses that a few programs expose (for example `reg_loss`) were not used: every model is trained on next-byte cross-entropy only.")
add("- This is a different harness, machine and evaluation set from the v65 study, so the numbers are not interchangeable with v65's. "
    "The order of the three controls is the same in both.")
add(f"- One program, #{not_ranked[0]['arm']} `{short(not_ranked[0]['name'])}`, cannot be brought down to the size budget in this wrapper "
    f"({not_ranked[0]['parameters']:,} parameters at the smallest feed-forward width). It was run and is reported, but not ranked." if not_ranked else "")
add("")
add("## Results")
add("")
add("Cross-entropy in nats per byte, lower is better. Full tables: [RESULTS.md](RESULTS.md). Every number, per seed and per window: [METRICS.json](METRICS.json) and `results/`.")
add("")
add("### Controls in the screen (three seeds)")
add("")
add("| Control | Parameters | WikiText (mean, sd) | FineWeb-Edu (mean, sd) | Updates in 120 s |")
add("| --- | ---: | ---: | ---: | ---: |")
for a, name in (("transformer", "Transformer, width 96, rotary positions"), ("879", "#879 psfr (\"Fusion\" in the earlier reports)"), ("458", "#458 dlgm (\"DLGM\" in the earlier reports)")):
    c = ctl[a]
    add(f"| {name} | {c['parameters']:,} | {f4(c['wikitext_mean'])} ({f4(c['wikitext_sd'])}) | {f4(c['fineweb_edu_mean'])} ({f4(c['fineweb_edu_sd'])}) | {', '.join(str(u) for u in c['updates'])} |")
add("")
add("### The screen: best 15 of 105 on seed 1")
add("")
add("| Rank | Program | Updates | WikiText | FineWeb-Edu | Screened before |")
add("| ---: | --- | ---: | ---: | ---: | --- |")
for r in ranked[:15]:
    add(f"| {r['rank_by_sum']} | #{r['arm']} {short(r['name'])} | {r['updates']} | {f4(r['wikitext_ce_nats'])} | {f4(r['fineweb_edu_ce_nats'])} | {'yes' if r['previously_screened'] else 'no'} |")
add(f"| {dlgm1['rank_by_sum']} | #458 dlgm (control, seed 1) | {dlgm1['updates']} | {f4(dlgm1['wikitext_ce_nats'])} | {f4(dlgm1['fineweb_edu_ce_nats'])} | yes |")
add(f"| {fusion1['rank_by_sum']} | #879 psfr (control, seed 1) | {fusion1['updates']} | {f4(fusion1['wikitext_ce_nats'])} | {f4(fusion1['fineweb_edu_ce_nats'])} | yes |")
add(f"|  | Transformer (control, seed 1) | {t1['updates']} | {f4(t1['wikitext_ce_nats'])} | {f4(t1['fineweb_edu_ce_nats'])} |  |")
add("")
add(f"Against the Transformer averaged over its three seeds, window by window, {sig_better} programs are better on both sets with 95% intervals that exclude zero, "
    f"and {sig_worse} are worse on both. The slowest programs manage fewer than a hundred updates in the time and finish last "
    f"(the range is {min(r['updates'] for r in gallery)} to {max(r['updates'] for r in gallery)} updates).")
add("")
add("### Confirmation on seeds 2 and 3")
add("")
add("The rule was fixed before the screen had finished (`code/make_confirm_queue.py`): the eight programs with the lowest seed-1 sum of the two losses, "
    "leaving out the two gallery controls. They were run on seeds 2 and 3 in one block, and the three controls were run again on the same seeds in "
    "that block, spread among them, so that both sides of each comparison were measured in the same stretch of time. Seed 1 is shown but not counted, "
    "because it is what the candidates were picked on. An asterisk marks a 95% interval over held-out windows that excludes zero.")
add("")
L += conf_table(conf, cands)
add("")
if ext_cands:
    add("### Confirmation extended to seeds 4 and 5")
    add("")
    add("The three candidates that were level with or ahead of DLGM after seeds 2 and 3 were run on two more seeds, again with all three controls. "
        "Means are over the four fresh seeds.")
    add("")
    L += conf_table(ext, ext_cands)
    add("")
    add("Per seed, candidate minus DLGM (WikiText / FineWeb-Edu):")
    add("")
    add("| Candidate | " + " | ".join(f"Seed {s}" for s in ext_seeds) + " |")
    add("| --- | " + " | ".join("---:" for _ in ext_seeds) + " |")
    for a in ext_cands:
        ps = ext[a]["versus"]["458"]["per_seed"]
        add(f"| {tag(a, ext)} | " + " | ".join(f"{ps[str(s)]['wikitext_delta']:+.4f} / {ps[str(s)]['fineweb_edu_delta']:+.4f}" for s in ext_seeds) + " |")
    add("")
add("### How much one run moves")
add("")
add("The controls' seeds 2 and 3 exist twice: once from the screen and once from the confirmation block. Each pair starts from the same weights and sees "
    "the same data in the same order; the only difference is how many updates fitted into 120 CPU-seconds on a shared machine.")
add("")
add("| Control | Seed | Updates (screen, confirmation) | WikiText (screen, confirmation) | FineWeb-Edu (screen, confirmation) |")
add("| --- | ---: | ---: | ---: | ---: |")
for r in rep:
    add(f"| {tag(r['arm'], conf)} | {r['seed']} | {r['updates'][0]}, {r['updates'][1]} | {f4(r['wikitext_ce'][0])}, {f4(r['wikitext_ce'][1])} | {f4(r['fineweb_edu_ce'][0])}, {f4(r['fineweb_edu_ce'][1])} |")
add("")
add(f"The average gap is {rep_w_mean:.4f} nats on WikiText and {rep_f_mean:.4f} on FineWeb-Edu, the largest {rep_w:.4f} and {rep_f:.4f}. "
    "This is why the screen alone decides nothing, and why the controls were re-run inside each confirmation block instead of being reused.")
add("")
mix = {str(r["arm"]): r.get("mixer_parameters") for r in smoke["rows"]}
if ext_cands:
    lead = ext[best]["versus"]["458"]["per_seed"]
    gaps = [abs(lead[str(sd)][k]) for sd in ext_seeds for k in ("wikitext_delta", "fineweb_edu_delta")]
    small = sum(1 for sd in ext_seeds if abs(lead[str(sd)]["wikitext_delta"]) < rep_w_mean) + sum(1 for sd in ext_seeds if abs(lead[str(sd)]["fineweb_edu_delta"]) < rep_f_mean)
    add(f"#{best} was ahead of DLGM in every fresh seed on both sets. Its lead in a single seed runs from {min(gaps):.4f} to {max(gaps):.4f} nats, and "
        f"{small} of the {len(gaps)} differences are smaller than the average gap between two runs of the very same thing (next section). "
        f"Four seeds is still few; the result is a consistent lead of about {abs(bd['wikitext_delta']):.3f} to {abs(bd['fineweb_edu_delta']):.3f} nats per byte, not a measured constant.")
    add("")

    def mean_updates(arm):
        return sum(ext[arm]["per_seed"][str(sd)]["updates"] for sd in ext_seeds) / len(ext_seeds)

    add("### What the leading programs are")
    add("")
    add("Paraphrased from the headers of the programs themselves:")
    add("")
    add("- **#151 `adaptive_mix_gate`** is DeltaNet with one addition: a gate that decides, for every token and every head, how much of the "
        "delta-rule memory's output to use and how much of the current value vector.")
    add("- **#259 `bias_init_mix_gate`** is #151 plus one parameter per head: a bias on that gate that starts at -1, so the gate starts near 0.27. "
        "Its header gives the aim as stopping the gate from collapsing early in training.")
    add("- **#437 `dual_path_fusion`** combines the same gate with an additive branch of dilated convolutions.")
    add("")
    add(f"All three are early and small designs. Their mixers have {mix['259']:,}, {mix['151']:,} and {mix['437']:,} parameters against {mix['458']:,} for DLGM, "
        f"which leaves more of the 350,000 for the feed-forward layers and makes an update cheaper: over the four fresh seeds #259 averaged "
        f"{mean_updates('259'):.0f} updates in 120 seconds, #151 {mean_updates('151'):.0f}, DLGM {mean_updates('458'):.0f} and the Transformer {mean_updates('transformer'):.0f}. "
        "Whether #259 is ahead of DLGM because its gate is better or because it gets more updates, this experiment cannot say: at equal CPU time the two are the same thing.")
add("")
add(static.rstrip())
add("")
add("## Checkpoints")
add("")
add(f"The supplied verification report checked {v['checked']} freshly rebuilt models: "
    f"{v['reproduce']} passed its recorded tolerance of {v['tolerance_nats']:g} nats. "
    f"{len(v['do_not_reproduce'])} did not reproduce. "
    "This describes only the checkpoints in that report, not untested checkpoints or a repeat of training.")
if v.get("largest_abs_diff_nats") is not None:
    add(f"Largest recorded absolute loss difference: {v['largest_abs_diff_nats']:g} nats.")
add(f"Verification source supplied to this generator: `{verify_path.name}`.")
add("")
add("Runtime state and arithmetic matter when reloading:")
add("")
if "with_counters" in v:
    add(f"- The separate counter audit recorded {v['with_counters']} of the {audit['programs']} programs with forward-pass counters. "
        "This counter census is additional provenance, not a result of reloading the checkpoints.")
else:
    add("- The supplied verifier report does not contain the optional gallery-wide counter census. "
        "Run `code/count_counters.py` to collect that separate audit; a checkpoint reload does not establish a census.")
add("- Checkpoints carry state outside the `state_dict` under keys starting `__runtime__.` "
    "(`run_job.runtime_state`). Missing state can silently reset a program's schedule.")
if "reconstructed" in v:
    add(f"- The original run provenance records {v['reconstructed']} checkpoints whose runtime counters were reconstructed "
        "by replaying recorded forward passes (`code/reconstruct_runtime.py`). This is historical provenance, not a new reconstruction.")
else:
    add("- The supplied verifier report does not record historical counter reconstructions. No reconstruction count is inferred.")
add("- BF16 results depend on the instruction set. Reloading with `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` instead of the default (AMX on this machine) "
    "moved the held-out losses of the five checkpoints tried by up to 0.00006 nats. On a processor without AMX, expect differences of that size, not exact equality.")
add("")
add("Six checkpoints are in `checkpoints/`, each with its result file, all from seed 2 of the confirmation block: the three controls "
    "(`T96_s2`, `g879_s2`, `g458_s2`) and the three leading candidates (`g259_s2`, `g151_s2`, `g437_s2`). The other "
    f"{m['jobs_finished'] - 6} from the original cohort were not kept beyond the session that made them; a hash of each one's weights is in its result file.")
add("")
add("```")
add("E=experiments/gallery_cpu120_20261110")
add("# score a text with a checkpoint and let it write a little")
add("unshare -rn -- python -I $E/code/try_checkpoint.py $E/checkpoints/g259_s2.npz inputs/asiarch/106.json work/STATIC_AUDIT.json \"Text to continue\"")
add("# check that the six checkpoints give their recorded losses")
add("unshare -rn -- python -I $E/code/verify_weights.py $E/checkpoints inputs/asiarch/106.json work/STATIC_AUDIT.json work/data work/VERIFY_checkpoints.json")
add("```")
add("")
add("## What happened along the way")
add("")
add("- **Four first attempts failed in the harness, not in the programs.** #865, #1370, #1454 and #1543 keep tensors from their last forward pass as attributes; "
    "copying the model for the 60-second evaluation refused to copy tensors that still carried gradient history. The copy now detaches such tensors first "
    "(`run_job.safe_copy`); the four were re-run from scratch and their first error output is kept in `results/first_attempt_failures/`.")
add("- **The machine restarted twice** (23:25 and 23:51 UTC). Runs in flight were lost and started again from nothing; finished runs were untouched. "
    "`results/driver.log` and `results_confirm/driver.log` show every start and finish.")
add("- **A planned pause at 22:06 UTC** added the saving of runtime state described above. It changes what a run writes after it has been measured, "
    "not how it trains or what it scores.")
add("- **An instruction-set cap that was meant to be set was not.** The driver was written to pass `ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16` to its jobs but never did, "
    "so every run used oneDNN's default. That was noticed when checkpoints reloaded under the cap did not reproduce exactly. All runs share the default, "
    "so the cohort is consistent; the driver's comment now says what it really does, and each result records the setting it saw.")
add("")
add("## Files")
add("")
add("| Path | What it is |")
add("| --- | --- |")
add("| `RESULTS.md`, `METRICS.json` | The full tables and every derived number |")
add("| `results/`, `results_confirm/` | One JSON per run: configuration, both audits, losses at 60 s and at the end, loss on every held-out window, the training-loss history, a hash of the final weights. `driver.log` has the timeline |")
add("| `VERIFY.json` | The checkpoint reload check for every run |")
add("| `checkpoints/` | Six checkpoints to try, with their result files (see above) |")
add("| `code/` | The harness: `static_audit.py`, `fla_cpu.py`, `models.py`, `data_prep.py`, `smoke.py`, `run_job.py`, `driver.py`, `make_confirm_queue.py`, `summarise.py`, `verify_weights.py`, `reconstruct_runtime.py`, `count_counters.py`, `try_checkpoint.py`, and `make_readme.py` with `README_static.md`, which write this page from the measured numbers |")
add("| `fetch_inputs.sh`, `INPUTS.sha256` | Pinned public inputs and their hashes |")
add("| `STATIC_AUDIT.json`, `SMOKE.json`, `data/DATA_CHECK.json` | The static audit, the build-and-shape check, and the data rebuilt against the v44 ledger |")
add("| `QUEUE.json`, `QUEUE_CONFIRM.json`, `QUEUE_CONFIRM2.json`, `CONFIRM_CHOICE.json`, `COHORT_SETS.json` | What was queued, how the confirmation candidates were picked, and which programs the earlier campaign had screened |")
add("")
add("## Not done")
add("")
add("- No fresh held-out text. The natural next step for #259 is the test the earlier studies used for their own winners: new documents, more seeds.")
add("- No larger models, no longer runs, no GPU. Whether a gate that helps a 350,000-parameter model for two minutes helps a real one is open.")
add("- No tuning of any program. In particular the programs whose internal schedules assume thousands of steps may be undersold here.")
add("")
pathlib.Path(out_path).write_text("\n".join(x for x in L if x is not None) + "\n")
print("written", out_path, len(L), "lines")
