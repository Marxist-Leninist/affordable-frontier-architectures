"""Reload saved weights into a freshly built model and check that it gives the recorded held-out losses.

Usage: verify_weights.py RESULTS_DIR GALLERY AUDIT DATA OUT_JSON [tag ...]
With no tags, every <tag>.npz in RESULTS_DIR is checked. Run it the way the jobs were run (same Python, same
ONEDNN_MAX_CPU_ISA, no network): the losses are compared at 1e-6 nats, which only holds on the same arithmetic.

A checkpoint is the model's state_dict plus, under keys starting "__runtime__.", what the program keeps outside
it (see run_job.runtime_state). Both are restored before the losses are recomputed.
"""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import models  # noqa: E402
import run_job  # noqa: E402


def main():
    res, gallery_path, audit_path, data, out = sys.argv[1:6]
    res, data = pathlib.Path(res), pathlib.Path(data)
    tags = sys.argv[6:] or sorted(p.stem for p in res.glob("*.npz"))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    z = np.load(data / "evaluation.npz")
    heldout = {"wikitext": torch.from_numpy(z["wiki"].astype(np.int64)),
               "fineweb_edu": torch.from_numpy(z["fineweb"].astype(np.int64))}
    gallery = models.load_gallery(gallery_path, audit_path)
    rows = []
    for tag in tags:
        rec = json.loads((res / f"{tag}.json").read_text())
        row = {"tag": tag, "arm": rec["arm"], "seed": rec["seed"], "name": rec["name"]}
        try:
            model, info = models.build(rec["arm"], rec["seed"], gallery)
            everything = dict(np.load(res / f"{tag}.npz"))
            saved = {k: v for k, v in everything.items() if not k.startswith(run_job.RUNTIME)}
            extra = {k: v for k, v in everything.items() if k.startswith(run_job.RUNTIME)}
            row["runtime_state_entries"] = len(extra)
            digest = hashlib.sha256()
            for k in sorted(saved):
                digest.update(k.encode())
                digest.update(np.ascontiguousarray(saved[k]).tobytes())
            row["file_matches_recorded_state_sha256"] = digest.hexdigest() == rec["state_sha256"]
            row["parameters"] = info["parameters"]
            row["parameters_match"] = info["parameters"] == rec["parameters"]
            own = model.state_dict()
            row["same_keys"] = sorted(own) == sorted(saved)
            model.load_state_dict({k: torch.from_numpy(v).to(own[k].dtype).reshape(own[k].shape) for k, v in saved.items()}, strict=True)
            run_job.restore_runtime_state(model, extra)
            ev = run_job.evaluate(model, heldout)
            row["wikitext_ce_nats"] = ev["wikitext"]["ce_nats"]
            row["fineweb_edu_ce_nats"] = ev["fineweb_edu"]["ce_nats"]
            row["wikitext_abs_diff"] = abs(ev["wikitext"]["ce_nats"] - rec["wikitext_ce_nats"])
            row["fineweb_edu_abs_diff"] = abs(ev["fineweb_edu"]["ce_nats"] - rec["fineweb_edu_ce_nats"])
            row["worst_window_abs_diff"] = float(max(
                np.abs(np.array(ev[n]["window_ce_nats"]) - np.array(rec[f"{n}_window_ce_nats"])).max() for n in heldout))
            row["reproduces"] = bool(row["file_matches_recorded_state_sha256"] and row["parameters_match"] and row["same_keys"]
                                     and row["wikitext_abs_diff"] <= 1e-6 and row["fineweb_edu_abs_diff"] <= 1e-6
                                     and row["worst_window_abs_diff"] <= 2e-6)
        except Exception as exc:  # report it against this checkpoint and carry on with the rest
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            row["reproduces"] = False
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("tag", "reproduces", "wikitext_abs_diff", "fineweb_edu_abs_diff", "error")}), flush=True)
    summary = {"checked": len(rows), "reproduce": sum(r["reproduces"] for r in rows),
               "do_not_reproduce": [r["tag"] for r in rows if not r["reproduces"]],
               "tolerance_nats": 1e-6, "rows": rows}
    pathlib.Path(out).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("checked", "reproduce", "do_not_reproduce")}))


if __name__ == "__main__":
    main()
