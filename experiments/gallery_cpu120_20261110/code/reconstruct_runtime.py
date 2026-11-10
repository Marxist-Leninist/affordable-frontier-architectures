"""Rebuild the forward-pass counters of a run whose checkpoint was written before counters were kept.

Usage: reconstruct_runtime.py RESULTS_DIR GALLERY AUDIT TAG [TAG ...]

Some gallery programs count their own training forward passes (a non-persistent buffer or a plain integer)
and use the count in a schedule. A state_dict does not carry that, so the first checkpoints of this cohort
were missing it. A counter is a function of how many forward passes the live model made, which is the
number of updates in the result file, so it can be replayed: a fresh model of the same program makes that
many training-mode forward passes on a short input, and every integer buffer or attribute that moved is
taken as a counter. The values are added to the checkpoint under "__runtime__." keys and the result file
gets "runtime_state_reconstructed": true. Nothing else in either file changes.

This is only accepted if verify_weights.py then reproduces the recorded losses exactly; run it afterwards.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import models  # noqa: E402
import run_job  # noqa: E402


def integer_state(model):
    out = {}
    for key, value in run_job.runtime_state(model).items():
        kind = key[len(run_job.RUNTIME):].partition(".")[0]
        if kind == "buffer" and np.issubdtype(value.dtype, np.integer):
            out[key] = value
        elif kind == "attr" and value.dtype.kind in "iu":
            out[key] = value
    return out


def main():
    res, gallery_path, audit_path = sys.argv[1:4]
    res = pathlib.Path(res)
    torch.set_num_threads(1)
    gallery = models.load_gallery(gallery_path, audit_path)
    for tag in sys.argv[4:]:
        rec = json.loads((res / f"{tag}.json").read_text())
        saved = dict(np.load(res / f"{tag}.npz"))
        if any(k.startswith(run_job.RUNTIME) for k in saved):
            print(json.dumps({"tag": tag, "skipped": "checkpoint already has runtime state"}))
            continue
        model, _ = models.build(rec["arm"], rec["seed"], gallery)
        before = integer_state(model)
        model.train()
        x = torch.zeros((1, 96), dtype=torch.long)   # the length the smoke test proved every program accepts
        with torch.no_grad():
            for _ in range(rec["updates"]):
                run_job.bf16(model, x)
        after = integer_state(model)
        counters = {k: v for k, v in after.items() if not np.array_equal(v, before[k])}
        np.savez(res / f"{tag}.npz", **saved, **counters)
        rec["runtime_state_keys"] = sorted(counters)
        rec["runtime_state_reconstructed"] = True
        tmp = res / f"{tag}.json.tmp"
        tmp.write_text(json.dumps(rec, allow_nan=False) + "\n")
        tmp.rename(res / f"{tag}.json")
        print(json.dumps({"tag": tag, "updates": rec["updates"], "counters": {k: np.array(v).reshape(-1)[:3].tolist() for k, v in counters.items()}}))


if __name__ == "__main__":
    main()
