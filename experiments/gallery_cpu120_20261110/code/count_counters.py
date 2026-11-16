"""Which gallery programs keep a forward-pass counter outside their state_dict?

Usage: count_counters.py GALLERY AUDIT OUT.json
Each program is built once and makes two training-mode forward passes on a short input; any integer buffer
or attribute that moved is reported.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import models  # noqa: E402
import run_job  # noqa: E402
from reconstruct_runtime import integer_state  # noqa: E402

torch.set_num_threads(1)
gallery = models.load_gallery(sys.argv[1], sys.argv[2])
found = {}
x = torch.zeros((1, 96), dtype=torch.long)
for idx in sorted(gallery):
    model, _ = models.build(idx, 1, gallery)
    before = integer_state(model)
    model.train()
    with torch.no_grad():
        run_job.bf16(model, x)
        run_job.bf16(model, x)
    after = integer_state(model)
    moved = sorted({k.rsplit(".", 1)[-1] for k, v in after.items() if not np.array_equal(v, before[k])})
    if moved:
        found[idx] = {"name": gallery[idx]["name"], "counters": moved}
pathlib.Path(sys.argv[3]).write_text(json.dumps({"programs": len(gallery), "with_counters": len(found), "which": found}, indent=1) + "\n")
print(json.dumps({"programs": len(gallery), "with_counters": len(found), "indices": sorted(found)}))
