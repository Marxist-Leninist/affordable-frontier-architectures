"""Build every audited gallery model once and run one tiny forward pass under BF16 autocast.

Also compares the parameter count of each build with the counts recorded for the earlier 350k-parameter cohort,
as a check that this wrapper has the same shape as the one those runs used.
"""
import copy
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import models  # noqa: E402

torch.set_num_threads(1)
gallery = models.load_gallery(sys.argv[1], sys.argv[2])
RECORDED = {879: 349956, 458: 349944, 1354: 349962, 1311: 349944, 537: 349950, 1132: 349950, 1735: 350028,
            641: 349944, 1235: 349956, 712: 349980, 957: 349962, 1434: 349950, 556: 350040, 927: 349968,
            988: 349941, 1393: 349944, 908: 349950, 703: 350046}
m, info = models.build("transformer", 1)
rows = [{"arm": "transformer", "parameters": info["parameters"], "ffn_inner": info["ffn_inner"], "recorded": 349920}]
x = torch.randint(0, 256, (2, 96), generator=torch.Generator().manual_seed(7))
fails = {}
for idx in sorted(gallery):
    try:
        # BF16 autocast is the only mode every program supports: 43 of them name torch.bfloat16 in their own
        # code, and 39 of those fail outright in plain float32 (a float32 tensor meets a bfloat16 one).
        m, info = models.build(idx, 1, gallery)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            y16 = m.eval()(x)
        y32 = y16.float()
        info["hardcodes_bf16"] = "torch.bfloat16" in gallery[idx]["program"]
        if y32.shape != (2, 96, 256) or not torch.isfinite(y32).all() or not torch.isfinite(y16.float()).all():
            raise RuntimeError("bad output shape or non-finite logits")
        rows.append({"arm": idx, "name": info["name"], "parameters": info["parameters"], "ffn_inner": info["ffn_inner"],
                     "mixer_parameters": info["mixer_parameters"], "recorded": RECORDED.get(idx),
                     "hardcodes_bf16": info["hardcodes_bf16"], "score": gallery[idx]["score"]})
    except Exception as e:  # noqa: BLE001
        fails[idx] = [gallery[idx]["name"], repr(e)[:200]]
checked = [r for r in rows if r.get("recorded")]
out = {"audited_programs": len(gallery), "built_and_forward_ok": len(rows) - 1, "failed": fails,
       "recorded_counts_checked": len(checked), "recorded_counts_matching": sum(r["parameters"] == r["recorded"] for r in checked),
       "mismatches": [r for r in checked if r["parameters"] != r["recorded"]],
       "parameter_range": [min(r["parameters"] for r in rows), max(r["parameters"] for r in rows)], "rows": rows}
pathlib.Path(sys.argv[3]).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
