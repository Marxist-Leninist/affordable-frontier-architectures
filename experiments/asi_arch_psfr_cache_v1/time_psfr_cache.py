"""Timing: PSFR cached decode vs PSFR full-prefix recompute vs the matched Transformer's KV-cached decode.

CPU, FP32, one thread, inference mode. Three seeds x prompt lengths 128/512/1024 x 4 held-out FineWeb documents
x 3 repeats, arm order shuffled per repeat (paired). Each arm produces the same 33 next-byte distributions:
a prompt of P bytes, then 32 more real bytes fed one at a time.
Usage: python -I time_psfr_cache.py <path to frontier_research_v44_asi>
"""
import json, random, statistics, sys, time
from pathlib import Path

if sys.platform == "win32":  # shared laptop: yield to interactive apps (BELOW_NORMAL_PRIORITY_CLASS)
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)

PKG = Path(sys.argv[1]).resolve()
WORK = Path(__file__).resolve().parent
sys.path[:0] = [str(PKG), str(WORK)]
import types
sys.modules.setdefault("resource", types.ModuleType("resource"))  # Linux-only import in source_v38; unused by the models
import numpy as np
import torch
from asi_models import build
from prior_models import state_bytes as kv_bytes
import psfr_cache as pc

torch.set_num_threads(1)
STEPS, CONTEXTS, DOCS, REPEATS = 32, (128, 512, 1024), 4, 3
docs = torch.from_numpy(np.load(WORK / "fineweb_long_docs.npy").astype(np.int64))[:DOCS]
rng = random.Random(879)


def load(seed, arm):
    m = build(seed, arm)
    m.load_state_dict(torch.load(PKG / "checkpoints" / f"{arm}_s{seed}.pt", weights_only=True))
    return m.eval().float()


def run(arm, models, x, p):
    t0 = time.perf_counter()
    if arm == "psfr_cached":
        lg, st = pc.lm_step(models["psfr"], x[:, :p])
        t1 = time.perf_counter()
        for i in range(STEPS):
            lg, st = pc.lm_step(models["psfr"], x[:, p + i:p + i + 1], st)
        state = pc.state_nbytes(st)
    elif arm == "transformer_cached":
        lg, st = models["transformer"](x[:, :p], cache=True)
        t1 = time.perf_counter()
        for i in range(STEPS):
            lg, st = models["transformer"](x[:, p + i:p + i + 1], st, cache=True)
        state = kv_bytes(st)[0]
    else:
        m = models[arm.split("_")[0]]
        m(x[:, :p])
        t1 = time.perf_counter()
        for i in range(STEPS):
            lg = m(x[:, :p + i + 1])
        state = 0
    t2 = time.perf_counter()
    assert torch.isfinite(lg).all()
    return {"prefill_s": t1 - t0, "decode_s": t2 - t1, "total_s": t2 - t0, "state_bytes": state}


ARMS = ("psfr_cached", "psfr_full", "transformer_cached", "transformer_full")
rows = []
with torch.inference_mode():
    for seed in (1, 2, 3):
        models = {"psfr": load(seed, "psfr"), "transformer": load(seed, "transformer")}
        # sanity: the Transformer's own cache must also match its full forward
        x = docs[:1]
        full = models["transformer"](x[:, :160])
        lg, st = models["transformer"](x[:, :128], cache=True)
        gap = float((lg - full[:, :128]).abs().max())
        for i in range(32):
            lg, st = models["transformer"](x[:, 128 + i:129 + i], st, cache=True)
            gap = max(gap, float((lg[:, 0] - full[:, 128 + i]).abs().max()))
        assert gap < 1e-4, gap
        for p in CONTEXTS:
            for arm in ARMS:
                run(arm, models, docs[:1], p)                       # warm-up
            for d in range(DOCS):
                for rep in range(REPEATS):
                    order = list(ARMS)
                    rng.shuffle(order)
                    for arm in order:
                        rows.append({"seed": seed, "context": p, "doc": d, "repeat": rep, "arm": arm,
                                     **run(arm, models, docs[d:d + 1], p)})
            print(json.dumps({"seed": seed, "context": p, "rows": len(rows)}), flush=True)

agg = {}
for p in CONTEXTS:
    sel = [r for r in rows if r["context"] == p]
    pick = lambda arm, k: [r[k] for r in sel if r["arm"] == arm]
    med = lambda v: statistics.median(v)
    pairs = {}
    for r in sel:
        pairs.setdefault((r["seed"], r["doc"], r["repeat"]), {})[r["arm"]] = r
    ratio = lambda a, b, k: med([z[a][k] / z[b][k] for z in pairs.values()])
    agg[str(p)] = {
        "per_token_decode_ms": {arm: 1000 * med(pick(arm, "decode_s")) / STEPS for arm in ARMS},
        "prefill_ms": {arm: 1000 * med(pick(arm, "prefill_s")) for arm in ARMS},
        "paired_ratios": {
            "psfr_full_over_psfr_cached_decode": ratio("psfr_full", "psfr_cached", "decode_s"),
            "psfr_full_over_psfr_cached_total": ratio("psfr_full", "psfr_cached", "total_s"),
            "psfr_cached_over_transformer_cached_decode": ratio("psfr_cached", "transformer_cached", "decode_s"),
            "psfr_cached_over_transformer_cached_total": ratio("psfr_cached", "transformer_cached", "total_s"),
            "psfr_cached_prefill_over_psfr_full_forward": ratio("psfr_cached", "psfr_full", "prefill_s"),
            "transformer_full_over_transformer_cached_decode": ratio("transformer_full", "transformer_cached", "decode_s"),
        },
        "state_bytes_after_prompt_plus_32": {"psfr_cached": pick("psfr_cached", "state_bytes")[0],
                                              "transformer_cached": pick("transformer_cached", "state_bytes")[0]},
    }
out = {"scope": "CPU FP32, 1 thread, batch 1, torch " + torch.__version__ + "; 3 seeds x 4 held-out FineWeb docs x 3 "
                "randomised paired repeats per prompt length; 32 real bytes fed after the prompt. Laptop shared with "
                "other work, so absolute ms are noisy; ratios are paired medians.",
       "aggregate": agg, "raw": rows}
(WORK / "timing_results.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(agg, indent=1), flush=True)
