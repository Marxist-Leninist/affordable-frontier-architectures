"""Verify psfr_cache against the ORIGINAL full-prefix forward on the three trained v44 checkpoints.

Usage: python -I verify_psfr_cache.py <path to frontier_research_v44_asi>
Data: the package's pinned held-out FineWeb windows, rebuilt and hash-checked by restore_fineweb.py.
"""
import copy, hashlib, json, sys, time
from pathlib import Path

PKG = Path(sys.argv[1]).resolve()
WORK = Path(__file__).resolve().parent
sys.path[:0] = [str(PKG), str(WORK)]
import types
sys.modules.setdefault("resource", types.ModuleType("resource"))  # Linux-only import in source_v38; unused by the models
import numpy as np
import torch
from asi_models import build
from cpu_adapter import Cache
import psfr_cache as pc

torch.set_num_threads(4)
TOL = {torch.float64: 1e-10, torch.float32: 1e-4}
PREFILLS = (1, 31, 32, 33, 64, 100, 300, 511)
fw = torch.from_numpy(np.load(WORK / "fineweb_eval.npy").astype(np.int64))
longd = torch.from_numpy(np.load(WORK / "fineweb_long_docs.npy").astype(np.int64))
mx = lambda a, b: float((a - b).abs().max())


def load(seed):
    meta = json.loads((PKG / f"psfr_s{seed}.json").read_text())
    f = PKG / "checkpoints" / f"psfr_s{seed}.pt"
    assert hashlib.sha256(f.read_bytes()).hexdigest() == meta["checkpoint_sha256"]
    m = build(seed, "psfr")
    m.load_state_dict(torch.load(f, weights_only=True))
    return m.eval()


def decode_from(m, x, p, full, use_chunks=True):
    lg, st = pc.lm_step(m, x[:, :p], use_chunks=use_chunks)
    e_pre = mx(lg, full[:, :p])
    e_dec = 0.0
    for t in range(p, x.shape[1]):
        lg1, st = pc.lm_step(m, x[:, t:t + 1], st)
        e_dec = max(e_dec, mx(lg1[:, 0], full[:, t]))
    return e_pre, e_dec, st


results = {"tolerance": {"float64": TOL[torch.float64], "float32": TOL[torch.float32]}, "seeds": {}}
all_pass = True
with torch.inference_mode():
    for seed in (1, 2, 3):
        base = load(seed)
        r = {}
        # (A) the bug, reproduced exactly as the v44 audit measured it (first layer, first FineWeb window, 65 bytes)
        tok = fw[:1, :65]
        hidden = base.blocks[0].n1(base.emb(tok))
        layer = base.blocks[0].mix.layer
        full1 = layer(hidden)[0]
        cache = Cache()
        inc = torch.cat([layer(hidden[:, i:i + 1], past_key_values=cache, use_cache=True)[0] for i in range(65)], 1)
        st, fixed = None, []
        for i in range(65):
            o, st = pc.layer_step(layer, hidden[:, i:i + 1], st or pc.empty_state(layer, 1, hidden.dtype, hidden.device))
            fixed.append(o)
        r["first_layer_65_bytes"] = {"original_cache_max_abs": mx(full1, inc),
                                     "fixed_cache_max_abs": mx(full1, torch.cat(fixed, 1))}
        for dtype in (torch.float64, torch.float32):
            m = copy.deepcopy(base).to(dtype)
            x = fw[:3, :512]                               # batch 3, real held-out bytes
            full = m(x)
            rows = {}
            for p in PREFILLS:
                e_pre, e_dec, _ = decode_from(m, x, p, full)
                rows[str(p)] = {"prefill_max_abs": e_pre, "decode_max_abs": e_dec, "decoded_tokens": 512 - p}
            # uneven pieces exercising every head/body/tail combination
            g = torch.Generator().manual_seed(seed)
            cuts, a = [], 0
            while a < 512:
                n = int(torch.randint(1, 80, (1,), generator=g))
                cuts.append(min(n, 512 - a)); a += cuts[-1]
            st, outs, a = None, [], 0
            for n in cuts:
                lg, st = pc.lm_step(m, x[:, a:a + n], st); outs.append(lg); a += n
            pieces = mx(torch.cat(outs, 1), full)
            # chunk-form prefill and pure token-loop prefill must leave the same state
            _, st_c = pc.lm_step(m, x[:, :300], use_chunks=True)
            _, st_t = pc.lm_step(m, x[:, :300], use_chunks=False)
            state_gap = max(max(mx(a_, b_) for a_, b_ in zip(sc.tensors(), stt.tensors())) for sc, stt in zip(st_c, st_t))
            # longer than the 512-byte training context: prefill 1024, decode 64
            y = longd[:2, :1088]
            fully = m(y)
            l_pre, l_dec, _ = decode_from(m, y, 1024, fully)
            worst = max([v["prefill_max_abs"] for v in rows.values()] + [v["decode_max_abs"] for v in rows.values()]
                        + [pieces, l_pre, l_dec])
            ok = worst <= TOL[dtype]
            all_pass &= ok
            r[str(dtype).split(".")[-1]] = {"prefill_then_decode": rows, "uneven_pieces": {"cuts": cuts, "max_abs": pieces},
                                            "chunk_vs_token_state_gap_at_300": state_gap,
                                            "long_1024_prefill_64_decode": {"prefill_max_abs": l_pre, "decode_max_abs": l_dec},
                                            "worst_max_abs": worst, "pass": ok}
        # (E) greedy generation in FP32: cached decode vs full recompute at every step
        m = copy.deepcopy(base).float()
        prompt = fw[:3, :256]
        lg, st = pc.lm_step(m, prompt)
        cached = [lg[:, -1].argmax(-1)]
        for _ in range(63):
            lg, st = pc.lm_step(m, cached[-1][:, None], st)
            cached.append(lg[:, -1].argmax(-1))
        seq, recomputed, gen_gap = prompt, [], 0.0
        for i in range(64):
            lgf = m(seq)[:, -1]
            recomputed.append(lgf.argmax(-1))
            seq = torch.cat((seq, recomputed[-1][:, None]), 1)
        same = bool(torch.equal(torch.stack(cached, 1), torch.stack(recomputed, 1)))
        all_pass &= same
        r["greedy_64_tokens_batch3_fp32"] = {"identical_tokens": same, "tokens_compared": 3 * 64}
        r["state_bytes_fp32_batch1"] = pc.state_nbytes(pc.lm_step(m, prompt[:1, :8])[1])
        results["seeds"][str(seed)] = r
        print(json.dumps({"seed": seed, "first_layer": r["first_layer_65_bytes"],
                          "fp64_worst": r["float64"]["worst_max_abs"], "fp32_worst": r["float32"]["worst_max_abs"],
                          "greedy_identical": same}), flush=True)
results["all_pass"] = all_pass
(WORK / "verify_results.json").write_text(json.dumps(results, indent=2) + "\n")
print("ALL_PASS" if all_pass else "FAILURES", flush=True)
