"""One training job: build a model, audit it, train it for a fixed amount of training CPU time, evaluate.

Recipe (the published v65 recipe as far as it was published, on the published v44 data contract):
  data      first 2,000,000 bytes of line-deduplicated WikiText-2-raw train; batch 4 x context 512; the same
            sampled windows for every arm that shares a seed
  optimiser AdamW lr 0.003, betas 0.9/0.999, weight decay 0.01, gradient clipping at 1, cross-entropy only;
            linear warm-up over 32 updates, then constant (the v65 schedule itself was not published)
  precision BF16 CPU autocast with FP32 master weights
  stop      the first completed update at or past BUDGET seconds of training CPU; the clock covers minibatch
            construction through the optimiser step and the finite check, and nothing else
  evaluate  at each mark and at the end: mean cross-entropy (nats per byte) of the last 128 targets of every
            held-out window with its full 512-byte context, 183 WikiText test windows and 99 FineWeb-Edu windows

Audits run before and after training on copies of the model, under the same BF16 autocast:
  causality          bytes at and after a cut are replaced; logits before the cut must not move
  batch independence a row's logits must not depend on what else is in the batch
"""
import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import platform
import resource
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import models  # noqa: E402

BATCH, CONTEXT, LR, WARMUP = 4, 512, 0.003, 32
CUTS = (1, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 400, 511)
AUDIT_TOLERANCE = 1e-4


def bf16(model, x):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return model(x)


_MODULE_INTERNALS = frozenset(vars(torch.nn.Module()))


def _detached(value):
    """(value with autograd history removed from any tensor inside it, whether anything changed)."""
    if torch.is_tensor(value):
        return (value.detach(), True) if value.grad_fn is not None else (value, False)
    if type(value) in (dict, list, tuple):
        items = list(value.items()) if type(value) is dict else list(enumerate(value))
        pairs = [(k, _detached(v)) for k, v in items]
        if not any(changed for _, (_, changed) in pairs):
            return value, False
        if type(value) is dict:
            return {k: v for k, (v, _) in pairs}, True
        return type(value)(v for _, (v, _) in pairs), True
    return value, False


def safe_copy(model):
    """A full copy of the model, runtime state included.

    Some gallery programs keep the last forward pass's tensors as attributes. Those still carry autograd
    history after a training step, which deepcopy refuses, so they are detached first (the values are
    unchanged, and the history of a finished step is never used again). torch.nn.Module's own
    bookkeeping attributes are left alone.
    """
    for module in model.modules():
        for name, value in list(vars(module).items()):
            if name in _MODULE_INTERNALS:
                continue
            new, changed = _detached(value)
            if changed:
                object.__setattr__(module, name, new)
    return copy.deepcopy(model)


RUNTIME = "__runtime__."


def runtime_state(model):
    """What a program keeps outside its state_dict, so a checkpoint can restore it too.

    Some gallery programs count their own forward passes in a non-persistent buffer or a plain attribute and
    use the count in a schedule. A state_dict alone would silently reset that. Collected here: buffers that
    are not in the state_dict, plain int/float/bool attributes, and small tensors kept as attributes.
    """
    saved = set(model.state_dict())
    out = {}
    for name, buf in model.named_buffers():
        if name not in saved and buf is not None:
            out[f"{RUNTIME}buffer.{name}"] = buf.detach().cpu().numpy()
    for mname, module in model.named_modules():
        prefix = mname + "." if mname else ""
        for attr, value in vars(module).items():
            if attr in _MODULE_INTERNALS:
                continue
            if type(value) in (int, float, bool):
                out[f"{RUNTIME}attr.{prefix}{attr}"] = np.array(value)
            elif torch.is_tensor(value) and value.numel() <= 4096 and value.dtype != torch.bfloat16:
                out[f"{RUNTIME}tensor.{prefix}{attr}"] = value.detach().cpu().numpy()
    return out


def restore_runtime_state(model, saved):
    """Put back what runtime_state() collected. `saved` maps the same keys to arrays."""
    for key, value in saved.items():
        if not key.startswith(RUNTIME):
            continue
        kind, _, rest = key[len(RUNTIME):].partition(".")
        mname, _, attr = rest.rpartition(".")
        module = model.get_submodule(mname)
        if kind == "buffer":
            module._buffers[attr] = torch.from_numpy(np.array(value)).to(module._buffers[attr].dtype)
        elif kind == "attr":
            object.__setattr__(module, attr, np.array(value).item())
        elif kind == "tensor":
            object.__setattr__(module, attr, torch.from_numpy(np.array(value)))


@torch.no_grad()
def fresh_logits(model, x):
    """Forward on a throw-away copy in eval mode, so counters or buffers a program keeps cannot leak between calls."""
    return bf16(safe_copy(model).eval(), x).float()


def audit(model, windows):
    g = torch.Generator().manual_seed(20261110)
    x = windows[:2, :CONTEXT].clone()
    base = fresh_logits(model, x)
    worst_cut, at = 0.0, None
    for c in CUTS:
        y = x.clone()
        y[:, c:] = (y[:, c:] + torch.randint(1, 256, y[:, c:].shape, generator=g)) % 256
        d = float((fresh_logits(model, y)[:, :c] - base[:, :c]).abs().max())
        if d > worst_cut:
            worst_cut, at = d, c
    other = windows[2:3, :CONTEXT].clone()
    solo = fresh_logits(model, x[:1])
    pair = fresh_logits(model, torch.cat((x[:1], other)))[:1]
    batch_diff = float((solo - pair).abs().max())
    return {"causality_max_abs_diff": worst_cut, "causality_worst_cut": at, "cuts": list(CUTS),
            "batch_independence_max_abs_diff": batch_diff, "tolerance": AUDIT_TOLERANCE,
            "causal": worst_cut <= AUDIT_TOLERANCE, "batch_independent": batch_diff <= AUDIT_TOLERANCE,
            "finite": bool(torch.isfinite(base).all())}


@torch.no_grad()
def evaluate(model, heldout):
    m = safe_copy(model).eval()
    out = {}
    for name, z in heldout.items():
        losses = []
        for a in z.split(BATCH):
            logits = bf16(m, a[:, :CONTEXT])[:, -128:]
            ce = F.cross_entropy(logits.double().flatten(0, 1), a[:, -128:].flatten(), reduction="none").reshape(-1, 128)
            losses.extend(ce.mean(1).tolist())
        out[name] = {"ce_nats": float(np.mean(losses)), "window_ce_nats": [round(v, 6) for v in losses]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gallery", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=float, default=120.0)
    p.add_argument("--marks", default="60")
    p.add_argument("--save-weights", action="store_true")
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (int(a.budget * 4 + 600), int(a.budget * 4 + 660)))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{'T96' if a.arm == 'transformer' else 'g' + str(a.arm)}_s{a.seed}"
    data = pathlib.Path(a.data)
    train = torch.from_numpy(np.load(data / "train_u8.npy").astype(np.int64))
    z = np.load(data / "evaluation.npz")
    heldout = {"wikitext": torch.from_numpy(z["wiki"].astype(np.int64)), "fineweb_edu": torch.from_numpy(z["fineweb"].astype(np.int64))}
    gallery = None if a.arm == "transformer" else models.load_gallery(a.gallery, a.audit)
    if gallery is not None and int(a.arm) not in gallery:
        raise SystemExit(f"record {a.arm} did not pass the static audit; refusing to run it")
    wall0 = time.perf_counter()
    model, info = models.build(a.arm, a.seed, gallery)
    result = dict(info, seed=a.seed, tag=tag, budget_cpu_seconds=a.budget, batch=BATCH, context=CONTEXT,
                  optimizer={"name": "AdamW", "lr": LR, "betas": [0.9, 0.999], "weight_decay": 0.01, "clip": 1.0,
                             "schedule": f"linear warm-up over {WARMUP} updates, then constant"},
                  precision="BF16 CPU autocast, FP32 master weights",
                  source_sha256=None if gallery is None else hashlib.sha256(gallery[int(a.arm)]["program"].encode()).hexdigest(),
                  gallery_score=None if gallery is None else gallery[int(a.arm)]["score"])
    result["audit_before"] = audit(model, heldout["wikitext"])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    starts = torch.randint(len(train) - CONTEXT - 1, (20000, BATCH), generator=torch.Generator().manual_seed(65000 + a.seed))
    result["sample_starts_sha256"] = hashlib.sha256(starts.numpy().tobytes()).hexdigest()
    offsets = torch.arange(CONTEXT + 1)[None, :]
    marks = sorted(float(m) for m in a.marks.split(",") if m and float(m) < a.budget)
    history, checkpoints, cpu, update = [], [], 0.0, 0
    model.train()
    while cpu < a.budget:
        t0 = time.process_time()
        x = train[starts[update][:, None] + offsets]
        for group in opt.param_groups:
            group["lr"] = LR * min(1.0, (update + 1) / WARMUP)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(bf16(model, x[:, :-1]).float().flatten(0, 1), x[:, 1:].flatten())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        cpu += time.process_time() - t0
        update += 1
        history.append(round(float(loss.detach()), 4))
        if marks and cpu >= marks[0]:
            marks.pop(0)
            ev = evaluate(model, heldout)
            checkpoints.append({"training_cpu_seconds": cpu, "updates": update,
                                "wikitext_ce_nats": ev["wikitext"]["ce_nats"], "fineweb_edu_ce_nats": ev["fineweb_edu"]["ce_nats"]})
            model.train()
    final = evaluate(model, heldout)
    result["audit_after"] = audit(model, heldout["wikitext"])
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    digest = hashlib.sha256()
    for k in sorted(state):
        digest.update(k.encode())
        digest.update(np.ascontiguousarray(state[k]).tobytes())
    extra = runtime_state(model)
    if a.save_weights:
        np.savez(out / f"{tag}.npz", **state, **extra)
    result["runtime_state_keys"] = sorted(extra)
    result.update(
        updates=update, byte_targets=update * BATCH * CONTEXT, training_cpu_seconds=cpu,
        byte_targets_per_cpu_second=update * BATCH * CONTEXT / cpu, mid_checkpoints=checkpoints,
        wikitext_ce_nats=final["wikitext"]["ce_nats"], fineweb_edu_ce_nats=final["fineweb_edu"]["ce_nats"],
        wikitext_window_ce_nats=final["wikitext"]["window_ce_nats"], fineweb_edu_window_ce_nats=final["fineweb_edu"]["window_ce_nats"],
        train_loss_history=history, final_train_loss_mean_last_20=float(np.mean(history[-20:])),
        state_sha256=digest.hexdigest(), wall_seconds_total=time.perf_counter() - wall0,
        process_cpu_seconds_total=time.process_time(), peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        environment={"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "onednn_max_cpu_isa": os.environ.get("ONEDNN_MAX_CPU_ISA", "default"), "threads": 1,
                     "machine": platform.machine()},
        status="ok" if (result["audit_before"]["causal"] and result["audit_after"]["causal"]) else "causality_audit_failed")
    tmp = out / f"{tag}.json.tmp"
    tmp.write_text(json.dumps(result, allow_nan=False) + "\n")
    tmp.rename(out / f"{tag}.json")
    print(json.dumps({k: result[k] for k in ("tag", "name", "parameters", "updates", "training_cpu_seconds",
                                             "wikitext_ce_nats", "fineweb_edu_ce_nats", "status", "peak_rss_mib")}))


if __name__ == "__main__":
    main()
