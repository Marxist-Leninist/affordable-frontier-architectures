"""build_hotmod.py -- hotmod = header + src/lctx_core.py (verbatim) + src/lctx_hotmod_glue.py.  Prints the sha256.
python build_hotmod.py [OUT_DIR]"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = 'z' * 31 + '_lctx.persist.py'

HEADER = '''# %s -- LEARNED CONTEXT LENGTH (lctx v1)
# claude-code-lctx888-20261116 for the owner, 16 Nov 2026: "idea looking it apply to 88B a learned context length that
# model decides in training and on inference".  Source, tests, toy results, README: laptop C:\\Users\\user\\agillm888_lctx_20261116,
# GETH /root/claude_lctx_20261116 (this file is built by tools/build_hotmod.py from src/lctx_core.py + src/lctx_hotmod_glue.py).
#
# WHAT: every causal attention call of every layer gets a learned forget gate (Forgetting Transformer, Lin et al. 2025):
#   a key fades by exp(-a_t) for every newer token t, a_t = min(exp(rho_h) * 2 sigmoid(w_h . unit(v_t) + c_h), amax).
#   exp(-rho_h) is head h's context length in tokens (learned, log scale); w_h, c_h let each token change it at inference
#   (data-dependent: a token can make the head forget what came before).  Exact, through one extra head dimension
#   (q' = [q, 1/s, 0..], k' = [k, running sum of a, 0..], hd 128 -> 136): no T x T mask, the same SDPA kernel family.
#   Install rate exp(rho) = 1e-6: the oldest of 2048 keys loses <= 0.002 nats, i.e. the model as it was.  It learns from
#   the training loss only (span_cost knob 0 = no price on context); w, c stay 0 until a head's rate passes 1e-3.
#   Bidirectional passes (NAT, local denoise: _MM_CAUSAL False) are NOT changed (v1).  No weight, latent, table,
#   optimizer state, checkpoint tensor or trainer function is changed; the trainer's _sdpa and _rs_status globals are
#   wrapped (the originals are called for everything this mod does not handle).
# MEMORY: BiasedAttn saves only q, k, v and the bias (SDPA itself would also save its output) and redoes each head
#   group's forward inside its backward: about +20 MB of transients on the card, one extra attention forward per backward.
# GRADIENTS: rho, w, c get the exact gradient of the formula (KeyBias.backward, analytic; no gradient into v from the
#   gate), summed per layer on the host; one Adam step per training backward, in a queue_callback at the end of that
#   backward.  No forward and no no_grad pass (deveval, evals, probes) changes anything.
# STATE: <run>/lctx.state.pt (rho, w, c, Adam moments, trainer step) after every update; <run>/lctx.ring/ (3 older
#   parameter copies, for a launch from an older checkpoint); <run>/lctx.rho.f16 (every head's rate after every update);
#   <run>/lctx.status.json, lctx.jsonl, lctx.spans.json; EXTRA_STATE['lctx'] (from training backwards only).
# KNOBS: <run>/hotmods/lctx.knobs.json, re-read at step boundaries: lr_rho 0.03, lr_w 0.003, span_cost 0, amax 6,
#   ddep_from 1e-3, rho_min log 1e-7, rho_max log 4, wmax 8, wdecay 1e-3, b1 0.9, b2 0.99, freeze false.
# KILL: touch <run>/hotmods/lctx.off  -> the original attention from the next step on (a step's forward and its
#   checkpoint recompute must agree, so the switch is read at boundaries).  3 steps in a row with a failing layer switch
#   it off for the process.  Crash marker <run>/lctx.inflight exists only while a training step runs with the mod: a
#   launch that finds it does not install (delete it to allow the mod again).
# NOT DONE HERE: serving.  The pack exporter and serve128 do not apply the gate yet; a pack cut after the spans have
#   moved must be served with lctx.state.pt (src/lctx_serve.py) or the served model differs from the trained one.
''' % NAME


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'hotmod')
    core = open(os.path.join(ROOT, 'src', 'lctx_core.py'), encoding='utf-8').read().replace('\r\n', '\n')
    glue = open(os.path.join(ROOT, 'src', 'lctx_hotmod_glue.py'), encoding='utf-8').read().replace('\r\n', '\n')
    src = HEADER + '\n' + core.rstrip('\n') + '\n\n\n' + glue
    compile(src, NAME, 'exec')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, NAME)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(src)
    print(hashlib.sha256(src.encode('utf-8')).hexdigest(), len(src.encode('utf-8')), path)


if __name__ == '__main__':
    main()
