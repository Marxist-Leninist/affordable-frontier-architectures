"""export_lctx.py PACK [--run RUN] -- write lctx.json + lctx.params.f32 next to a pack, from the gate the pack's own
checkpoint was trained with.

The pack carries its own state.json (p128_export copies RUN/state.json into it); its 'lctx' record (written by the
trainer mod) gives the update count, geometry and, from v1.1 on, the sha256 of the exact parameters.  The gate itself
is found in RUN/lctx.pinned (pinned automatically at every PAUSE/STOP/HALT boundary, i.e. for every exported step),
RUN/lctx.ring or RUN/lctx.state.pt, and accepted only when updates, geometry and sha256 all match.  It can run any time
after the export (the pins are not pruned by training).  No GPU; reads ~20 MB per candidate.
Exit: 0 exported or NO_LCTX (the pack predates the mod); 2 refused."""
import hashlib
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from lctx_serve import FORM, LctxServe  # noqa: E402


def _rw(sd):
    """(r, w) of a v2 copy/state, or of a v1 one (rho, w, c with w = c = 0 -> r = softplus^-1(exp(rho)))"""
    if 'r' in sd:
        return sd['r'].float(), sd['w'].float()
    if bool(sd['w'].abs().gt(0).any()) or bool(sd['c'].abs().gt(0).any()):
        raise ValueError('a v1 gate with nonzero w/c cannot be expressed in the v2 form')
    lam = sd['rho'].double().exp()
    r = torch.where(lam > 30.0, lam, torch.log(torch.expm1(lam.clamp_min(1e-300))))
    return r.float(), torch.zeros(sd['rho'].shape + (int(sd['E']),))


def main():
    args = sys.argv[1:]
    run = None
    if '--run' in args:
        i = args.index('--run')
        run = args[i + 1]
        del args[i:i + 2]
    pack = args[0]
    with open(os.path.join(pack, 'state.json')) as f:
        st = json.load(f)
    step = int(st.get('step', -1))
    rec = st.get('lctx')
    if not isinstance(rec, dict):
        print('NO_LCTX: the pack\'s checkpoint (step %d) has no lctx record; nothing to export' % step)
        return 0
    want = int(rec['updates'])
    run = run or os.path.dirname(rec.get('state_file') or '')
    cands = []
    for sub in ('lctx.pinned', 'lctx.ring'):
        d = os.path.join(run, sub)
        if os.path.isdir(d):
            cands += [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.endswith('_u%07d.pt' % want)]
    cands.append(os.path.join(run, 'lctx.state.pt'))
    for path in cands:
        if not os.path.exists(path):
            continue
        try:
            sd = torch.load(path, map_location='cpu')
        except Exception:
            continue
        if int(sd.get('updates', -1)) != want or [int(sd['L']), int(sd['H']), int(sd['E'])] != list(rec['geometry']):
            continue
        if sd.get('trainer_step') is not None and int(sd['trainer_step']) > step + 1:
            continue  # a copy written after the exported step (cur_step semantics allow step + 1 at that boundary)
        try:
            r, w = _rw(sd)
        except ValueError:
            continue
        sha = hashlib.sha256(LctxServe.blob(r, w)).hexdigest()
        if rec.get('sha256') and sha != rec['sha256']:
            continue
        amax = float(rec.get('amax') or sd.get('amax') or 6.0)
        meta = LctxServe.write(pack, r, w, amax, dict(glue=rec.get('version'), trainer_step=step, updates=want,
                                                      source=path, installed_step=rec.get('installed_step'),
                                                      record_sha256=rec.get('sha256')))
        back = LctxServe.from_pack(pack, expect=rec)
        span = 1.0 / F.softplus(back.r.double())
        print('EXPORTED', json.dumps({k: meta[k] for k in ('updates', 'trainer_step', 'L', 'H', 'E', 'sha256')}),
              'from', path, 'span median %.4g min %.4g' % (float(span.median()), float(span.min())))
        return 0
    print('REFUSED: no saved gate in %s has updates == %d%s for step %d' % (
        run, want, ' and sha256 %s' % rec['sha256'][:12] if rec.get('sha256') else '', step))
    return 2


if __name__ == '__main__':
    sys.exit(main())
