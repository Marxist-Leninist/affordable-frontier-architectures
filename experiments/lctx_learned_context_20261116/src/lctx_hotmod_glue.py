# ==================================================================================== trainer glue (lctx hotmod v1.1)
# Everything above this line is lctx_core.py verbatim (tested by tests/test_lctx_core.py); below: the trainer side.
import hashlib as _lh
import json as _lj
import os as _lo
import sys as _lsys
import time as _lt

_H = HOTMOD  # noqa: F821  (the hotmod loader's namespace)
_T = _H.get('trainer') if isinstance(_H.get('trainer'), dict) else {}
_RUN = _H['run']
VG = 'lctx-v1.1-20261116'
HCHUNK, HGROUP = 8, 8      # heads per chunk (gate) and per SDPA group: bounds the device transients to ~20 MB
WARM = 2                   # training backwards after an install whose gradients are dropped (observe only)
FAILS = 3                  # failing steps in a row before the mod switches itself off for the process
RING = 4                   # newest parameter copies kept (by write order) besides the pinned ones
PINS = 48                  # parameter copies pinned at PAUSE / STOP / HALT boundaries (exports, backups), by age
ID_CHECKS = 4              # first forward calls compared with the original attention
HIST_MAX = 32 << 20

try:
    from torch.nn.attention import sdpa_kernel as _lc_sk, SDPBackend as _lc_be
except Exception:  # older torch: no forcing (the self-test still has to pass)
    _lc_sk = _lc_be = None


def _lc_say(msg):
    print('[lctx] ' + msg, flush=True)


def _lc_atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        _lj.dump(obj, f)
    _lo.replace(tmp, path)


def _lc_num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _lc_step():
    try:
        return int(getattr(_T.get('_RS'), 'cur_step'))
    except Exception:
        return None


def _lc_scale(E):
    """the scale the trainer's _sdpa uses: T / sqrt(hd) when TERNTF_ATTN_T is in effect, SDPA's default otherwise"""
    if _T.get('ATTN_T') is not None:
        t = float(_T['_attn_t']())
        if t != 1.0:
            return t / math.sqrt(E)
    return 1.0 / math.sqrt(E)


def _lc_kernel(q, k, v, is_causal=True, scale=None):
    """SDPA with the memory-efficient kernel forced on CUDA (never a silent fallback to the T x T math kernel)"""
    if q.is_cuda and _lc_sk is not None:
        with _lc_sk([_lc_be.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)


def _lc_in_backward():
    return torch._C._current_graph_task_id() != -1


# ------------------------------------------------------------------------------------------------ persistence

def _lc_paths():
    return dict(state=_lo.path.join(_RUN, 'lctx.state.pt'), ring=_lo.path.join(_RUN, 'lctx.ring'),
                pins=_lo.path.join(_RUN, 'lctx.pinned'), status=_lo.path.join(_RUN, 'lctx.status.json'),
                log=_lo.path.join(_RUN, 'lctx.jsonl'), hist=_lo.path.join(_RUN, 'lctx.r.f16'),
                spans=_lo.path.join(_RUN, 'lctx.spans.json'), inflight=_lo.path.join(_RUN, 'lctx.inflight'),
                off=_lo.path.join(_RUN, 'hotmods', 'lctx.off'),
                knobs=_lo.path.join(_RUN, 'hotmods', 'lctx.knobs.json'), statejson=_lo.path.join(_RUN, 'state.json'))


def _lc_copy_name(step, updates):
    return 's%09d_u%07d.pt' % (int(step), int(updates))


def _lc_parse_copy(name):
    """(step, updates) of a ring / pin file name, or None"""
    try:
        if name.startswith('s') and name.endswith('.pt') and '_u' in name:
            s, u = name[1:-3].split('_u')
            return int(s), int(u)
    except Exception:
        pass
    return None


def _lc_save_copy(S, folder, step, keep, by_age=False):
    """parameter copy {r, w, updates, trainer_step, amax, form, sha256}; keeps the newest `keep` by write order"""
    st = S['st']
    _lo.makedirs(folder, exist_ok=True)
    blob = st.params_bytes()
    tmp = _lo.path.join(folder, 'tmp_copy.pt')  # torch.save refuses a bare '.tmp' name
    torch.save(dict(version=V, form=FORM, r=st.r.clone(), w=st.w.clone(), trainer_step=int(step), updates=st.updates,
                    L=st.L, H=st.H, E=st.E, amax=S['knobs']['amax'], sha256=_lh.sha256(blob).hexdigest()), tmp)
    _lo.replace(tmp, _lo.path.join(folder, _lc_copy_name(step, st.updates)))
    names = [n for n in _lo.listdir(folder) if _lc_parse_copy(n)]
    names.sort(key=lambda n: _lo.path.getmtime(_lo.path.join(folder, n)))
    for old in names[:-keep]:
        try:
            _lo.remove(_lo.path.join(folder, old))
        except FileNotFoundError:
            pass


def _lc_record(S, stats=None):
    """EXTRA_STATE['lctx']: what the checkpoint was trained with (read by export_lctx.py and the serving check)"""
    st = S['st']
    sm = (stats or {}).get('summary') or st.summary()
    blob = st.params_bytes()
    return dict(version=VG, form=FORM, installed=bool(S['installed']), active=bool(S['active'] and not S['disabled']),
                why=None if S['why'] is None else str(S['why'])[:300], installed_step=int(S['installed_step']),
                updates=int(st.updates), geometry=[int(st.L), int(st.H), int(st.E)], amax=float(S['knobs']['amax']),
                trainer_step=_lc_step(), sha256=_lh.sha256(blob).hexdigest(), span_min=_lc_num(sm['span_min']),
                span_p10=_lc_num(sm['span_p10']), span_median=_lc_num(sm['span_median']),
                heads_under_2048=int(sm['heads_under_2048']), state_file=S['p']['state'],
                at_utc=_lt.strftime('%Y-%m-%dT%H:%M:%SZ', _lt.gmtime()))


def _lc_publish(S, stats=None):
    es = _H.get('extra_state')
    if not isinstance(es, dict):
        return
    ent = _lc_record(S, stats)
    if _lj.loads(_lj.dumps(ent)) == ent:  # state.json must survive a JSON round trip exactly
        es['lctx'] = ent
        S['extra_dirty'] = False
    else:
        S['persist_errors'] += 1
        S['last_error'] = 'extra_state entry does not survive a JSON round trip; not published'


def _lc_stats(S):
    st = S['st']
    return dict(version=VG, core=V, form=FORM, at_utc=_lt.strftime('%Y-%m-%dT%H:%M:%SZ', _lt.gmtime()),
                trainer_step=_lc_step(), installed_step=S['installed_step'], geometry=[st.L, st.H, st.E],
                installed=S['installed'], active=S['active'], disabled=S['disabled'], why=S['why'],
                post_crash=S['post_crash'], calls=S['calls'], bidir_calls=S['bidir_calls'],
                skipped_calls=S['skipped_calls'], fallback_calls=S['fallback_calls'], grad_passes=S['grad_passes'],
                held=S['held'], updates=st.updates, layers_updated=S['layers_updated'], bad_layers=S['bad_layers'],
                warm_left=S['warm_left'], fails=S['fails'], fail_streak=S['fail_streak'],
                recompute_errors=S['recompute_errors'], gate_errors=LctxAttn.gate_errors,
                attn_retries=LctxAttn.retries, update_errors=S['update_errors'], persist_errors=S['persist_errors'],
                last_error=S['last_error'], gate_last_error=LctxAttn.last_error, pins=S['pins'],
                identity_max_abs_diff=S['id_diffs'], knobs=S['knobs'], summary=st.summary(),
                off_file_present=_lo.path.exists(S['p']['off']), state_file=S['p']['state'], state_source=S['source'])


def _lc_persist(S, save_state=False, extra=False):
    p = S['p']
    st = S['st']
    try:
        if save_state:
            if S.get('keep_prev') and _lo.path.exists(p['state']):  # never overwrite a state this launch did not load
                try:
                    _lo.replace(p['state'], p['state'] + '.prev-%s' % S['keep_prev'])
                except Exception as ex:
                    S['persist_errors'] += 1
                    S['last_error'] = 'keep_prev: ' + repr(ex)[:200]
                S['keep_prev'] = None
            sd = st.state_dict()
            sd['m'] = {k: t.float() for k, t in sd['m'].items()}
            sd['v'] = {k: t.float() for k, t in sd['v'].items()}
            sd.update(trainer_step=_lc_step(), saved_at=_lt.time(), glue=VG, amax=S['knobs']['amax'])
            tmp = p['state'] + '.tmp'
            torch.save(sd, tmp)
            _lo.replace(tmp, p['state'])
            try:
                _lc_save_copy(S, p['ring'], _lc_step() or 0, RING)
            except Exception as ex:
                S['persist_errors'] += 1
                S['last_error'] = 'ring: ' + repr(ex)[:200]
            try:  # the rate parameter of every head after every update (float16 rows of L*H), rotated at HIST_MAX
                if _lo.path.exists(p['hist']) and _lo.path.getsize(p['hist']) > HIST_MAX:
                    _lo.replace(p['hist'], p['hist'] + '.1')
                with open(p['hist'], 'ab') as f:
                    f.write(st.r.to(torch.float16).numpy().tobytes())
            except Exception as ex:
                S['persist_errors'] += 1
                S['last_error'] = 'history: ' + repr(ex)[:200]
            if st.updates <= 3 or st.updates % 20 == 0:
                try:
                    _lc_atomic_json(p['spans'], dict(trainer_step=_lc_step(), updates=st.updates,
                                                     layer_median_span=[_lc_num(x) for x in (1.0 / F.softplus(
                                                         st.r.double())).median(dim=1).values],
                                                     r=[[round(float(x), 3) for x in row] for row in st.r]))
                except Exception as ex:
                    S['persist_errors'] += 1
                    S['last_error'] = 'spans: ' + repr(ex)[:200]
        stats = _lc_stats(S)
        _lc_atomic_json(p['status'], stats)
        if save_state:
            with open(p['log'], 'a') as f:
                f.write(_lj.dumps({k: stats[k] for k in ('at_utc', 'trainer_step', 'updates', 'layers_updated',
                                                         'bad_layers', 'summary', 'calls', 'bidir_calls',
                                                         'fallback_calls', 'gate_errors', 'attn_retries')}) + '\n')
        if extra or S.get('extra_dirty'):
            _lc_publish(S, stats)
        return stats
    except Exception as ex:
        S['persist_errors'] += 1
        S['last_error'] = 'persist: ' + repr(ex)[:200]
        return None


def _lc_read_knobs(S):
    p = S['p']['knobs']
    try:
        mt = _lo.path.getmtime(p)
    except OSError:
        mt = None
    if mt == S['knobs_mtime'] and S['knobs'] is not None:
        return
    S['knobs_mtime'] = mt
    d = None
    if mt is not None:
        try:
            with open(p) as f:
                d = _lj.load(f)
        except Exception as ex:
            _lc_say('knobs file unreadable, defaults kept: ' + repr(ex)[:160])
    k = clean_knobs(d)
    if k != S['knobs']:
        S['knobs'] = k
        _lc_say('knobs ' + _lj.dumps(k))


def _lc_clear_inflight(S):
    if S['inflight_written']:
        try:
            _lo.remove(S['p']['inflight'])
        except FileNotFoundError:
            pass
        except Exception as ex:
            S['persist_errors'] += 1
            S['last_error'] = 'inflight: ' + repr(ex)[:200]
        S['inflight_written'] = False


def _lc_mark_inflight(S):
    try:
        _lc_atomic_json(S['p']['inflight'], dict(version=VG, step=_lc_step(), pid=_lo.getpid(), at=_lt.time(),
                                                 post_crash=bool(S['post_crash'])))
    except Exception as ex:
        S['persist_errors'] += 1
        S['last_error'] = 'inflight: ' + repr(ex)[:200]
    S['inflight_written'] = True


def _lc_fail(S, why):
    S['disabled'], S['why'] = True, why
    _lc_say('SWITCHED OFF for this process from the next step on (the original attention again): ' + why)
    S['extra_dirty'] = True
    _lc_persist(S)


def _lc_trip(S, why):
    S['fails'] += 1
    S['step_failed'] = True
    S['last_error'] = why
    if S['fails'] <= 5:
        _lc_say('this layer uses the original attention for the rest of the step: ' + why)


def _lc_pin(S, step, why):
    """the exact gate of a step that is being paused / stopped / halted (exports and backups capture that step)"""
    try:
        _lc_save_copy(S, S['p']['pins'], step, PINS)
        S['pins'] += 1
        _lc_say(f'pinned the gate of step {step} ({why}): {S["p"]["pins"]}/{_lc_copy_name(step, S["st"].updates)}')
    except Exception as ex:
        S['persist_errors'] += 1
        S['last_error'] = 'pin: ' + repr(ex)[:200]


# ------------------------------------------------------------------------------------------------ the hooks

def _lc_attend(S, q, k, v, layer, small=False):
    r, w = S['st'].layer_params(layer, q.device)
    sink = S['sinks'][layer] if torch.is_grad_enabled() else None
    return lctx_attention(q, k, v, r, w, _lc_scale(q.shape[-1]), amax=S['knobs']['amax'], sink=sink,
                          hchunk=1 if small else HCHUNK, hgroup=1 if small else HGROUP, sdpa=_lc_kernel)


def _lc_layer(S, depth=3):
    """the block index of the caller (Block.forward / forward_with_aux has `self`): a function frame's f_locals is a
    snapshot dict cached on the frame (CPython 3.11/3.12), so it is emptied right away to keep no tensor alive longer"""
    f = _lsys._getframe(2)
    for _ in range(depth):
        if f is None:
            return None
        if f.f_code.co_flags & 0x1:  # CO_OPTIMIZED: a function frame (never a module or class namespace)
            d = f.f_locals
            s = d.get('self')
            if type(d) is dict:
                try:
                    d.clear()
                except Exception:
                    pass
            if s is not None:
                i = S['idx'].get(id(s))
                if i is not None:
                    return i
        f = f.f_back
    return None


def _lc_make_sdpa(prev):
    def _lc_sdpa(q, k, v, _prev=prev):
        """the trainer's _sdpa with the learned forgetting on causal passes.  Bidirectional passes, unknown callers and
        anything unexpected go to the original.  LctxAttn saves nothing when its forward fails, so the forward's choice
        (gate, or the original after a failure) is the recompute's choice too."""
        S = _LC  # noqa: F821
        if not S['active'] or S.get('retired'):
            return _prev(q, k, v)
        in_bwd = _lc_in_backward()
        try:
            if not _T.get('_MM_CAUSAL', True):
                if not in_bwd:
                    S['bidir_calls'] += 1
                return _prev(q, k, v)
            layer = _lc_layer(S)
            ok = (layer is not None and q.dim() == 4 and q.dtype == torch.float32 and k.dtype == torch.float32
                  and v.dtype == torch.float32 and q.shape[1] == S['st'].H and q.shape[-1] == S['st'].E
                  and tuple(k.shape) == tuple(q.shape) and tuple(v.shape[:3]) == tuple(q.shape[:3])
                  and _T.get('ATTN_DTYPE', 'fp32') == 'fp32')
        except Exception:
            layer, ok = None, False
        if not ok:
            if not in_bwd:
                S['skipped_calls'] += 1
            return _prev(q, k, v)
        if layer in S['fallback']:
            if not in_bwd:
                S['fallback_calls'] += 1
            return _prev(q, k, v)
        if in_bwd:
            failed = False
            try:
                return _lc_attend(S, q, k, v, layer)
            except Exception as ex:
                failed = True
                S['recompute_errors'] += 1
                S['last_error'] = f'layer {layer} recompute: ' + repr(ex)[:200]
            if failed:  # outside the except: the failed attempt's frames and memory are released first
                if q.is_cuda:
                    torch.cuda.empty_cache()
                return _lc_attend(S, q, k, v, layer, small=True)
        try:
            if torch.is_grad_enabled() and not S['inflight_written']:
                _lc_mark_inflight(S)
            out = _lc_attend(S, q, k, v, layer)
        except Exception as ex:
            S['fallback'].add(layer)
            _lc_trip(S, f'layer {layer} forward: ' + repr(ex)[:200])
            return _prev(q, k, v)
        S['calls'] += 1
        if S['id_left'] > 0:
            S['id_left'] -= 1
            try:
                with torch.no_grad():
                    ref = _prev(q.detach(), k.detach(), v.detach())
                    S['id_diffs'].append([layer, _lc_num((out.detach() - ref).abs().max()), _lc_num(ref.abs().max())])
                del ref
            except Exception as ex:
                S['last_error'] = 'identity check: ' + repr(ex)[:200]
        return out
    _lc_sdpa._lctx = True
    return _lc_sdpa


def _lc_make_sink(S, layer):
    def sink(dr, dw):
        S['st'].add(layer, dr, dw)
        gid = torch._C._current_graph_task_id()
        if S['cb_gid'] != gid:
            S['cb_gid'] = gid
            torch.autograd.Variable._execution_engine.queue_callback(_lc_after_backward)
    return sink


def _lc_after_backward():
    """end of a backward that reached the gates: one optimiser step for every layer that got a gradient"""
    S = _LC  # noqa: F821
    S['cb_gid'] = None
    try:
        S['grad_passes'] += 1
        S['post_crash'] = False  # a clean training backward: the next marker is a first strike again
        if S['step_failed']:
            S['st'].drop()  # a step with a fallback layer: its gate gradients are incomplete
            return
        if S['warm_left'] > 0:
            S['warm_left'] -= 1
            S['held'] += 1
            S['st'].drop()
            _lc_persist(S, extra=True)  # a training backward: the record may be (re)published
            return
        try:
            inv = float(getattr(_T.get('TO'), 'inv_scale'))
            inv = inv if 0.0 < inv < float('inf') else 1.0
        except Exception:
            inv = 1.0
        res = S['st'].step(S['knobs'], gscale=inv)
        S['layers_updated'] += res['layers']
        S['bad_layers'] += res['bad']
        if res['layers']:
            stats = _lc_persist(S, save_state=True, extra=True)
            u = S['st'].updates
            if stats is not None and (u <= 3 or u % 20 == 0):
                sm = stats['summary']
                _lc_say(f"update {u} (step {stats['trainer_step']}): {res['layers']} layers, span median "
                        f"{sm['span_median']:.3g} p10 {sm['span_p10']:.3g} min {sm['span_min']:.3g} tokens, "
                        f"heads under 2048: {sm['heads_under_2048']}/{sm['heads']}; |w| max {sm['w_absmax']:.3g}; "
                        f"bad {res['bad']}; errors {S['fails']}+{S['recompute_errors']}+{LctxAttn.gate_errors}+"
                        f"{S['update_errors']}")
    except Exception as ex:
        S['update_errors'] += 1
        S['last_error'] = 'update: ' + repr(ex)[:200]
    finally:
        _lc_clear_inflight(S)


def _lc_boundary(S, step):
    """step boundary: the only place where the mod's on/off state, knobs and per-step fallbacks change; pins the gate
    of a step that is about to be paused, stopped or halted"""
    if S['step_failed']:
        S['fail_streak'] += 1
        if S['fail_streak'] >= FAILS and not S['disabled']:
            _lc_fail(S, f'{FAILS} steps in a row with a failing layer, last: {S["last_error"]}')
    elif S['calls']:
        S['fail_streak'] = 0
    S['step_failed'] = False
    S['fallback'] = set()
    _lc_read_knobs(S)
    want = not S['disabled'] and not _lo.path.exists(S['p']['off'])
    if want != S['active']:
        S['active'] = want
        S['extra_dirty'] = True  # the record changes with the next training backward (or now, if it stays off)
        _lc_say('ON (learned context length in causal attention)' if want else
                'OFF (the original attention; ' + ('disabled: ' + str(S['why']) if S['disabled'] else 'lctx.off present') + ')')
        if not want:
            _lc_publish(S)
    pend = getattr(_T.get('_RS'), 'pending', None)
    if pend in ('pause', 'stop', 'halt') and type(step) is int:
        _lc_pin(S, step, pend)
    _lc_clear_inflight(S)
    _lc_persist(S)


def _lc_make_status(prev):
    def _lc_rs_status(*a, _prev=prev, **kw):  # kwarg named _prev, like the other _rs_status hooks (chain walkers)
        out = _prev(*a, **kw)
        S = _LC  # noqa: F821
        if S.get('retired'):
            return out
        try:
            _lc_boundary(S, a[1] if len(a) > 1 else kw.get('step'))
        except Exception as ex:
            S['last_error'] = 'boundary: ' + repr(ex)[:200]
        return out
    _lc_rs_status._lctx = True
    return _lc_rs_status


# ------------------------------------------------------------------------------------------------ install

def _lc_selftest(dev):
    """the core on dev against the explicit reference (forward, gradients through the sink, the kernel the mod uses)
    and, on the CPU, under non-reentrant checkpointing (double unpack reproduced; a failure inside the forward saves
    nothing); None or what failed"""
    try:
        from torch.utils.checkpoint import checkpoint as _ck, CheckpointError as _CE
        g = torch.Generator().manual_seed(11)
        dt = torch.float64 if dev.type == 'cpu' else torch.float32
        tol = 1e-9 if dt == torch.float64 else 2e-4
        B, Hh, Tt, E = 1, 3, 40, 128 if dev.type != 'cpu' else 16

        def rn(*s, sc=1.0):
            return (torch.randn(*s, generator=g, dtype=torch.float64) * sc).to(device=dev, dtype=dt)

        q = F.normalize(rn(B, Hh, Tt, E), dim=-1) * math.sqrt(E)
        k, v = F.normalize(rn(B, Hh, Tt, E), dim=-1), rn(B, Hh, Tt, E)
        r = r_from_rate(torch.tensor([0.3, 0.02, 1e-6])).to(device=dev, dtype=dt)
        w = rn(Hh, E, sc=0.3)
        got = {}

        def sink(dr, dw_):
            got['g'] = (dr, dw_)

        qq, kk, vv = (t.clone().requires_grad_() for t in (q, k, v))
        o = lctx_attention(qq, kk, vv, r, w, 1.0, amax=6.0, sink=sink, hchunk=2, hgroup=2, sdpa=_lc_kernel)
        wt = rn(B, Hh, Tt, E)
        (o * wt).sum().backward()
        q2, k2, v2, r2, w2 = (t.clone().requires_grad_() for t in (q, k, v, r, w))
        ref = ref_attention(q2, k2, v2, r2, w2, 1.0, amax=6.0)
        (ref * wt).sum().backward()
        if float((o.detach() - ref.detach()).abs().max()) > tol:
            return f'forward differs from the reference on {dev}'
        for name, a, b in (('q', qq.grad, q2.grad), ('k', kk.grad, k2.grad), ('v', vv.grad, v2.grad),
                           ('r', got['g'][0], r2.grad), ('w', got['g'][1], w2.grad)):
            if a is None or float((a - b).abs().max()) > tol * max(1.0, float(b.abs().max())):
                return f'gradient of {name} differs from the reference on {dev}'
        if dev.type == 'cpu':
            Wm = rn(E * Hh, E * Hh, sc=1.0 / math.sqrt(E * Hh))

            class BadAttn(LctxAttn):
                @staticmethod
                def backward(ctx, dout):
                    ctx.saved_tensors
                    return LctxAttn.backward(ctx, dout)

            state = {'n': 0, 'fail': set(), 'fb': False}

            def flaky(*a, **kw):
                state['n'] += 1
                if state['n'] in state['fail']:
                    raise RuntimeError('selftest: simulated failure')
                return F.scaled_dot_product_attention(*a, **kw)

            def blk(x, mode='good'):
                h = (x @ Wm).view(B, Tt, Hh, E).transpose(1, 2)
                qn, kn = F.normalize(h, dim=-1) * math.sqrt(E), F.normalize(h, dim=-1)
                if mode == 'bad':
                    a = BadAttn.apply(qn, kn, h, r, w, 6.0, 1.0, lambda *a_: None, 2, 2, None)
                elif mode == 'glue':
                    if state['fb']:
                        a = F.scaled_dot_product_attention(qn, kn, h, is_causal=True, scale=1.0)
                    elif torch._C._current_graph_task_id() == -1:
                        try:
                            a = lctx_attention(qn, kn, h, r, w, 1.0, sink=lambda *a_: None, hgroup=2, sdpa=flaky)
                        except Exception:
                            state['fb'] = True
                            a = F.scaled_dot_product_attention(qn, kn, h, is_causal=True, scale=1.0)
                    else:
                        a = lctx_attention(qn, kn, h, r, w, 1.0, sink=lambda *a_: None, hgroup=2, sdpa=flaky)
                else:
                    a = lctx_attention(qn, kn, h, r, w, 1.0, sink=lambda *a_: None, hchunk=2, hgroup=2)
                return a.transpose(1, 2).reshape(B, Tt, -1)

            x0 = rn(B, Tt, E * Hh)
            xa, xb = x0.clone().requires_grad_(), x0.clone().requires_grad_()
            (blk(xa) ** 2).sum().backward()
            (_ck(blk, xb, use_reentrant=False) ** 2).sum().backward()
            if float((xa.grad - xb.grad).abs().max()) > 1e-9:
                return 'checkpointed gradient differs'
            try:
                xc = x0.clone().requires_grad_()
                (_ck(lambda t: blk(t, 'bad'), xc, use_reentrant=False) ** 2).sum().backward()
                return 'the double-unpack failure was not reproduced (the checkpoint test cannot see it)'
            except _CE:
                pass
            state['fail'] = {2}  # the second SDPA group of the original forward fails: fall back for the step
            xd = x0.clone().requires_grad_()
            (_ck(lambda t: blk(t, 'glue'), xd, use_reentrant=False) ** 2).sum().backward()
            if not state['fb']:
                return 'the simulated forward failure did not fall back'
        return None
    except Exception as ex:
        return 'raised ' + repr(ex)[:300]


def _lc_list_copies(folder):
    out = []
    if _lo.path.isdir(folder):
        for n in _lo.listdir(folder):
            su = _lc_parse_copy(n)
            if su:
                out.append((su[0], su[1], _lo.path.join(folder, n)))
    return out


def _lc_load_state(L, Hh, E, now):
    """(LctxState, source text, keep_prev).  The checkpoint's own record (state.json['lctx'].updates = U) decides:
    no record -> fresh (the checkpoint predates the mod); a saved copy with updates == U -> that copy; otherwise the
    newest saved state (latents are updated in place and not rewound by a resume), with a warning.  A state file this
    launch does not continue from is renamed, never overwritten (keep_prev)."""
    p = _lc_paths()
    st = LctxState(L, Hh, E)
    rec = None
    try:
        with open(p['statejson']) as f:
            rec = _lj.load(f).get('lctx')
    except Exception:
        rec = None
    have = _lo.path.exists(p['state'])
    if not isinstance(rec, dict) or rec.get('updates') is None:
        return st, 'fresh (the checkpoint has no lctx record)', ('s%s' % now) if have else None
    want = int(rec['updates'])
    cands = [(s_, u, path) for s_, u, path in _lc_list_copies(p['pins']) + _lc_list_copies(p['ring']) if u == want]
    if have:
        try:
            sd = torch.load(p['state'], map_location='cpu')
            if int(sd.get('updates', -1)) == want:
                cands.insert(0, (sd.get('trainer_step'), want, p['state']))
        except Exception:
            sd = None
    for s_, u, path in cands:
        try:
            sd = torch.load(path, map_location='cpu')
            n = min(L, int(sd['L']))
            if (int(sd['H']), int(sd['E'])) != (Hh, E):
                continue
            if 'm' in sd:  # a full state file
                tmp = LctxState(int(sd['L']), Hh, E)
                sd2 = dict(sd)
                sd2['m'] = {kk: t.double() for kk, t in sd['m'].items()}
                sd2['v'] = {kk: t.double() for kk, t in sd['v'].items()}
                note = tmp.load_state_dict(sd2)
                st.r[:n], st.w[:n] = tmp.r[:n], tmp.w[:n]
                for name in ('r', 'w'):
                    st.m[name][:n], st.v[name][:n] = tmp.m[name][:n], tmp.v[name][:n]
                st.t[:n] = tmp.t[:n]
                st.updates = tmp.updates
            else:  # a parameter copy (ring / pin): Adam moments start again
                st.r[:n], st.w[:n] = sd['r'][:n].float(), sd['w'][:n].float()
                st.updates = int(sd['updates'])
                note = None
            extra = '' if int(sd['L']) == L else f'; geometry {sd["L"]} -> {L} layers'
            keep = None if path == p['state'] else ('s%s' % now if have else None)
            return st, f'{path} (update {want}, step {s_}){extra}' + (f'; {note}' if note else ''), keep
        except Exception:
            continue
    if have:  # no exact copy: the newest state, never a silent reset
        try:
            sd = torch.load(p['state'], map_location='cpu')
            tmp = LctxState(int(sd['L']), Hh, E)
            sd2 = dict(sd)
            sd2['m'] = {kk: t.double() for kk, t in sd['m'].items()}
            sd2['v'] = {kk: t.double() for kk, t in sd['v'].items()}
            note = tmp.load_state_dict(sd2)
            n = min(L, tmp.L)
            st.r[:n], st.w[:n] = tmp.r[:n], tmp.w[:n]
            st.updates = tmp.updates
            _lc_say(f'WARNING: no saved gate has the checkpoint\'s update count {want}; continuing from the newest '
                    f'state (update {tmp.updates}, step {sd.get("trainer_step")}), this process runs from step {now}')
            return st, f'NEWEST {p["state"]} (update {tmp.updates}; the checkpoint records {want})' + \
                (f'; {note}' if note else ''), None
        except Exception as ex:
            return st, f'fresh (the saved state was unreadable: {repr(ex)[:160]}); the file is kept', 's%s' % now
    return st, f'fresh (the checkpoint records {want} updates but no saved gate exists)', None


def _lc_refuse(S0, msg):
    """a launch that does not install still publishes what the checkpoint was trained with, and says why"""
    try:
        S0['why'] = msg
        _lc_publish(S0)
        _lc_atomic_json(S0['p']['status'], dict(_lc_stats(S0), active=False))
    except Exception:
        pass
    return dict(installed=False, msg=msg)


def _lc_install(model):
    p = _lc_paths()
    if model is None or not hasattr(model, 'blocks') or not isinstance(getattr(model, 'c', None), dict):
        return dict(installed=False, msg='not installed: no block model in HOTMOD')
    if not callable(_T.get('_sdpa')) or not callable(_T.get('_rs_status')):
        return dict(installed=False, msg='not installed: the trainer has no _sdpa / _rs_status')
    if _T.get('ATTN_DTYPE', 'fp32') != 'fp32':
        return dict(installed=False, msg=f'not installed: attention dtype {_T.get("ATTN_DTYPE")} (this mod is fp32 only)')
    if not hasattr(torch._C, '_current_graph_task_id') or \
            not hasattr(torch.autograd.Variable._execution_engine, 'queue_callback'):
        return dict(installed=False, msg='not installed: this torch has no graph-task id / queue_callback')
    L, d, Hh = int(model.c['L']), int(model.c['d']), int(model.c['H'])
    if d % Hh or len(model.blocks) != L:
        return dict(installed=False, msg=f'not installed: geometry d={d} H={Hh} blocks={len(model.blocks)} L={L}')
    E = d // Hh
    old = _T.get('_LCTX')
    carried = old if (isinstance(old, dict) and old.get('fn') is _T['_sdpa']) else None
    if carried is not None:
        st = LctxState(L, Hh, E)
        note = st.load_state_dict(carried['st'].state_dict())  # v1 (rho, w, c) or v2 (r, w), in this process
        src = 'carried over from the copy of this mod already live in this process' + (f'; {note}' if note else '')
        keep = None
    else:
        st, src, keep = _lc_load_state(L, Hh, E, _H.get('step'))
    S = dict(st=st, idx={id(b): i for i, b in enumerate(model.blocks)}, p=p, installed_step=int(_H.get('step') or 0),
             source=src, keep_prev=keep, installed=False, active=False, disabled=False, why=None, post_crash=False,
             calls=0, bidir_calls=0, skipped_calls=0, fallback_calls=0, grad_passes=0, held=0, layers_updated=0,
             bad_layers=0, warm_left=WARM, fails=0, fail_streak=0, step_failed=False, recompute_errors=0,
             update_errors=0, persist_errors=0, last_error=None, id_left=ID_CHECKS, id_diffs=[], knobs=None,
             knobs_mtime=-1.0, fallback=set(), cb_gid=None, inflight_written=False, pins=0, extra_dirty=False)
    _lc_read_knobs(S)
    if carried is None and _lo.path.exists(p['inflight']):
        try:
            with open(p['inflight']) as f:
                mk = _lj.load(f)
        except Exception:
            mk = {}
        if mk.get('post_crash'):
            return _lc_refuse(S, f'not installed: the crash marker {p["inflight"]} says two launches in a row died '
                                 f'inside a step with lctx active ({mk}). Delete the marker to allow it again.')
        S['post_crash'] = True
        _lc_say(f'WARNING: the last process died inside a step with lctx active ({mk}); installing again (a second '
                f'crash like it switches lctx off at the next launch)')
    if _lo.path.exists(p['off']):
        return _lc_refuse(S, f'not installed: {p["off"]} present')
    dev = next(model.parameters()).device
    devs = [torch.device('cpu')] + ([dev] if dev.type != 'cpu' else [])
    for sd_ in devs:
        bad = _lc_selftest(sd_)
        if bad is not None:
            return _lc_refuse(S, f'not installed: self-test on {sd_}: {bad}')
    prev_sdpa, prev_status = _T['_sdpa'], _T['_rs_status']
    if carried is not None:
        prev_sdpa = carried['prev']
        if _T['_rs_status'] is carried.get('fn_status'):
            prev_status = carried['prev_status']
        carried['retired'] = True  # an older copy still wrapped somewhere passes straight through from now on
    S['prev'], S['prev_status'] = prev_sdpa, prev_status
    S['sinks'] = [_lc_make_sink(S, i) for i in range(L)]
    S['installed'], S['active'] = True, True
    globals()['_LC'] = S
    LctxAttn.SAFE_GATE = True
    S['fn'], S['fn_status'] = _lc_make_sdpa(prev_sdpa), _lc_make_status(prev_status)
    _T['_sdpa'] = S['fn']
    _T['_rs_status'] = S['fn_status']
    _T['_LCTX'] = S
    _lc_persist(S, extra=True)  # every launch records what this checkpoint is trained with
    sm = st.summary()
    msg = (f'installed {VG} (core {V}, form {FORM}): the trainer\'s _sdpa adds a learned forget gate to every causal '
           f'attention call (per layer and head: rate softplus(r + w.unit(v))); bidirectional passes unchanged; '
           f'L={L} H={Hh} hd={E}; state {src}; span median {sm["span_median"]:.3g} tokens; gradients of the first {WARM} '
           f'training backwards dropped, then one Adam step per backward (no_grad passes change nothing); self-tests '
           f'passed on {[str(x) for x in devs]}; kill: touch {p["off"]} (from the next step)')
    _lc_say(msg)
    return dict(installed=True, version=VG, msg=msg, state=p['state'], off=p['off'], source=src)


try:
    HOTMOD_RESULT = _lc_install(_H.get('model'))
except Exception as _lc_ex:  # an install failure leaves the trainer exactly as it was
    HOTMOD_RESULT = dict(installed=False, msg='not installed: install raised ' + repr(_lc_ex)[:400])
if not HOTMOD_RESULT.get('installed'):
    _lc_say(str(HOTMOD_RESULT.get('msg')))
