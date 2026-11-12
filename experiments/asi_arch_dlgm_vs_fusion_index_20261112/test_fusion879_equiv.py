#!/usr/bin/env python3
"""test_fusion879_equiv.py -- fidelity test for the fusion879 plugin (claude-fable52, 2026-11-12).
Loads the ORIGINAL ASI-Arch #879 program (delta_net_psfr, from upstream/106.json at the pinned commit), copies its weights into the plugin's block
(dense fp control mode so the projections can be copied exactly) and compares the fused four-path output -- the tensor upstream hands to o_norm.
  phase cpu64 : float64 on CPU. FLA 0.2.2's ShortConvolution and l2norm only have GPU (Triton) kernels here, so for this phase they are replaced
                by the plain forms their own sources state (Conv1d forward truncated to T, then silu; x / sqrt(sum x^2 + 1e-6)), and the output norm
                that comes after the compared tensor is skipped; everything else is the untouched upstream code. Tolerance 1e-9.
  phase gpu32 : float32 on the GPU with the REAL FLA Triton kernels, nothing replaced. Tolerance 2e-4 (float32 through a 1024-token recursion).
Also: causality of the plugin's Fusion layer, the causal flag is ignored, layer 3 stays attention, the low-rank form runs. Exit 0 only if all pass."""
import json, os, sys
import torch, torch.nn.functional as F
torch.manual_seed(0)
torch.set_num_threads(2)
ROOT = '/workspace/asi_agillm'
STAGE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT + '/src')
import agillm_mini as sim
import fla
from fla.modules import ShortConvolution

entries = json.load(open(ROOT + '/upstream/106.json'))
e = next(x for x in entries if x['name'] == 'delta_net_psfr')
src = e['program']
assert e['index'] == 879 and src.count('@torch.compile') == 1, (e['index'], src.count('@torch.compile'))
ns = {'__name__': 'psfr_upstream'}
exec(compile(src.replace('@torch.compile', '# (torch.compile removed for the test)'), 'delta_net_psfr.py', 'exec'), ns)
FLA_L2NORM = ns['l2norm']
D, H = sim.SIM['d'], sim.SIM['H']
hd = D // H
OWN = ('short_q', 'short_k', 'short_v', 'fs', 'fl', 'b', 'ret', 'gate', 'log_temp')

def plain_shortconv(self, x, residual=None, mask=None, cache=None, output_final_state=False, cu_seqlens=None, **kw):
    y = self._conv_forward(x.transpose(1, 2), self.weight, self.bias)[..., :x.shape[1]].transpose(1, 2)
    return (F.silu(y) if self.activation is not None else y), cache

def upstream():
    torch.manual_seed(1)
    U = ns['DeltaNet'](hidden_size=D, num_heads=H).eval()
    with torch.no_grad():   # move every specially initialised parameter off its init so that each path and gate matters
        U.retention_param.copy_(torch.randn(H))
        U.fusion_gate.log_temp.copy_(0.3 * torch.randn(H))
        U.fir_short.filters.add_(0.3 * torch.randn_like(U.fir_short.filters))
        U.fir_long.filters.add_(0.1 * torch.randn_like(U.fir_long.filters))
        U.fusion_gate.mlp[-1].bias.add_(0.5 * torch.randn_like(U.fusion_gate.mlp[-1].bias))
    return U

def block(path, index=0):
    plugin = sim.load_plugin(path)
    c = dict(sim.SIM); c['expert_counts'] = [2 + int((i + 1) * 36 // 80 > i * 36 // 80) for i in range(c['L'])]
    FB, _ = plugin.patch(sim.Block, sim.Model, c)
    return plugin, FB(c, index).eval()

def copy_weights(U, blk):
    with torch.no_grad():
        blk.qkv.w.copy_(torch.cat([U.q_proj.weight, U.k_proj.weight, U.v_proj.weight], 0)); blk.qkv.scale.fill_(1.0)
        blk.short_q.copy_(U.q_conv1d.weight); blk.short_k.copy_(U.k_conv1d.weight); blk.short_v.copy_(U.v_conv1d.weight)
        blk.fs.copy_(U.fir_short.filters.reshape(H * hd, 1, -1)); blk.fl.copy_(U.fir_long.filters.reshape(H * hd, 1, -1))
        blk.b.weight.copy_(U.b_proj.weight); blk.b.bias.zero_()
        blk.ret.copy_(U.retention_param)
        for i in (0, 2):
            blk.gate[i].weight.copy_(U.fusion_gate.mlp[i].weight); blk.gate[i].bias.copy_(U.fusion_gate.mlp[i].bias)
        blk.log_temp.copy_(U.fusion_gate.log_temp)

def phase(name, dev, dtype, tol, replace_shortconv):
    U = upstream()
    convs = (U.q_conv1d, U.k_conv1d, U.v_conv1d)
    info = dict(shortconv=[(type(m).__name__, m.kernel_size[0], m.padding[0], m.activation, m.bias is None) for m in convs][0], replaced=replace_shortconv)
    ns['l2norm'] = (lambda x, eps=1e-6, output_dtype=None: x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)) if replace_shortconv else FLA_L2NORM
    if replace_shortconv:
        for m in convs:
            m.forward = plain_shortconv.__get__(m)
    _, blk = block(STAGE + '/fusion879.py')
    copy_weights(U, blk)
    U, blk = U.to(dev, dtype), blk.to(dev, dtype)
    if replace_shortconv:
        U.o_norm = torch.nn.Identity()   # FLA's RMSNorm is a GPU-only kernel too; it sits AFTER the compared tensor, so swapping it changes nothing measured
    cap = {}
    U.o_norm.register_forward_pre_hook(lambda m, inp: cap.__setitem__('fused', inp[0].detach().clone()))
    cases, ok = [], True
    for T in (50, 64, 100, 256, 257, 1024):
        torch.manual_seed(100 + T)
        h = torch.randn(2, T, D).to(dev, dtype)
        with torch.no_grad():
            U(h); mine = blk.mix(h, True)
        ref = cap['fused'].reshape(2, T, D)
        err, mag = float((mine - ref).abs().max()), float(ref.abs().max())
        cases.append(dict(T=T, max_abs_err=err, ref_max=mag)); ok = ok and err < tol * max(1.0, mag)
    return dict(name=name, device=str(dev), dtype=str(dtype), tol=tol, ok=bool(ok), cases=cases, **info), blk

sim.BinLinear.FP = True
res = dict(fla=getattr(fla, '__version__', '?'), torch=torch.__version__, upstream_index=e['index'], phases=[])
p1, blk = phase('cpu64', torch.device('cpu'), torch.float64, 1e-9, True)
res['phases'].append(p1)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(min(0.5, 720 / (torch.cuda.get_device_properties(0).total_memory / 2**20)))   # the queue's own 720 MiB cap
    try:
        p2, _ = phase('gpu32', torch.device('cuda'), torch.float32, 2e-4, False)
    except Exception as ex:
        p2 = dict(name='gpu32', ok=False, error=repr(ex)[:400])
    res['phases'].append(p2)
else:
    res['phases'].append(dict(name='gpu32', ok=False, error='no GPU visible'))
ok = all(p['ok'] for p in res['phases'])
with torch.no_grad():
    torch.manual_seed(7)
    h = torch.randn(2, 160, D, dtype=torch.float64); h2 = h.clone(); h2[:, 97] += 1.0
    a, b = blk.mix(h, True), blk.mix(h2, True)
    res['causality'] = dict(leak_before_edit=float((a[:, :97] - b[:, :97]).abs().max()), change_from_edit=float((a[:, 97:] - b[:, 97:]).abs().max()))
    res['ignores_causal_flag'] = bool(torch.allclose(blk.mix(h, False), a, rtol=0, atol=1e-12))
    _, attn = block(STAGE + '/fusion879.py', index=3)
    res['layer3_is_attention'] = bool(attn.attn_layer) and not hasattr(attn, 'fs')
    plr, lr = block(STAGE + '/fusion879_lr.py')
    lr = lr.double()
    y, y2 = lr.mix(h, True), lr.mix(h2, True)
    res['lr'] = dict(full_flag=bool(plr.FULL), shape=list(y.shape), finite=bool(torch.isfinite(y).all()), leak_before_edit=float((y[:, :97] - y2[:, :97]).abs().max()),
                     fp_params=sum(p.numel() for n, p in lr.named_parameters() if n.split('.')[0] in OWN))
    res['full_fp_params'] = sum(p.numel() for n, p in blk.named_parameters() if n.split('.')[0] in OWN)
ok = ok and res['causality']['leak_before_edit'] < 1e-12 and res['causality']['change_from_edit'] > 1e-6 and res['ignores_causal_flag'] and res['layer3_is_attention']
ok = ok and (not res['lr']['full_flag']) and res['lr']['finite'] and res['lr']['leak_before_edit'] < 1e-12 and res['lr']['shape'] == [2, 160, D]
res['ok'] = bool(ok)
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
