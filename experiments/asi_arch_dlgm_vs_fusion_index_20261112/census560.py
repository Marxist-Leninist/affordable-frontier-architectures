"""census560.py -- storage/state bill of each candidate at the trainer's current depth (560 blocks), computed the way the simulator's own census()
does it (module catalog on the meta device: no training, no allocation). 380 blocks is included as a cross-check against what every queue run records."""
import json, sys
import torch
sys.path.insert(0, '/workspace/asi_agillm/src')
import agillm_mini as sim
P = '/workspace/asi_agillm/plugins/'
formula = lambda L: [2 + int((i + 1) * 36 // 80 > i * 36 // 80) for i in range(L)]
MIX = ('short_q', 'short_k', 'short_v', 'lc', 'mc', 'fs', 'fl', 'b', 'ret', 'rt', 'gate', 'log_temp')
out = dict(formula_matches_380=formula(380) == sim.CFG_600['expert_counts'], rows={})
for L in (380, 560):
    for name in ('base', 'dlgm458', 'dlgm458_lr', 'fusion879', 'fusion879_lr'):
        cfg = dict(sim.CFG_600); cfg['L'] = L; cfg['expert_counts'] = formula(L)
        plugin = sim.load_plugin(P + name + '.py')
        B, M = sim.Block, sim.Model
        if hasattr(plugin, 'patch'):
            r = plugin.patch(sim.Block, sim.Model, cfg)
            if r is not None:
                B, M = r
        with torch.device('meta'):
            m = M(cfg, B)
        table = m.table
        body = [x for x in m.modules() if isinstance(x, sim.BinLinear) and x is not table]
        mixer = sum(p.numel() for n, p in m.named_parameters() if n.startswith('blocks.') and n.split('.')[2] in MIX)
        out['rows'][f'{name}@{L}'] = dict(latent_bytes=int(sum(x.shape[0] * x.shape[1] for x in body)), table_bytes=int(table.shape[0] * table.shape[1]), n_matrices=len(body),
                                          fp_params=int(sum(p.numel() for p in m.parameters())), kv_bytes_per_token=int(sum(b.kv_bytes_per_token() for b in m.blocks)),
                                          mixer_fp_params=int(mixer), candidate_layers=sum(1 for b in m.blocks if getattr(b, 'attn_layer', True) is False), layers=L)
        del m
        print(name, L, 'done', file=sys.stderr, flush=True)
json.dump(out, open('/workspace/fusion879_stage/census560.json', 'w'))
print(json.dumps(out))
