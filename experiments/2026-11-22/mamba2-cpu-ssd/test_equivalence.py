#!/usr/bin/env python3
"""Check that patch_mamba2 reproduces transformers' Mamba-2.

Random configurations (1-4 layers, 1-3 groups, projection and conv biases on/off, dt limits, a left-padding
mask, lengths that are not a multiple of the chunk, an untied head): logits, every parameter gradient,
prefill + step-by-step decoding (prefill of half the text and of a single token), and greedy generate() on a
left-padded batch. Extra checks: torch.save/torch.load of a patched model, a float64 model, CPU bf16 autocast,
patching twice and unpatching. Optionally a pretrained checkpoint (default AntonV/mamba2-130m-hf): logits on real
text and greedy generation. Writes results/equivalence.json; exit status 1 if any check misses its tolerance.
"""
import argparse, copy, io, json, os, sys, time, warnings
from pathlib import Path

import torch
from torch.nn import functional as F
import transformers
from transformers import Mamba2Config, Mamba2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mamba2_cpu_ssd as fast  # noqa: E402

TEXT = ("The Industrial Revolution began in Great Britain in the late eighteenth century and spread to "
        "continental Europe and North America over the following decades. Mechanised spinning and weaving, "
        "the steam engine and new methods of producing iron changed how goods were made, how people worked "
        "and where they lived. Historians still debate why it started in Britain: cheap coal, high wages, "
        "colonial markets, a culture of tinkering and stable property rights have all been proposed, and "
        "most accounts combine several of them. Its effects on living standards were uneven at first, and "
        "the long working hours, child labour and crowded towns of the early factory era prompted reforms.")


def make_cfg(**kw):
    base = dict(vocab_size=97, hidden_size=64, num_heads=4, head_dim=32, state_size=16, n_groups=1, expand=2,
                conv_kernel=4, num_hidden_layers=2, chunk_size=16, use_cache=True, tie_word_embeddings=True,
                pad_token_id=0, bos_token_id=0, eos_token_id=0)
    base.update(kw)
    return Mamba2Config(**base)


CASES = [
    ('4 layers, chunk 32 (the OM2 size)', dict(vocab_size=256, hidden_size=128, num_heads=8, head_dim=32, state_size=32,
                                              num_hidden_layers=4, chunk_size=32), (1, 5, 31, 32, 33, 256), 2, False),
    ('3 groups, projection biases on, dt limit (0.01, 0.4)', dict(hidden_size=96, num_heads=6, head_dim=32, state_size=16,
                                                                 n_groups=3, use_bias=True, time_step_limit=(0.01, 0.4)),
     (17, 64), 2, False),
    ('2 groups, left-padding mask, 3 layers', dict(hidden_size=64, num_heads=4, head_dim=32, state_size=8, n_groups=2,
                                                   num_hidden_layers=3, chunk_size=64), (130,), 3, True),
    ('chunk 256 (longer and shorter than the text), untied head', dict(hidden_size=64, num_heads=2, head_dim=64,
                                                                       state_size=64, chunk_size=256,
                                                                       tie_word_embeddings=False), (300, 77), 2, False),
    ('1 layer, conv bias off, 4 groups of 1 head', dict(hidden_size=64, num_heads=4, head_dim=32, state_size=16, n_groups=4,
                                                         num_hidden_layers=1, chunk_size=8, use_conv_bias=False),
     (1, 2, 9, 40), 2, False),
]


def max_rel_grad(m1, m2):
    worst, name = 0.0, None
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        if p1.grad is None and p2.grad is None:
            continue
        g1, g2 = p1.grad, p2.grad
        scale = float(g1.abs().max()) + 1e-12
        r = float((g1 - g2).abs().max()) / scale
        if r > worst:
            worst, name = r, n1
    return worst, name


@torch.no_grad()
def decode(m, x, k):
    o = m(input_ids=x[:, :k], use_cache=True)
    cache, steps = o.cache_params, [o.logits[:, -1]]
    for t in range(k, x.shape[1]):
        o = m(input_ids=x[:, t:t + 1], cache_params=cache, use_cache=True, cache_position=torch.tensor([t]))
        cache = o.cache_params
        steps.append(o.logits[:, -1])
    return torch.stack(steps, 1)


def random_checks(seed):
    rows = []
    for title, kw, lengths, bsz, masked in CASES:
        for T in lengths:
            torch.manual_seed(seed + T)
            cfg = make_cfg(**kw)
            ref = Mamba2ForCausalLM(cfg).float()
            new = copy.deepcopy(ref)
            assert fast.patch_mamba2(new) == cfg.num_hidden_layers
            x = torch.randint(1, cfg.vocab_size, (bsz, T))
            y = torch.randint(0, cfg.vocab_size, (bsz, T))
            mask = None
            if masked:
                mask = torch.ones(bsz, T, dtype=torch.long)
                for i in range(1, bsz):
                    mask[i, :7 * i] = 0          # left padding of 7 and 14 tokens
                    x[i, :7 * i] = 0
            lr = ref(input_ids=x, attention_mask=mask, use_cache=False).logits
            ln = new(input_ids=x, attention_mask=mask, use_cache=False).logits
            d_logit = float((lr - ln).abs().max())
            F.cross_entropy(lr.flatten(0, 1), y.flatten()).backward()
            F.cross_entropy(ln.flatten(0, 1), y.flatten()).backward()
            g_rel, g_name = max_rel_grad(ref, new)
            row = dict(case=title, T=T, batch=bsz, layers=cfg.num_hidden_layers, groups=cfg.n_groups,
                       use_bias=cfg.use_bias, use_conv_bias=cfg.use_conv_bias, logits_max_abs=d_logit,
                       logits_scale=float(lr.abs().max()), grad_max_rel=g_rel, grad_worst_param=g_name)
            ok = d_logit <= 1e-4 and g_rel <= 1e-4
            if masked:
                with torch.no_grad():
                    gr = ref.generate(x, attention_mask=mask, max_new_tokens=12, do_sample=False)
                    gn = new.generate(x, attention_mask=mask, max_new_tokens=12, do_sample=False)
                row['padded_generate_identical'] = bool(torch.equal(gr, gn))
                ok = ok and row['padded_generate_identical']
            elif T >= 2:
                full = ref(input_ids=x, use_cache=False).logits
                for k in sorted({max(1, T // 2), 1}):
                    dr, dn = decode(ref, x, k), decode(new, x, k)
                    row[f'decode_prefill{k}_max_abs'] = float((dr - dn).abs().max())
                    row[f'decode_prefill{k}_vs_full_max_abs'] = float((dn - full[:, k - 1:]).abs().max())
                    ok = ok and row[f'decode_prefill{k}_max_abs'] <= 1e-4
            row['pass'] = bool(ok)
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def extra_checks(seed):
    torch.manual_seed(seed)
    cfg = make_cfg(hidden_size=64, num_heads=4, head_dim=32, state_size=16, n_groups=2, num_hidden_layers=2, chunk_size=16)
    ref = Mamba2ForCausalLM(cfg).float().eval()
    new = copy.deepcopy(ref)
    x = torch.randint(1, cfg.vocab_size, (2, 45))
    out = {}
    # patch twice is a no-op, unpatch restores the stock class
    out['patch_counts'] = [fast.patch_mamba2(new), fast.patch_mamba2(new)]
    with torch.no_grad():
        lp = new(input_ids=x, use_cache=False).logits
        lr = ref(input_ids=x, use_cache=False).logits
    # torch.save / torch.load of the whole patched model
    buf = io.BytesIO()
    torch.save(new, buf)
    buf.seek(0)
    back = torch.load(buf, weights_only=False)
    with torch.no_grad():
        lb = back(input_ids=x, use_cache=False).logits
    out['pickle_roundtrip_max_abs'] = float((lb - lp).abs().max())
    out['pickle_class'] = type(back.backbone.layers[0].mixer).__name__
    probe = copy.deepcopy(new)
    out['unpatch_count'] = fast.unpatch_mamba2(probe)
    with torch.no_grad():
        lu = probe(input_ids=x, use_cache=False).logits
    out['unpatched_vs_stock_max_abs'] = float((lu - lr).abs().max())
    # float64 model
    r64, n64 = copy.deepcopy(ref).double(), copy.deepcopy(ref).double()
    fast.patch_mamba2(n64)
    with torch.no_grad():
        a, b = r64(input_ids=x, use_cache=False).logits, n64(input_ids=x, use_cache=False).logits
    out['float64_logits_max_abs'] = float((a - b).abs().max())
    # CPU bf16 autocast: the SSD core of both paths stays in float32
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        a, b = ref(input_ids=x, use_cache=False).logits, new(input_ids=x, use_cache=False).logits
    out['bf16_autocast_logits_max_abs'] = float((a.float() - b.float()).abs().max())
    out['bf16_autocast_logits_scale'] = float(a.float().abs().max())
    out['pass'] = bool(out['patch_counts'] == [2, 2] and out['pickle_roundtrip_max_abs'] == 0.0 and
                       out['pickle_class'] == 'FastMamba2Mixer' and out['unpatch_count'] == 2 and
                       out['unpatched_vs_stock_max_abs'] == 0.0 and out['float64_logits_max_abs'] <= 1e-4 and
                       out['bf16_autocast_logits_max_abs'] <= 0.05 * max(1.0, out['bf16_autocast_logits_scale']))
    print(json.dumps(out), flush=True)
    return out


def pretrained_check(repo):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    ref = Mamba2ForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).eval()
    new = copy.deepcopy(ref)
    fast.patch_mamba2(new)
    ids = tok(TEXT, return_tensors='pt').input_ids
    with torch.no_grad():
        lr, ln = ref(input_ids=ids, use_cache=False).logits, new(input_ids=ids, use_cache=False).logits
        nll_r = float(F.cross_entropy(lr[0, :-1], ids[0, 1:]))
        nll_n = float(F.cross_entropy(ln[0, :-1], ids[0, 1:]))
        prompt = ids[:, :48]
        am = torch.ones_like(prompt)
        gr = ref.generate(prompt, attention_mask=am, max_new_tokens=40, do_sample=False)
        gn = new.generate(prompt, attention_mask=am, max_new_tokens=40, do_sample=False)
    return dict(repo=repo, tokens=int(ids.shape[1]), logits_max_abs=float((lr - ln).abs().max()),
                logits_scale=float(lr.abs().max()), argmax_agreement=float((lr.argmax(-1) == ln.argmax(-1)).float().mean()),
                nll_ref=nll_r, nll_new=nll_n, generation_identical=bool(torch.equal(gr, gn)),
                generated_tokens=int(gn.shape[1] - prompt.shape[1]),
                sample=tok.decode(gn[0, prompt.shape[1]:]),
                pass_=bool(float((lr - ln).abs().max()) <= 2e-3 and torch.equal(gr, gn)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20261122)
    ap.add_argument('--pretrained', default='AntonV/mamba2-130m-hf')
    ap.add_argument('--skip-pretrained', action='store_true')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'results' / 'equivalence.json'))
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    warnings.filterwarnings('ignore')
    t0 = time.time()
    rows = random_checks(a.seed)
    extra = extra_checks(a.seed)
    res = dict(transformers=transformers.__version__, torch=torch.__version__, kernel_version=fast.__version__,
               torch_forward_source_sha256=fast.torch_forward_source_sha(), random=rows, extra=extra,
               random_pass=all(r['pass'] for r in rows))
    if not a.skip_pretrained:
        res['pretrained'] = pretrained_check(a.pretrained)
        print(json.dumps(res['pretrained']), flush=True)
    res['all_pass'] = bool(res['random_pass'] and extra['pass'] and res.get('pretrained', {}).get('pass_', True))
    res['wall_s'] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print('ALL_PASS' if res['all_pass'] else 'SOME_FAILED', flush=True)
    sys.exit(0 if res['all_pass'] else 1)


if __name__ == '__main__':
    main()
