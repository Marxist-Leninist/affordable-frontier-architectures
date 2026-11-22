#!/usr/bin/env python3
"""Render README.md from results/equivalence.json, results/bench_*.json and checkpoints/*/metrics.json.
Every number comes from those files; the descriptive text names what each file records."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
eq = json.loads((HERE / 'results' / 'equivalence.json').read_text())
benches = [json.loads(p.read_text()) for p in sorted((HERE / 'results').glob('bench_*.json'))]
ckpts = sorted((HERE / 'checkpoints').glob('*/metrics.json'))


def e(v):
    return f'{v:.1e}'


rows, ex, pt = eq['random'], eq['extra'], eq.get('pretrained')
dec = [v for r in rows for k, v in r.items() if k.startswith('decode_prefill') and k.endswith('_max_abs') and '_vs_' not in k]
layers = sorted({r['layers'] for r in rows})
groups = sorted({r['groups'] for r in rows})
Ts = sorted({r['T'] for r in rows})
L = [
    '# Exact fast CPU path for Hugging Face Mamba-2',
    '',
    '`mamba2_cpu_ssd.py` makes transformers\' `Mamba2ForCausalLM` compute the same function on CPU with less work and far less '
    'memory traffic. On CPU, transformers runs Mamba-2 through `Mamba2Mixer.torch_forward`, which builds '
    '(batch, chunk, l, s, heads, state)-sized temporaries and repeats B and C for every head. This patch uses einsum '
    'contractions that compute C.B^T once per group of heads. Only the full-sequence path (training, scoring, prompt prefill) '
    'is replaced. One-token decoding keeps transformers\' own recurrent step, and the patched prefill writes the same cache.',
    '',
    '```python',
    'from transformers import Mamba2ForCausalLM',
    'from mamba2_cpu_ssd import patch_mamba2',
    'model = Mamba2ForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")',
    'patch_mamba2(model)   # returns the number of mixers patched; unpatch_mamba2(model) undoes it',
    '```',
    '',
    'Patching swaps each mixer\'s class to `FastMamba2Mixer`, a `Mamba2Mixer` subclass defined in the module, so a patched model '
    'still pickles (`torch.save` / `torch.load`) when `mamba2_cpu_ssd` is importable. The kernel is adapted from the `ssd_minimal` '
    'reference in state-spaces/mamba and the surrounding code follows transformers\' `torch_forward`. Both are Apache-2.0; see '
    '`THIRD_PARTY_NOTICES.md`.',
    '',
    f"Tested with transformers {eq['transformers']} and torch {eq['torch']} on CPU. `patch_mamba2` warns when the installed "
    f"`torch_forward` source differs from the tested one (sha256 `{(eq['torch_forward_source_sha256'] or 'unreadable')[:16]}...`) "
    'or cannot be read; in that case run `test_equivalence.py`.',
    '',
    '## Same outputs',
    '',
    f"`test_equivalence.py` compares the patched model with stock transformers on {len(rows)} random cases: "
    f"{layers[0]}-{layers[-1]} layers, {groups[0]}-{groups[-1]} groups, projection biases on and off, the conv bias on and off, "
    f"a dt limit, a left-padding mask, an untied head, and lengths {Ts[0]} to {Ts[-1]} (including lengths that are not a "
    'multiple of the chunk). All comparisons are in float32 unless stated.',
    '',
    f'- logits: worst max-abs difference {e(max(r["logits_max_abs"] for r in rows))}',
    f'- every parameter gradient (cross-entropy loss): worst max-abs difference relative to that gradient\'s scale '
    f'{e(max(r["grad_max_rel"] for r in rows))}. This is float32 rounding: the contraction order differs, and both paths '
    'carry rounding error of this size on poorly conditioned parameters such as `D` and `A_log`.',
    f'- prefill with cache (half the text, and a single token), then step-by-step decoding: worst max-abs logit difference {e(max(dec))}',
    f"- greedy `generate()` on a left-padded batch: {'identical tokens' if all(r.get('padded_generate_identical', True) for r in rows) else 'DIFFERENT tokens'}",
    f"- a patched model through `torch.save` / `torch.load`: logits identical (max abs {ex['pickle_roundtrip_max_abs']:.1e}); "
    f"`unpatch_mamba2` restores the stock class and stock logits exactly (max abs {ex['unpatched_vs_stock_max_abs']:.1e}); "
    f"patching twice changes nothing ({ex['patch_counts']})",
    f"- a float64 model: logits within {e(ex['float64_logits_max_abs'])} of stock; under CPU bf16 autocast, logits "
    + ('identical to stock' if ex['bf16_autocast_logits_max_abs'] == 0 else
       f"within {ex['bf16_autocast_logits_max_abs']:.3f} of stock at a logit scale of {ex['bf16_autocast_logits_scale']:.1f}")
    + ' (both paths keep the SSD core in float32 under autocast)',
]
if pt:
    L.append(f"- pretrained `{pt['repo']}` on a {pt['tokens']}-token English paragraph: logits max-abs difference "
             f"{e(pt['logits_max_abs'])} at a logit scale of {pt['logits_scale']:.1f}, arg-max agreement "
             f"{pt['argmax_agreement'] * 100:.2f}%, loss {pt['nll_ref']:.6f} vs {pt['nll_new']:.6f} nats, greedy generation of "
             f"{pt['generated_tokens']} tokens {'identical' if pt['generation_identical'] else 'DIFFERENT'}")
L += ['', f"Result: {'all checks pass' if eq['all_pass'] else 'SOME CHECKS FAILED'} (`results/equivalence.json`).", '',
      '## Time and memory', '',
      '`bench.py` uses the same weights and inputs and one CPU thread. Each implementation runs in its own fresh process per '
      'round; rounds alternate which implementation goes first. Each pass records wall time, user and system CPU time and page '
      'faults. "fwd+bwd" is a forward and backward pass without an optimizer step. Peak RSS includes the model and Python. The stock '
      'path allocates large short-lived temporaries, so its speed depends on the memory allocator and on the machine\'s memory '
      'pressure. Every case is therefore run under the default glibc allocator and under tcmalloc (`LD_PRELOAD`).', '']
LABEL = {'fwd_bwd_small_b8_T256': 'fwd+bwd, 0.47M-param byte model (4 layers, d 128, chunk 32), batch 8 x 256',
         'forward_130m_T256': 'forward, mamba2-130m, 1 x 256 tokens', 'forward_130m_T512': 'forward, mamba2-130m, 1 x 512 tokens',
         'forward_130m_T2048': 'forward, mamba2-130m, 1 x 2048 tokens', 'fwd_bwd_130m_b1_T256': 'fwd+bwd, mamba2-130m, 1 x 256 tokens'}
ranges, sys_share = {}, []
for b in benches:
    m = b['memory_at_start']
    L += [f"**{b['label']}**: {b['cpu']}, {b['logical_cpus']} logical CPUs, {b['threads']} thread used; at the start "
          f"{m['MemAvailable'] / 1024:.1f} of {m['MemTotal'] / 1024:.1f} GiB memory available and "
          f"{(m['SwapTotal'] - m['SwapFree']) / 1024:.1f} of {m['SwapTotal'] / 1024:.1f} GiB swap in use, load {m['load1']:.1f}.", '',
          '| case | allocator | stock | patched | speed-up (wall) | speed-up (user CPU) | stock system time | stock minor faults per pass | peak RSS stock / patched |',
          '|---|---|---|---|---|---|---|---|---|']
    for case, by_alloc in b['cases'].items():
        for alloc, r in by_alloc.items():
            if 'new' not in r:
                continue
            n = r['new']
            if 'hf' in r:
                h = r['hf']
                ranges.setdefault(case, []).append(r['speedup'])
                if alloc == 'glibc':
                    sys_share.append((LABEL.get(case, case).split(',')[0] + ' ' + LABEL.get(case, case).split(', ', 1)[1],
                                      h['sys_s'] / max(1e-9, h['user_s'] + h['sys_s'])))
                L.append(f"| {LABEL.get(case, case)} | {alloc} | {h['median_s']:.3f} s | {n['median_s']:.3f} s | **{r['speedup']:.1f}x** | "
                         f"{r['speedup_user_cpu']:.1f}x | {h['sys_s']:.2f} s | {h['minor_faults']:,.0f} | "
                         f"{h['peak_rss_mib']:,.0f} / {n['peak_rss_mib']:,.0f} MiB |")
            else:
                L.append(f"| {LABEL.get(case, case)} | {alloc} | not run | {n['median_s']:.3f} s | - | - | - | - | - / {n['peak_rss_mib']:,.0f} MiB |")
    L.append('')
if ranges:
    L += ['Across these runs the patched path was ' + '; '.join(
        f"{min(v):.1f}-{max(v):.1f}x faster for {LABEL.get(k, k).split(',')[0]} ({LABEL.get(k, k).split(', ', 1)[1]})"
        for k, v in ranges.items()) + '. Under glibc, system time (mostly page-fault handling for the stock path\'s temporaries) '
        'was ' + ', '.join(f'{s * 100:.0f}% of the stock CPU time for {n}' for n, s in sys_share) + '; tcmalloc reuses that memory '
        'instead. Expect different ratios on other machines, thread counts, lengths and memory loads, and run `bench.py` to measure '
        'your own.', '',
        '"not run": stock transformers was not timed for those cases. At its measured speed for 256 and 512 tokens, the 2048-token '
        'forward and the 130m forward+backward would each take several minutes per pass.', '']
if ckpts:
    L += ['## Checkpoints', '',
          'Two byte-level Mamba-2 models from the OM2 study, in standard transformers format (`checkpoints/<name>/`); they load '
          'with stock transformers and need no patch. They were trained on one CPU thread by the study\'s own single-group copy of '
          'this chunked kernel (`om2bench.py`, arm `m2`). Their logits with `patch_mamba2` equal the study model\'s, and stock '
          'transformers agrees to the export-check value below.', '',
          '| checkpoint | params | updates x batch x bytes | train time | test bits/byte (macro) | article | fiction | dialogue | export check (stock vs study, max abs) |',
          '|---|---|---|---|---|---|---|---|---|']
    for f in ckpts:
        m = json.loads(f.read_text())
        t, c = m['test_bpb'], m['export_check']
        L.append(f"| `{f.parent.name}` | {m['params']:,} | {m['steps']:,} x {m['batch']} x {m['ctx']} | {m['train_s'] / 60:.1f} min | "
                 f"{t['macro']:.4f} | {t['article']:.4f} | {t['fiction']:.4f} | {t['dialogue']:.4f} | "
                 f"{e(c['study_vs_transformers_max_abs'])} ({c['windows']} windows) |")
    L += ['', 'Test set: the 303 held-out windows of the P056 corpus; every scored byte sees 129-256 bytes of context. Each model card '
          'has a usage example and a note on saving.', '']
L += ['## Names used here', '',
      '- **P056**: a human-written three-domain byte corpus (WikiText-2 articles, Project Gutenberg fiction, OpenAssistant '
      'dialogue) built for these experiments, with a source-entity split and train/test overlap exclusion. Sources and licences '
      'are in `THIRD_PARTY_NOTICES.md`. The corpus text is not redistributed here.',
      '- **OM2**: the study that trained the checkpoints (Transformer, Mamba-2, two versions of our own Mamba, and a gated '
      'exact-token store, all matched at about 0.47M parameters, with one all-metrics index). Its code, results and '
      'checkpoints are in `../om2-mamba-v2/`; `export_hf.py` needs its `om2bench.py`.', '',
      '## Files', '',
      '- `mamba2_cpu_ssd.py`: `ssd_chunked` (the kernel), `fast_torch_forward`, `FastMamba2Mixer`, `patch_mamba2` / `unpatch_mamba2`',
      '- `test_equivalence.py`: the checks above (`--skip-pretrained` for offline use)',
      '- `bench.py`: the timing tables above (`--label` names the machine; `--allocators glibc,tcmalloc`)',
      '- `export_hf.py`: turns an OM2 `m2` run into a transformers checkpoint and checks its logits three ways',
      '- `make_readme.py`: renders this page from the JSON files',
      '- `results/equivalence.json`, `results/bench_*.json`, `checkpoints/*/metrics.json`: the raw numbers',
      '- `THIRD_PARTY_NOTICES.md`, `SHA256SUMS`',
      '', '## Scope and limits', '',
      '- CPU. CUDA users already have the mamba_ssm kernels; this patch does not touch that path.',
      '- The speed-ups compare against transformers\' own CPU fallback, not against hand-written C++ or the CUDA kernels.',
      '- Agreement means floating-point rounding (the contraction order differs), not bit identity.',
      '- This folder states no licence of its own; the third-party code it adapts keeps its Apache-2.0 licence (`THIRD_PARTY_NOTICES.md`).',
      '']
(HERE / 'README.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
