#!/usr/bin/env python3
"""Build the speed table index_all.py uses from the per-process speed passes on one machine:
  SPEED_JSON  (speed_all.py, second pass): training s/step, training memory (whole process and above the process
              baseline), per-sequence and total decode state, prefill and decode, 3 rounds per arm
  FIRST_JSON  (speed_all.py, first pass): training s/step, prefill and decode with the same code path, 3 rounds per arm
Rule: every timing is the MEDIAN over all valid per-process rounds of both passes. Exceptions: the state-gated store
(om2g) uses only the second pass, because its first-pass decode ran with an untrained gate (its decode work depends on the
weights); arms missing from the first pass (the tfplain control) use the second. Memory and state come from the second
pass (the first did not measure them). The best round of each timing is kept as *_best for the alternative way of
counting that uses it, and 'raw' lists every valid round with its own timings, for the seed x timing-round spread.
LATENCY_JSON (latency_interleaved.py): prefill and decode with all models interleaved in one process, kept per arm
('interleaved') for one of the alternative ways of counting. In that run every model switched between a fast mode, close to
its best per-process rounds, and a slow mode 25-35% slower, mostly in the same iterations for all models; the shared machine
flips between the two within tens of seconds, so no estimator here separates models whose speeds differ by a few per cent.
Single-pass medians are kept as 'pass1' and 'pass2' for the alternatives that use one pass only.
Usage: merge_speed.py SPEED_JSON FIRST_JSON LATENCY_JSON OUT_JSON"""
import json, statistics, sys
from pathlib import Path

S, F, Lt = (json.loads(Path(p).read_text()) for p in sys.argv[1:4])
SECOND_ONLY = {'om2g'}
TIMING = ('train_s_per_step', 'prefill_tok_s', 'decode_ms_median', 'decode_ms_p95', 'decode_tok_s')
MEMORY = ('train_peak_rss_mib', 'train_model_rss_mib', 'train_base_rss_mib', 'decode_state_bytes_at_256',
          'decode_state_bytes_per_seq_at_256')
out = dict({k: v for k, v in S.items() if k not in ('arms', 'raw')},
           timing=dict(rule='median over all valid per-process rounds of both passes (om2g and tfplain: second pass only)',
                       first_pass_time=F.get('time'), second_pass_time=S.get('time')),
           interleaved_reference=dict(source=Path(sys.argv[3]).name, iters=Lt['iters'], time=Lt['time'],
                                      decode_ms_median={a: v['decode_ms_median'] for a, v in Lt['arms'].items()}),
           arms={}, raw={})
for arm, v in S['arms'].items():
    rounds = [dict(x, timing_pass=2, timing_round=j) for j, x in enumerate(S['raw'][arm]) if 'error' not in x]
    if arm not in SECOND_ONLY:
        rounds += [dict(x, timing_pass=1, timing_round=j) for j, x in enumerate(F.get('raw', {}).get(arm, [])) if 'error' not in x]
    mem = {k: v[k] for k in MEMORY if k in v}          # second-pass medians
    head = {k: statistics.median(x[k] for x in rounds) for k in TIMING}
    best = dict(prefill_tok_s_best=max(x['prefill_tok_s'] for x in rounds),
                decode_ms_median_best=min(x['decode_ms_median'] for x in rounds),
                train_s_per_step_best=min(x['train_s_per_step'] for x in rounds))
    jit = dict(decode_ms_range=[min(x['decode_ms_median'] for x in rounds), max(x['decode_ms_median'] for x in rounds)],
               train_s_range=[min(x['train_s_per_step'] for x in rounds), max(x['train_s_per_step'] for x in rounds)],
               timing_rounds=len(rounds))
    p2 = [x for x in rounds if x['timing_pass'] == 2]
    p1 = [x for x in rounds if x['timing_pass'] == 1]
    pass2 = {k: statistics.median(x[k] for x in p2) for k in TIMING}       # single-pass medians (alternatives, the control)
    pass1 = {k: statistics.median(x[k] for x in p1) for k in TIMING} if p1 else dict(pass2)
    il = Lt['arms'].get(arm, {})
    out['arms'][arm] = dict({k: v[k] for k in ('arm', 'params', 'trained_weights', 'torch') if k in v}, **mem, **head, **best, **jit,
                            pass1=pass1, pass1_from_second=not p1, pass2=pass2,
                            interleaved=dict(prefill_tok_s=il.get('prefill_tok_s', head['prefill_tok_s']),
                                             decode_ms_median=il.get('decode_ms_median', head['decode_ms_median'])))
    out['raw'][arm] = [dict({k: x[k] for k in TIMING}, **{k: x.get(k, mem.get(k)) for k in MEMORY},
                            timing_pass=x['timing_pass'], timing_round=x['timing_round'],
                            prefill_tok_s_best=best['prefill_tok_s_best'], decode_ms_median_best=best['decode_ms_median_best'])
                       for x in rounds]
Path(sys.argv[4]).write_text(json.dumps(out, indent=1) + '\n')
for arm, v in out['arms'].items():
    print(f"{arm:8s} rounds {v['timing_rounds']} train {v['train_s_per_step']:.3f} decode {v['decode_ms_median']:.2f} "
          f"({v['decode_ms_range'][0]:.2f}-{v['decode_ms_range'][1]:.2f}) prefill {v['prefill_tok_s']:.0f}")
