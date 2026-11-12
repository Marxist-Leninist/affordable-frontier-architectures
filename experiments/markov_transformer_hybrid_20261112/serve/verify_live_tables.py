#!/usr/bin/env python3
"""verify_live_tables.py EXPECTED_TABLE_ID [--wait=MIN] [--window=N] -- on the serving box, after current.json was
pointed at a new table folder: show that the running worker serves exactly those tables. Two /score requests of one
dev window (128 targets): the model alone and chain_mix 0.5. The second must equal -log(0.5 p_model + 0.5 p_chain)
token by token, with p_chain computed here from the folder current.json names, and must report the expected table id.
It never cancels anybody's request: it waits for an idle slot (default up to 30 min) and gives up otherwise.
Read-only apart from the two requests."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, '/workspace/fable52_markov_20261112')
import chainmix512 as cmx  # noqa: E402

URL = 'http://127.0.0.1:18773'
CHAIN_ROOT = '/workspace/agillm256b_serve/chain'
WINDOWS = '/workspace/claude_servetune_20261107/data/devset_v3_first129.json'
want_id = sys.argv[1]
wait_min = next((float(a[7:]) for a in sys.argv[2:] if a.startswith('--wait=')), 30.0)
window = next((int(a[9:]) for a in sys.argv[2:] if a.startswith('--window=')), 0)


def health():
    try:
        with urllib.request.urlopen(URL + '/health', timeout=5) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        return None


def idle(h):
    return bool(h and h.get('ready') and not h.get('busy') and not (h.get('active_request_ids') or []) and not int(h.get('active_count') or 0))


def wait_slot(deadline):
    quiet = None
    while time.time() < deadline:
        if idle(health()):
            quiet = quiet or time.time()
            if time.time() - quiet >= 3:
                return True
        else:
            quiet = None
        time.sleep(1)
    return False


def score(req, tag, deadline):
    for attempt in range(40):
        if not wait_slot(deadline):
            raise SystemExit('VERIFY-GAVE-UP: the worker had no idle slot within %.0f min (nothing was cancelled)' % wait_min)
        q = dict(req, request_id='fable52markov-%s-%d-%d' % (tag, attempt, int(time.time())))
        try:
            with urllib.request.urlopen(urllib.request.Request(URL + '/score', data=json.dumps(q).encode(),
                                                               headers={'Content-Type': 'application/json'}), timeout=1800) as r:
                out = json.load(r)
            if isinstance(out.get('per_token'), list) and len(out['per_token']) == len(req['ids']) - 1:
                return out
            time.sleep(10)                       # cancelled for a page run, or cut short: try again when the slot is free
        except urllib.error.HTTPError as e:
            body = e.read()[:400].decode('utf-8', 'replace')
            if e.code in (409, 429, 503):
                time.sleep(3)
                continue
            raise SystemExit('VERIFY-LIVE-FAILED: score HTTP %d %s' % (e.code, body))
    raise SystemExit('VERIFY-GAVE-UP: 40 attempts')


cur = json.load(open(os.path.join(CHAIN_ROOT, 'current.json')))
t = cmx.Tables(os.path.join(CHAIN_ROOT, str(cur['tables'])))
t.selftest()
if t.meta['id'] != want_id:
    raise SystemExit('VERIFY-LIVE-FAILED: current.json names %s (%s), expected %s' % (cur['tables'], t.meta['id'], want_id))
ids = [int(v) for v in json.load(open(WINDOWS))['windows'][window]['ids']]
deadline = time.time() + wait_min * 60
h = health() or {}
s0 = score(dict(ids=ids, per_token=True), 'plain', deadline)
s1 = score(dict(ids=ids, per_token=True, chain_mix=0.5), 'chain', deadline)
cm = s1.get('chain_mix') or {}
pc = t.target_prob(ids)
m, x = np.array(s0['per_token'], dtype=np.float64), np.array(s1['per_token'], dtype=np.float64)
worst = float(np.abs(-np.log(0.5 * np.exp(-m) + 0.5 * pc) - x).max())
ok = worst < 5e-3 and cm.get('table') == want_id and cm.get('tokens') == len(m) and 'chain_mix' not in s0
print(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), step=h.get('step'), name=h.get('name') or h.get('model_name'), window=window, tokens=int(len(m)),
                      table_served=cm.get('table'), table_expected=want_id, folder=cur['tables'], model_alone=round(float(m.mean()), 4),
                      chain_alone=round(float(-np.log(pc).mean()), 4), chain_mix_0_5_served=round(float(x.mean()), 4),
                      chain_mix_0_3=round(float(-np.log(0.7 * np.exp(-m) + 0.3 * pc).mean()), 4),
                      chain_mix_0_7=round(float(-np.log(0.3 * np.exp(-m) + 0.7 * pc).mean()), 4),
                      worst_token_difference=worst, seconds=[s0.get('seconds'), s1.get('seconds')])))
print('VERIFY-LIVE-OK' if ok else 'VERIFY-LIVE-FAILED')
