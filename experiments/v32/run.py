"""v32: validation-calibrated match-order mixture, frozen before scoring."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch
from models import LM, state_bytes
from local_attention import localize
from prefill import prefill
from pruning import local_prefill
from compact_cache import CompactCache, reachable_bytes

P = Path(__file__).parent
ARMS = ('local', 'gru')
VARIANTS = ('neural', 'historical', 'constant', 'by_order')
HISTORICAL = {'local': .3, 'gru': .2}
SEEDS = (1, 2, 3)
GRID = np.arange(11) / 10
TRAIN = P.parent / 'frontier_research_v8'
CHECKPOINTS = P.parent / 'frontier_research_v19/checkpoints'
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()

def sha(path):
    return sha_bytes(path.read_bytes())

def write(name, obj):
    (P / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def load(arm, seed):
    file = CHECKPOINTS / f'{arm}_seed{seed}_adapt500.pt'
    expected = json.loads((P / 'checkpoint_manifest.json').read_text())
    assert sha(file) == expected[file.name]
    model = LM('gru' if arm == 'gru' else 'transformer')
    if arm == 'local':
        model = localize(model, 128)
    model.load_state_dict(torch.load(file, weights_only=True))
    return model.eval()

def data(split):
    manifest = json.loads((P / 'items.json').read_text())
    if split == 'validation':
        raw = (TRAIN / 'validation.txt').read_bytes()
        assert sha_bytes(raw) == json.loads((P / 'training_data_manifest.json').read_text())['splits']['validation']['text_sha256']
        out = [dict(item, raw=raw[item['start']:item['start'] + 4097]) for item in manifest['validation']]
    else:
        assert sha(P / 'rows.json') == json.loads((P / 'data_manifest.json').read_text())['response_sha256']
        rows = {r['row_idx']: r['row']['text'].encode() for r in json.loads((P / 'rows.json').read_text())['rows']}
        out = [dict(item, raw=rows[item['doc']][item['start']:item['start'] + 4097]) for item in manifest['evaluation']]
    for item in out:
        assert sha_bytes(item['raw']) == item['window_sha256']
    return out

@torch.inference_mode()
def score(model, arm, items):
    rows = []
    for start in range(0, len(items), 4):
        tokens = torch.tensor([list(item['raw']) for item in items[start:start + 4]])
        logits, _ = (local_prefill(model, tokens[:, :-1], 128, prune=True)
                     if arm == 'local' else prefill(model, tokens[:, :-1], None, 128))
        loss = -logits.double().log_softmax(-1).gather(-1, tokens[:, -128:, None]).squeeze(-1) / math.log(2)
        rows.extend(loss.numpy())
    return np.array(rows)

def copy_scores(items):
    probabilities, orders, counts = [], [], []
    for item in items:
        cache = CompactCache()
        pp, kk, nn = [], [], []
        for position, byte in enumerate(item['raw'][:-1]):
            cache.feed(byte)
            if position >= 3968:
                probability, count, order = cache.target_probability(item['raw'][position + 1])
                pp.append(probability)
                kk.append(order)
                nn.append(count)
        probabilities.append(pp)
        orders.append(kk)
        counts.append(nn)
    return np.array(probabilities), np.array(orders), np.array(counts)

def weights(variant, arm, order, selection):
    if variant == 'neural':
        return np.zeros_like(order, dtype=float)
    if variant == 'historical':
        return HISTORICAL[arm] * (order > 0)
    if variant == 'constant':
        return selection[arm]['constant'] * (order > 0)
    result = np.zeros_like(order, dtype=float)
    for key, value in selection[arm]['by_order'].items():
        result[order == int(key)] = value
    return result

def mixture_loss(neural_loss, copy_probability, weight):
    return -np.log2((1 - weight) * np.exp2(-neural_loss) + weight * copy_probability)

def verify():
    protocol = json.loads((P / 'protocol.json').read_text())
    for file, expected in protocol['source_sha256'].items():
        assert sha(P / file) == expected, file
    for file, expected in protocol['inputs_sha256'].items():
        assert sha(P / file) == expected, file
    return protocol

def freeze():
    assert not (P / 'protocol.json').exists(), 'Never refreeze historical protocol'
    write('protocol.json', {
        'experiment': 'v32 validation-calibrated match-order copy mixture',
        'hypothesis': 'Use only already-observed longest exact match length to choose copy weight. Does it improve held-out loss versus a constant retuned on the identical validation ledger?',
        'design': 'Same frozen neural checkpoints, full CompactCache orders4/8/16, pseudocount1, identical target ledger; no target-dependent gating, no new neural training.',
        'arms': ARMS, 'seeds': SEEDS, 'variants': VARIANTS,
        'selection': 'First32 eligible nonoverlapping4096-input WikiText validation windows after exact training-target exclusion; final128 targets. Fit one constant per arm and one weight per observed order4/8/16 separately on grid0,.1,...1, pooled over3seeds. First/lower grid weight wins ties. Missing bucket falls back to constant. Save selection before any FineWeb scoring.',
        'evaluation': 'FineWeb-Edu pinned rows1800-1899, first8193 bytes/document; nonoverlapping4096-input windows, final128 targets. Exclude exact full-document or target128 overlaps with training/validation/prior/current other documents before scoring.',
        'primary_contrasts': ['local by_order minus constant', 'gru by_order minus constant'],
        'quality_gate': 'Improvement supported only when upper paired document-bootstrap95% interval <0, both primary comparisons retained without winner selection; unadjusted multiple-comparison caveat.',
        'statistics': 'Average3seed losses within each target, then5000 target-weighted document cluster resamples, seed32. Conditional convenience-cohort uncertainty; seeds not independent documents.',
        'timing': 'First8 eligible evaluation documents with4128 bytes; seeds1-3, batch1, one CPU thread.4096 prefix+32 teacher-forced updates,33 full distributions. Neural/constant/by_order; one warmup,3 randomized repetitions seed320032. Include cache build/query/mix, prefill/decode. No overlapping task-owned heavy correctness audit.',
        'limits': 'No frontier parity, commercial-token conversion, service throughput/cost, longer-context generalization, or equal neural training-FLOPs claim.',
        'cpu_limit_seconds': 240, 'timing_cpu_limit_seconds': 200,
        'torch': torch.__version__, 'platform': platform.platform(),
        'source_sha256': {f.name: sha(f) for f in P.glob('*.py') if not f.name.startswith('independent')},
        'inputs_sha256': {f: sha(P / f) for f in ('data_manifest.json', 'training_data_manifest.json', 'checkpoint_manifest.json', 'items.json', 'provenance.json')}
    })

def select():
    verify()
    start = time.process_time()
    items = data('validation')
    q, order, count = copy_scores(items)
    selection = {}
    arrays = {'copy_probability': q, 'order': order, 'count': count}
    for arm in ARMS:
        neural = np.stack([score(load(arm, seed), arm, items) for seed in SEEDS])
        arrays[arm + '_neural'] = neural
        losses = [float(mixture_loss(neural, q, weight * (order > 0)).mean()) for weight in GRID]
        constant = float(GRID[np.argmin(losses)])
        by_order, grid_losses = {}, {}
        for key in (4, 8, 16):
            mask = order == key
            if mask.any():
                vals = [float(mixture_loss(neural[:, mask], q[mask], weight).mean()) for weight in GRID]
                by_order[str(key)] = float(GRID[np.argmin(vals)])
                grid_losses[str(key)] = vals
            else:
                by_order[str(key)] = constant
        selection[arm] = dict(constant=constant, by_order=by_order, constant_grid_bpb=losses, order_grid_bpb=grid_losses)
        print(arm, selection[arm], flush=True)
    np.savez_compressed(P / 'validation_arrays.npz', **arrays)
    write('selection.json', dict(arms=selection, grid=GRID.tolist(), counts_by_order={str(k): int((order == k).sum()) for k in (0,4,8,16)}, cpu_seconds=time.process_time() - start, evaluated_test=False, protocol_sha256=sha(P / 'protocol.json')))

def evaluate():
    verify()
    selection = json.loads((P / 'selection.json').read_text())
    assert selection['protocol_sha256'] == sha(P / 'protocol.json') and selection['evaluated_test'] is False
    start = time.process_time()
    items = data('evaluation')
    q, order, count = copy_scores(items)
    arrays = {'copy_probability': q, 'order': order, 'count': count}
    for arm in ARMS:
        for seed in SEEDS:
            neural = score(load(arm, seed), arm, items)
            for variant in VARIANTS:
                arrays[f'{arm}_seed{seed}_{variant}'] = mixture_loss(neural, q, weights(variant, arm, order, selection['arms']))
            print(arm, seed, float(neural.mean()), flush=True)
    np.savez_compressed(P / 'losses.npz', **arrays)
    write('results.json', dict(complete=True, selection_sha256=sha(P / 'selection.json'), cpu_seconds=time.process_time()-start, bpb={k: float(v.mean()) for k,v in arrays.items() if '_seed' in k}, copy_only_bpb=float(-np.log2(q).mean()), counts_by_order={str(k): int((order==k).sum()) for k in (0,4,8,16)}))

if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_CPU, (240, 245))
    {'--freeze': freeze, '--select': select, '--evaluate': evaluate}[sys.argv[1]]()
