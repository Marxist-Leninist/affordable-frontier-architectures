"""Interleaved whole-model observed-byte update benchmark, not service throughput."""
from run import *

@torch.inference_mode()
def main():
    resource.setrlimit(resource.RLIMIT_CPU, (200,205))
    verify()
    cpu = time.process_time()
    selection = json.loads((P / 'selection.json').read_text())['arms']
    eligible = {item['doc'] for item in data('evaluation')}
    docs = [r for r in json.loads((P / 'rows.json').read_text())['rows'] if r['row_idx'] in eligible and len(r['row']['text'].encode()) >= 4128][:8]
    assert len(docs) == 8
    rng = np.random.default_rng(320032)
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            model = load(arm, seed)
            for row in docs:
                raw = row['row']['text'].encode()[:4128]
                tokens = torch.tensor(list(raw[:4096]))[None,:]
                updates = torch.tensor(list(raw[4096:]))[None,:]
                def go(variant):
                    def complete_distribution(logits, cache):
                        probability = logits.double().softmax(-1).numpy().reshape(256)
                        if cache is not None:
                            copy, count, order = cache.distribution()
                            if not count:
                                weight = 0
                            elif variant == 'constant':
                                weight = selection[arm]['constant']
                            else:
                                weight = selection[arm]['by_order'][str(order)]
                            probability = (1-weight)*probability + weight*copy
                        assert np.isfinite(probability).all() and abs(probability.sum()-1) < 1e-10
                        return probability
                    start = time.perf_counter()
                    cache = None if variant == 'neural' else CompactCache()
                    if cache is not None:
                        for byte in raw[:4096]:
                            cache.feed(byte)
                    logits, state = (local_prefill(model,tokens,1,prune=True) if arm=='local' else prefill(model,tokens))
                    complete_distribution(logits,cache)
                    prefill_seconds = time.perf_counter()-start
                    start = time.perf_counter()
                    for pos in range(32):
                        if cache is not None:
                            cache.feed(raw[4096+pos])
                        logits,state = model(updates[:,pos:pos+1],state,cache=True)
                        complete_distribution(logits,cache)
                    decode_seconds = time.perf_counter()-start
                    return dict(prefill_seconds=prefill_seconds,decode_seconds=decode_seconds,index_bytes=reachable_bytes(cache) if cache is not None else 0,state_bytes=state_bytes(state),weight_bytes=sum(p.numel()*p.element_size() for p in model.parameters()))
                for variant in ('neural','constant','by_order'):
                    go(variant)
                for rep in range(3):
                    order = ['neural','constant','by_order']
                    rng.shuffle(order)
                    for position,variant in enumerate(order):
                        result = go(variant)
                        result.update(arm=arm,seed=seed,doc=row['row_idx'],rep=rep,variant=variant,order_position=position)
                        rows.append(result)
    write('timing.json',dict(rows=rows,cpu_seconds=time.process_time()-cpu,torch=torch.__version__,threads=torch.get_num_threads(),batch=1,weights_dtype='float32',probabilities_dtype='float64',hardware=platform.processor()))
    print('Timing CPU seconds', time.process_time()-cpu, flush=True)
if __name__ == '__main__':
    main()
