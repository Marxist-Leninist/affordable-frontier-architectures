"""Aggregate frozen results with window-bootstrap uncertainty; no tuning."""
from pathlib import Path
import json,math,hashlib
import numpy as np
P=Path(__file__).parent
def read(n):return json.loads((P/n).read_text())
def main():
    base=read('untouched.json');rows=[]
    for kind in ['normal_finetune','pretrained_diffusion']:
        for endpoint in (['equal_data'] if kind=='normal_finetune' else ['equal_data','equal_cpu']):
            rs=[read(f'{kind}_{s}_{endpoint}.json') for s in [1,2,3]]
            vals=[r['test']['nll_nats'] for r in rs]
            rows.append(dict(method=kind,endpoint=endpoint,test_mean_nll=float(np.mean(vals)),test_nll_sd_across_seeds=float(np.std(vals,ddof=1)),test_nll_by_seed=vals,steps_by_seed=[r['steps'] for r in rs],training_cpu_by_seed=[r['training_cpu_seconds'] for r in rs],training_tokens_by_seed=[r['token_exposures'] for r in rs],validation_mean_nll=float(np.mean([r['validation']['nll_nats'] for r in rs]))))
    normal=np.array([read(f'normal_finetune_{s}_equal_data.json')['test']['window_nll_nats'] for s in [1,2,3]])
    rng=np.random.default_rng(300191);draw=rng.integers(0,64,(10000,64));diffs=[]
    for endpoint in ['equal_data','equal_cpu']:
        dd=np.array([read(f'pretrained_diffusion_{s}_{endpoint}.json')['test']['window_nll_nats'] for s in [1,2,3]])-normal
        avg=dd.mean(0);ci=np.quantile(avg[draw].mean(1),[.025,.975]).tolist()
        diffs.append(dict(endpoint=endpoint,converted_minus_normal_nll=float(avg.mean()),window_bootstrap_95=ci,scope='Resamples 64 test windows conditional on 3 fixed training seeds; not uncertainty over new datasets, models, or the seed population. Finite 4-particle decoder, not implicit diffusion likelihood.'))
    out=dict(complete=True,untouched_test_nll=base['test']['nll_nats'],rows=rows,paired_differences=diffs,verdict='This small direct pretrained conversion did not preserve baseline quality. It is not a successful large-model conversion and does not refute better adaptation methods or the paper.',limits=['14M model, 1.2M trainable with frozen lexical maps; not a large-model trial','Only 6,144 training token exposures per equal-data run and 32-token context','Only 1,024 test targets; pretrained exposure to WikiText unknown','No random-initialization ablation, distillation, MAUVE or teacher generative perplexity','Four-particle decoder approximation can overstate loss of the implicit model','Architecture not function-preserving: later pretrained blocks now receive lexical and noise codes rather than original hidden states','CPU timing and saved-tensor accounting are not GPU peak memory or end-to-end cost'])
    (P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
