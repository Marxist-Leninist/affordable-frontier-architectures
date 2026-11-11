"""Paired descriptive summary; refuses unequal initial/data/evaluation identities."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parent
rows=[]
for seed in (1,2,3):
    dirs=[ROOT/'runs'/f'{kind}_s{seed}' for kind in ('split','whole')]
    if not all((d/'training.json').exists() and (d/'evaluation.json').exists() for d in dirs):
        rows.append({'seed':seed,'status':'missing_run_or_evaluation'});continue
    training=[json.loads((d/'training.json').read_text()) for d in dirs]
    assert training[0]['initial_hash']==training[1]['initial_hash'],'unequal initial weights'
    ev=[{x['tag']:x for x in json.loads((d/'evaluation.json').read_text())} for d in dirs]
    for tag in ('matched_step','cpu_300','cpu_900'):
        if not all(tag in e for e in ev):
            rows.append({'seed':seed,'tag':tag,'status':'missing_checkpoint'});continue
        a,b=(e[tag] for e in ev)
        if tag=='matched_step':assert a['batch_sha256']==b['batch_sha256'],'unequal matched-step data'
        for dataset in ('validation','test','fineweb'):
            da,db=a['datasets'][dataset],b['datasets'][dataset]
            assert da['manifest']['selection_sha256']==db['manifest']['selection_sha256'],'unequal evaluation windows'
            rows.append(dict(seed=seed,tag=tag,dataset=dataset,status='paired',split_bpb=da['metrics']['bpb'],whole_bpb=db['metrics']['bpb'],difference_split_minus_whole_bpb=da['metrics']['bpb']-db['metrics']['bpb'],split_step=a['step'],whole_step=b['step'],split_cpu=a['cpu'],whole_cpu=b['cpu']))
summary=[]
for tag in ('matched_step','cpu_300','cpu_900'):
    for dataset in ('validation','test','fineweb'):
        pairs=[r for r in rows if r.get('tag')==tag and r.get('dataset')==dataset and r['status']=='paired']
        vals=[r['difference_split_minus_whole_bpb'] for r in pairs]
        summary.append(dict(tag=tag,dataset=dataset,paired_seeds=len(vals),mean_difference=statistics.mean(vals) if vals else None,seed_difference_sd=statistics.stdev(vals) if len(vals)>1 else None,complete=len(vals)==3))
out={'interpretation':'Negative difference favors split. Three-seed descriptive result only; no Dory replication/scaling claim. Missing runs remain missing.','pairs':rows,'summary':summary}
(ROOT/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
