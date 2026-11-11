"""Post-protocol coverage extension, specified before opening primary test scores.

Original selector takes the first 512 of 1024 candidates and thus only covers
the first half of each official held-out file. This secondary evaluation scores
all admissible candidates across the full official test file without retraining.
"""
import json
import numpy as np
import torch
from models import NAT
from run import HERE, corpus, digest, evaluate

def main():
    assert (HERE/'results.json').exists()
    b=corpus('test');train=corpus('train')[:2_000_000]
    rows=[]; seen=set(); rejected=0; starts=[]
    for st in np.linspace(0,len(b)-48,1024,dtype=np.int64):
        w=b[st:st+48]
        if w in train or w in seen:rejected+=1;continue
        rows.append(list(w));seen.add(w);starts.append(int(st))
    a=torch.tensor(rows,dtype=torch.long);sets={'text':(a[:,:32],a[:,32:])}
    out={'purpose':'Post-protocol secondary full-file coverage extension, no retuning/retraining. Original first512 cohort remains primary.','examples':len(a),'rejected_exact48':rejected,'selection_sha256':digest(a.numpy().tobytes()),'first_start':starts[0],'last_start':starts[-1],'file_bytes':len(b),'target16_matches_used_train':sum(bytes(r[32:]) in train for r in rows),'runs':[]}
    for seed in [1,2,3]:
        for kind in ['transformer','mamba']:
            for budget in ['steps','time']:
                suffix='' if budget=='steps' else '_time'
                model=NAT(kind);model.load_state_dict(torch.load(HERE/'checkpoints'/f'final_{kind}_{seed}{suffix}.pt',weights_only=True))
                metrics=evaluate(model,sets)['text']
                out['runs'].append({'kind':kind,'seed':seed,'budget':budget,'metrics':metrics})
                print(kind,seed,budget,metrics['bpb'],flush=True)
    (HERE/'whole_split_evaluation.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
