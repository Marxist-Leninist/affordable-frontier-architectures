"""Read-only independent reconstruction of benchmark data selection."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; D=R.parent/'frontier_research_v8'
manifest=json.loads((D/'data_manifest.json').read_text())
raw={s:(D/(s+'.txt')).read_bytes() for s in ['train','validation','test']}
used=raw['train'][:2_000_000]
hashes={s:{'actual':hashlib.sha256(b).hexdigest(),'expected':manifest['splits'][s]['text_sha256'],'matches':hashlib.sha256(b).hexdigest()==manifest['splits'][s]['text_sha256'],'bytes':len(b)} for s,b in raw.items()}
def select(b):
    seen=set();wins=[];reject=0
    for st in np.linspace(0,len(b)-48,1024,dtype=np.int64):
        w=b[st:st+48]
        if w in used or w in seen:reject+=1;continue
        seen.add(w);wins.append(w)
        if len(wins)==512:break
    assert len(wins)==512
    return wins,reject
v,vr=select(raw['validation']);t,tr=select(raw['test'])
def winreport(w,r):
    return {'n':len(w),'rejected':r,'selection_sha256':hashlib.sha256(np.array([list(x) for x in w],dtype=np.int64).tobytes()).hexdigest(),'exact48_in_used_train':sum(x in used for x in w),'duplicate48':len(w)-len(set(w)),'target16_in_used_train':sum(x[32:] in used for x in w),'duplicate_target16':len(w)-len(set(x[32:] for x in w))}
# Reproduce every RNG draw used by train: starts, digits, then task identifiers.
def train_digits(seed,steps):
    rng=np.random.default_rng(100000+seed); seq=[]; task_pairs=[]
    for _ in range(steps):
        rng.integers(0,len(used)-48,8)
        digits=rng.integers(0,10,(8,16),dtype=np.int64)+48
        tasks=rng.integers(0,3,8)
        seq.extend(bytes(row) for row in digits.astype(np.uint8))
        task_pairs.extend((int(task),bytes(row)) for task,row in zip(tasks,digits.astype(np.uint8)))
    return seq,task_pairs
def eval_digits(base):
    seq=[];pairs=[]
    for task in range(3):
        d=np.random.default_rng(base+task).integers(0,10,(256,16),dtype=np.int64)+48
        seq.extend(bytes(row) for row in d.astype(np.uint8));pairs.extend((task,bytes(row)) for row in d.astype(np.uint8))
    return seq,pairs
sets={f'train_seed{s}':train_digits(s,300 if s==101 else 800) for s in [101,1,2,3]}
sets['validation']=eval_digits(22000);sets['test']=eval_digits(33000)
within={k:{'presentations':len(x),'unique_digit_sequences':len(set(x)),'duplicate_digit_sequences':len(x)-len(set(x)),'duplicate_task_input_pairs':len(p)-len(set(p))} for k,(x,p) in sets.items()}
inter=[]
for i,a in enumerate(sets):
    for b in list(sets)[i+1:]:
        inter.append({'a':a,'b':b,'shared_digit_sequences':len(set(sets[a][0])&set(sets[b][0])),'shared_task_input_pairs':len(set(sets[a][1])&set(sets[b][1]))})
report={'hashes':hashes,'text':{'validation':winreport(v,vr),'test':winreport(t,tr),'validation_test_shared48':len(set(v)&set(t)),'validation_test_shared_target16':len(set(w[32:] for w in v)&set(w[32:] for w in t))},'toy':{'within':within,'between':inter},'caveats':['Seed101 tuning batches deliberately reused across both learning rates and architectures; each final seed deliberately reused across architectures. Audit deduplicates these planned paired repetitions.','Digit sequences are independent input keys; task-conditioned outputs can coincide even for different inputs, especially sorted digits.','Target16 matches and shorter natural-language overlap are transparency metrics, not contamination exclusions.','Full48 exclusion covers the exact 2 MB actually sampled for training, not the unused remainder of the official training split.','Zero candidate rejections means selected starts cover only the first approximately half of each evaluation file, not the full official split.','No raw data or trainer code was changed. This script uses NumPy only, no model execution or benchmark timing.']}
(R/'data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
