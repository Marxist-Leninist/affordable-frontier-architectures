"""Exact categorical speculative sampling, a CPU toy, not a trained LM."""
import random, math

def normalized(p):
    if not p or any(not math.isfinite(x) or x<0 for x in p) or sum(p)<=0:raise ValueError('invalid distribution')
    s=sum(p)
    if not math.isfinite(s):raise ValueError('probability sum overflow')
    return tuple(x/s for x in p)

def sample(p,rng):
    p=normalized(p);u=rng.random();s=0
    for i,v in enumerate(p):
        s+=v
        if u<s:return i
    return len(p)-1

def corrected_distribution(p,q):
    p,q=normalized(p),normalized(q)
    if len(p)!=len(q):raise ValueError('vocab mismatch')
    accepted=[min(a,b) for a,b in zip(p,q)]
    reject=1-sum(accepted)
    residual=[max(a-b,0.) for a,b in zip(p,q)]
    total=sum(residual)
    # When rejection probability is zero residual is never sampled.
    return tuple(a+(reject*r/total if total>1e-15 else 0.) for a,r in zip(accepted,residual))

def speculative(target,draft,n,k=4,seed=17):
    if type(n) is not int or n<0 or type(k) is not int or k<1:raise ValueError('invalid counts')
    if not target or not draft:raise ValueError('empty transition matrix')
    target=[normalized(x) for x in target];draft=[normalized(x) for x in draft]
    if len(target)!=len(draft) or len(target)!=len(target[0])+1 or any(len(x)!=len(target[0]) for x in target+draft):raise ValueError('expected vocab+1 transition rows')
    rng=random.Random(seed);state=len(target)-1;out=[]
    stats={'target_block_calls':0,'draft_proposed':0,'accepted':0,'rejections':0,'bonus':0}
    while len(out)<n:
        proposals=[];prev=state
        for _ in range(min(k,n-len(out))):
            q=draft[prev];y=sample(q,rng);proposals.append((y,q));prev=y
        stats['draft_proposed']+=len(proposals);stats['target_block_calls']+=1
        rejected=False
        for y,q in proposals:
            p=target[state]
            if rng.random()<min(1.,p[y]/q[y]):
                out.append(y);state=y;stats['accepted']+=1
            else:
                residual=[max(a-b,0.) for a,b in zip(p,q)]
                state=sample(residual,rng);out.append(state)
                stats['rejections']+=1;rejected=True;break
        if not rejected and len(out)<n:
            state=sample(target[state],rng);out.append(state);stats['bonus']+=1
    assert len(out)==n
    return out,stats

TARGET=((.10,.20,.60,.10),(.40,.20,.10,.30),(.10,.55,.25,.10),(.25,.25,.25,.25),(.20,.30,.40,.10))
DRAFT=((.50,.20,.20,.10),(.10,.60,.20,.10),(.35,.20,.20,.25),(.25,.25,.25,.25),(.65,.20,.10,.05))

def probe(n=400000,k=4,seed=17):
    out,stats=speculative(TARGET,DRAFT,n,k,seed)
    counts=[[0]*4 for _ in range(5)];state=4
    for token in out:counts[state][token]+=1;state=token
    rows=[]
    for i in range(4):
        total=sum(counts[i]);emp=[v/total for v in counts[i]]
        rows.append({'previous_token':i,'samples':total,'observed':emp,'target':TARGET[i],'total_variation':sum(abs(a-b) for a,b in zip(emp,TARGET[i]))/2})
    return {'classification':'untrained four-symbol Markov CPU correctness probe','seed':seed,'output_tokens':n,'block_length':k,'stats':stats,'transition_rows':rows,'max_row_total_variation':max(r['total_variation'] for r in rows),'exact_one_step_max_abs_error':max(abs(a-b) for p,q in zip(TARGET,DRAFT) for a,b in zip(p,corrected_distribution(p,q))),'timing_claim':'none; no GPU performance or LM quality inferred'}
