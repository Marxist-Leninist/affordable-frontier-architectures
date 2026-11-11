"""Stdlib-only reporting: finalized JSON inputs, never models or dataset text."""
import argparse,csv,hashlib,html,json,math,statistics
from collections import Counter
from pathlib import Path
PHASES={'all101':list(range(101)),'selected':[0,1,2,4,8,32,100]}
KINDS=('split','whole');SEEDS=(1,2,3);DOMAINS=('test','fineweb')
STATUSES={'completed','capped','nonfinite_logits','nonfinite_loss','missing_resource_cap','missing_checkpoint','not_run','missing_cell','missing_file','malformed_cell'}
def hash_valid(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
METRICS=('suffix_bpb','byte_accuracy','exact_32_byte_match','forward_cpu_seconds','forward_wall_seconds')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def finite(v):return isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v)
def csvout(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def collect(src):
    status=read(src/'status.json');manifest=read(src/'manifest.json');selection=read(src/'cohort_selection.json')
    if {(r['kind'],r['seed']) for r in status['runs']}!={(k,s) for k in KINDS for s in SEEDS} or len(status['runs'])!=6:raise ValueError('Require finalized six-checkpoint status')
    n=selection.get('windows_per_domain')
    if selection.get('status')!='selected' or n not in (8,16,32) or manifest['selection']['windows_per_domain']!=n:raise ValueError('Invalid runtime cohort selection')
    for phase in PHASES:
        for domain in DOMAINS:
            c=manifest['cohorts'][phase][domain]
            if c['windows']!=(n if phase=='all101' else 128) or not hash_valid(c['selection_sha256']):raise ValueError('Invalid cohort manifest')
    rows=[];hashes={p.name:sha(p) for p in (src/'status.json',src/'manifest.json',src/'cohort_selection.json')};ckhash={}
    for kind in KINDS:
        for seed in SEEDS:
            p=src/f'{kind}_s{seed}.json';lookup={}
            if p.exists():
                obj=read(p);hashes[p.name]=sha(p)
                if obj['kind']!=kind or obj['seed']!=seed or not hash_valid(obj['checkpoint_sha256']) or obj['checkpoint_sha256']!=selection['checkpoint_hashes'][f'{kind}_{seed}']:raise ValueError('Checkpoint identity mismatch')
                ckhash[f'{kind}_{seed}']=obj['checkpoint_sha256']
                for raw in obj['rows']:
                    key=(raw['phase'],raw['loops'])
                    if key in lookup or key[0] not in PHASES or key[1] not in PHASES[key[0]]:raise ValueError('Duplicate/unexpected row')
                    if raw['trained_loops']!=2 or raw['applied_layers']!=16+20*raw['loops'] or raw.get('recurrent_applications')!=20*raw['loops']:raise ValueError('Invalid loop provenance')
                    if raw.get('unique_parameters')!=764304 or raw.get('invoked_module_parameters')!=(353424 if raw['loops']==0 else 764304):raise ValueError('Invalid outer-stack parameter provenance')
                    lookup[key]=raw
            for phase,counts in PHASES.items():
                for loops in counts:
                    for domain in DOMAINS:
                        v=lookup.get((phase,loops),{}).get('datasets',{}).get(domain,{'status':'missing_cell' if p.exists() else 'missing_file'})
                        stat=v.get('status','malformed_cell');stat=stat if stat in STATUSES else 'unrecognized_status';expected=n if phase=='all101' else 128
                        if stat=='completed' and not (all(finite(v.get(m)) and v[m]>=0 for m in METRICS) and all(v[m]<=1 for m in ('byte_accuracy','exact_32_byte_match')) and v.get('evaluated_windows')==expected):stat='invalid_completed_metrics'
                        row=dict(phase=phase,domain=domain,kind=kind,seed=seed,loops=loops,trained_loops=2,loop_distribution='trained_count' if loops==2 else 'extrapolation',unique_parameters=764304,invoked_module_parameters=353424 if loops==0 else 764304,applied_layers=16+20*loops,recurrent_applications=20*loops,cohort_windows=expected,cohort_selection_sha256=manifest['cohorts'][phase][domain]['selection_sha256'],status=stat,evaluated_windows=v.get('evaluated_windows',0) if isinstance(v.get('evaluated_windows',0),int) and not isinstance(v.get('evaluated_windows',0),bool) else 0)
                        for m in METRICS:row[m]=v[m] if stat=='completed' else None
                        rows.append(row)
    return rows,n,dict(raw_file_sha256=hashes,checkpoint_sha256=ckhash,report_generator_sha256=sha(Path(__file__)),report_template_sha256=sha(Path(__file__).parent/'REPORT_TEMPLATE.md'))
def aggregate(rows):
    result=[]
    for phase,counts in PHASES.items():
        for domain in DOMAINS:
            for kind in KINDS:
                for loops in counts:
                    group=[r for r in rows if (r['phase'],r['domain'],r['kind'],r['loops'])==(phase,domain,kind,loops)];good=[r for r in group if r['status']=='completed']
                    item=dict(phase=phase,domain=domain,kind=kind,loops=loops,completed_seeds=len(good),expected_seeds=3,complete=len(good)==3,status_counts=json.dumps(dict(Counter(r['status'] for r in group)),sort_keys=True))
                    for m in METRICS:
                        vals=[r[m] for r in good];item[m+'_mean']=statistics.mean(vals) if vals else None;item[m+'_seed_sd']=statistics.stdev(vals) if len(vals)>1 else None
                    result.append(item)
    return result

def plot(path,rows,phase,domain,metric):
    data=[r for r in rows if r['phase']==phase and r['domain']==domain];vals=[r[metric] for r in data if r['status']=='completed'];lo=min(vals) if vals else 0;hi=max(vals) if vals else 1;spread=hi-lo or 1;lo=max(0,lo-.08*spread);hi+=.08*spread
    x=lambda r:90+800*r/100;y=lambda v:400-290*(v-lo)/(hi-lo)
    title=f'{domain} / {phase}: '+('Suffix bits per byte' if metric=='suffix_bpb' else 'Forward wall seconds summed over cohort')
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="540"><rect width="980" height="540" fill="white"/><g font-family="Arial,sans-serif" fill="#17212b">',f'<text x="35" y="32" font-size="20">{html.escape(title)}</text>','<text x="35" y="57" font-size="13">Exploratory sweep; trained r=2. Gaps indicate missing/invalid cells. All seeds shown.</text>']
    for i in range(6):
        v=lo+(hi-lo)*i/5;yy=y(v);parts.extend([f'<path d="M90 {yy}H890" stroke="#e4e7eb"/>',f'<text x="80" y="{yy+4}" text-anchor="end" font-size="12">{v:.3g}</text>'])
    for r in (0,20,40,60,80,100):parts.append(f'<text x="{x(r)}" y="423" text-anchor="middle" font-size="12">{r}</text>')
    parts.extend([f'<path d="M{x(2)} 95V400" stroke="#777" stroke-dasharray="3 4"/>','<text x="490" y="448" text-anchor="middle" font-size="14">Inference repetition count r (16 outer + 20r recurrent layer applications)</text>'])
    for ki,kind in enumerate(KINDS):
        color=('#1167b1','#bc4e18')[ki]
        for seed in SEEDS:
            series=sorted((r for r in data if r['kind']==kind and r['seed']==seed),key=lambda r:r['loops']);segments=[];segment=[];dash=('', '6 3','2 3')[seed-1]
            for r in series:
                if r['status']=='completed':segment.append((x(r['loops']),y(r[metric])))
                elif segment:segments.append(segment);segment=[]
            if segment:segments.append(segment)
            for seg in segments:
                points=' '.join(f'{xx:.2f},{yy:.2f}' for xx,yy in seg);parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6" stroke-dasharray="{dash}"/>')
                for xx,yy in seg:parts.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="1.5" fill="{color}"/>')
            lx=45+ki*430+(seed-1)*130;parts.extend([f'<path d="M{lx} 480h25" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>',f'<text x="{lx+30}" y="484" font-size="12">{kind} seed{seed}</text>'])
    note='Per-seed traces; means and seed SD in CSV/JSON. No best count selected or validated.' if metric=='suffix_bpb' else 'Cohort aggregate runtime, not request latency or equal FLOPs across counts.'
    parts.append(f'<text x="35" y="520" font-size="12">{note}</text></g></svg>');path.write_text('\n'.join(parts))

def generate(src,out):
    rows,n,provenance=collect(src);summary=aggregate(rows);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('cells',rows),('summary',summary)]:dump(out/(name+'.json'),value);csvout(out/(name+'.csv'),value)
    files=[]
    for phase in PHASES:
        for domain in DOMAINS:
            for metric in ('suffix_bpb','forward_wall_seconds'):
                name=f'{phase}_{domain}_{metric}.svg';plot(out/name,rows,phase,domain,metric);files.append(name)
    counts=dict(Counter(r['status'] for r in rows));dump(out/'provenance.json',provenance)
    template=(Path(__file__).parent/'REPORT_TEMPLATE.md').read_text()
    report=template.replace('{{WINDOWS}}',str(n)).replace('{{CELLS}}',str(len(rows))).replace('{{STATUSES}}',json.dumps(counts,sort_keys=True)).replace('{{PLOTS}}','\n'.join(f'- [{f}]({f})' for f in files))
    (out/'REPORT.md').write_text(report);dump(out/'report_status.json',dict(status='completed',planned_cells=len(rows),cell_status_counts=counts,publication_performed=False))
    return rows,summary
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();generate(a.input,a.output)
