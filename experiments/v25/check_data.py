import json,hashlib,urllib.request
from pathlib import Path
P=Path(__file__).parent
out={}
for name,url in {'fineweb':'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/raw/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md','wikitext':'https://huggingface.co/datasets/Salesforce/wikitext/raw/b08601e04326c79dfdd32d625aee71d232d685c3/README.md'}.items():
 try:
  raw=urllib.request.urlopen(url,timeout=30).read(200001);assert len(raw)<=200000;(P/(name+'_card.md')).write_bytes(raw);out[name]={'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'status':'retrieved'}
 except Exception as e:out[name]={'url':url,'error':repr(e),'status':'inherited prior verification only'}
fw=json.loads((P/'data_manifest.json').read_text());prior=set()
for n in range(15,25):
 p=P.parent/f'frontier_research_v{n}'/'data_manifest.json'
 if p.exists():
  for d in json.loads(p.read_text()).get('documents',[]):
   if 'full_text_sha256' in d:prior.add(d['full_text_sha256'])
train=(P.parent/'frontier_research_v8/train.txt').read_bytes();rows=json.loads((P/'rows.json').read_text())['rows']
full=[r['row']['text'].encode() for r in rows]
out['contamination']={'prior_exact_full_document_matches':[d['row_index'] for d in fw['documents'] if d['full_text_sha256'] in prior],'within_cohort_exact_full_duplicates':len(full)-len(set(full)),'full_documents_occurring_in_wikitext_train':[rows[i]['row_idx'] for i,d in enumerate(full) if d in train],'limitations':'No near duplicate or exhaustive passage-overlap detection; absence of exact full-document matches does not rule out contamination. Previous FineWeb rows disjoint.'}
(P/'data_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
