from pathlib import Path
import os,hashlib,json
P=Path(__file__).resolve().parent
raw=(Path(os.environ['ANTI_DATA_DIR'])/'train.txt').read_bytes();man=json.loads((P/'source_data_manifest.json').read_text());assert hashlib.sha256(raw).hexdigest()==man['splits']['train']['text_sha256'];seen=set();clean=[]
for line in raw.splitlines(keepends=True):
 k=line.strip()
 if k and k in seen:continue
 if k:seen.add(k)
 clean.append(line)
train=b''.join(clean)[:2000000];source=json.loads((P/'rows.json').read_text());selected={x['row'] for x in json.loads((P/'data_manifest.json').read_text())['documents']};targets=[r['row']['text'].encode()[385:513] for r in source['rows'] if r['row_idx'] in selected];hits=[r['row_idx'] for r in source['rows'] if r['row_idx'] in selected and r['row']['text'].encode()[385:513] in train];assert not hits and len(set(targets))==len(targets)
r={'actual_training_sha256':hashlib.sha256(train).hexdigest(),'actual_training_bytes':len(train),'actual_training_target_overlap_rows':hits,'unique_current_target_spans':len(set(targets)),'current_documents':len(targets),'shorter_and_near_duplicate_contamination':'not measured'}
expected=P/'provenance_checks.json'
if expected.exists():assert json.loads(expected.read_text())==r
else:expected.write_text(json.dumps(r,indent=2))
print('Actual deduplicated training stream and current-target checks pass')
