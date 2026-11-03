import json,statistics,hashlib
from pathlib import Path
p=Path(__file__).parent;r=json.loads((p/'results.json').read_text());lines=['# v10 completed result','',f"Training tokens per run: {r['protocol']['total_training_tokens']:,}. Learning rates selected by validation CE: {r['selection']}",'','Chance is6.25%. Accuracy percentages below are mean [min,max] across3 initialization seeds on the same256 held-out cases per cell.','']
summary={}
for arch,runs in r['final'].items():
 lines.append('## '+arch); summary[arch]={}
 for cell in runs[0]['test']:
  v=[a['test'][cell]['accuracy']*100 for a in runs]; ce=[a['test'][cell]['ce'] for a in runs];summary[arch][cell]={'mean_accuracy_percent':statistics.mean(v),'minimum':min(v),'maximum':max(v),'mean_ce':statistics.mean(ce)};lines.append(f"- {cell}: {statistics.mean(v):.2f}% [{min(v):.2f}, {max(v):.2f}], CE{statistics.mean(ce):.4f}")
checks=[]
for section in ['tuning','final']:
 for arch,runs in r[section].items():
  for a in runs:
   c=a['checkpoint'];path=Path(c['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==c['file_sha256'];assert c['reload_max_logit_difference']==0;checks.append(c)
lines+=['','All12 checkpoints saved, exact reload parity, file SHA256 reverified after completion.','', 'Both architectures failed to learn robust retrieval at this budget. These results do not rank memory capacity or support an architectural advantage. No-position architecture avoids the earlier absolute-position failure mode, but does not itself solve optimization.']
(p/'SUMMARY.md').write_text('\n'.join(lines));(p/'summary.json').write_text(json.dumps(summary,indent=2));(p/'checkpoint_manifest.json').write_text(json.dumps(checks,indent=2));print('\n'.join(lines))
