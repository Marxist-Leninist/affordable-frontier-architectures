"""Verify and unpack archived text-only research artifacts beside this directory."""
from pathlib import Path
import argparse,gzip,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parent
for item in json.loads((root/'manifest.json').read_text()):
 data=b''.join((root/part).read_bytes() for part in item['parts'])
 assert hashlib.sha256(data).hexdigest()==item['sha256'],'Archive SHA256 mismatch'
 files=json.loads(gzip.decompress(data));dest=a.output/f"frontier_research_v{item['version']}"
 for name,text in files.items():
  rel=Path(name)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe path')
  f=dest/rel
  if f.exists() and f.read_text()!=text:raise FileExistsError(f'Would overwrite differing file: {f}')
  f.parent.mkdir(parents=True,exist_ok=True);f.write_text(text)
 print(item['version'],len(files),'verified text files')
