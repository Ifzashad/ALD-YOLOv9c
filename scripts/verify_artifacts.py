"""Verify actual dataset/model bytes after clone; detects missing LFS objects."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'ARTIFACT_MANIFEST.json').read_text())
failures=[]
for n,item in enumerate(manifest['files']):
    p=ROOT/item['path']
    if not p.is_file() or p.stat().st_size!=item['bytes']:
        failures.append(item['path']);continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=item['sha256']:failures.append(item['path'])
    if n%1000==0:print('Verified',n,'/',len(manifest['files']),flush=True)
if failures:raise SystemExit('FAILED; run git lfs pull or restore files:\n'+'\n'.join(failures[:30]))
print('ALL ARTIFACT HASHES MATCH:',len(manifest['files']))
