"""Fetch pinned official OC-SORT sources (with provenance and license)."""
import hashlib
import json
import urllib.request
from pathlib import Path

repo = 'noahcao/OC_SORT'
try:
    request = urllib.request.Request(f'https://api.github.com/repos/{repo}/commits/master',headers={'User-Agent':'SmartDoublesAI'})
    with urllib.request.urlopen(request, timeout=40) as response:
        commit = json.load(response)['sha']
except urllib.error.HTTPError:
    # GitHub unauthenticated API may be rate-limited. File hashes still record
    # exactly which code was used; do not claim an immutable commit in that case.
    commit = 'master'
root=Path('third_party/ocsort')
root.mkdir(parents=True,exist_ok=True)
(root/'__init__.py').touch()
manifest={'repository':f'https://github.com/{repo}','commit':commit,'files':{}}
for name in ['trackers/ocsort_tracker/ocsort.py','trackers/ocsort_tracker/association.py',
             'trackers/ocsort_tracker/kalmanfilter.py','LICENSE']:
    url=f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
    with urllib.request.urlopen(url,timeout=40) as response:
        content=response.read()
    target=root/Path(name).name
    target.write_bytes(content)
    manifest['files'][target.name]={'url':url,'sha256':hashlib.sha256(content).hexdigest()}
(root/'PROVENANCE.json').write_text(json.dumps(manifest,indent=2))
print('OC-SORT downloaded at',commit)
