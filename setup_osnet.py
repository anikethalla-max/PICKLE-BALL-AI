"""Official Torchreid OSNet-x0.25 MSMT17 checkpoint and source."""
import hashlib
import json
import urllib.request
from pathlib import Path
import gdown

root=Path('third_party/osnet')
root.mkdir(parents=True,exist_ok=True)
(root/'__init__.py').touch()
sources={}
for name in ['torchreid/models/osnet.py','LICENSE']:
    url=f'https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/{name}'
    with urllib.request.urlopen(url,timeout=45) as response:
        content=response.read()
    target=root/Path(name).name
    target.write_bytes(content)
    sources[target.name]={'url':url,'sha256':hashlib.sha256(content).hexdigest()}
weights=Path('models/osnet_x0_25_msmt17.pth')
weights.parent.mkdir(exist_ok=True)
url='https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF'
if not weights.exists():
    gdown.download(url,str(weights),quiet=False)
sources['weights']={'url':url,'sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
    'reference':'https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md',
    'training_dataset':'MSMT17','architecture':'osnet_x0_25'}
(root/'PROVENANCE.json').write_text(json.dumps(sources,indent=2))
print('Official OSNet checkpoint ready')
