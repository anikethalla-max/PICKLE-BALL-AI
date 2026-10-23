"""Exercise a new upload at a different resolution/FPS without any GT labels."""
import json
from pathlib import Path
import cv2
from dashboard_backend import ROOT,save_upload,start_job

folder=ROOT/'outputs/dashboard_custom_check'
folder.mkdir(exist_ok=True)
path=folder/'new_clip.mp4'
writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,720))
for original in sorted((ROOT/'data/occlusion_0633/frames').glob('*.jpg'))[:6]:
    writer.write(cv2.resize(cv2.imread(str(original)),(1280,720)))
writer.release()
clip=save_upload(path.read_bytes())
assert not clip['prepared'] and clip['frames']==6 and clip['fps']==15
corners=json.loads((ROOT/'reference/court_393s.json').read_text())['corners']
seeds=json.loads((ROOT/'reference/seeds_393s.json').read_text())
scale=lambda points:[[round(x*2/3),round(y*2/3)] for x,y in points]
job=start_job(clip,scale(corners),scale(seeds),'full')
(folder/'job.json').write_text(json.dumps(dict(job=job,expected_frames=6,expected_fps=15,
    expected_width=1280,expected_height=720,metrics_expected=False),indent=2))
print('New-upload full-pipeline job:',job)
