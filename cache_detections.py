"""One shared YOLO detection stream for all association methods."""
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

parser=argparse.ArgumentParser()
parser.add_argument('--data-root',default='data/occlusion_0633')
parser.add_argument('--court',default='reference/court_393s.json')
args=parser.parse_args()
root=Path(args.data_root)
cfg=json.loads(Path(args.court).read_text())
matrix=cv2.getPerspectiveTransform(np.float32(cfg['corners']),np.float32([[0,0],[1,0],[1,1],[0,1]]))
model=YOLO('yolo11n.pt')
frames=sorted((root/'frames').glob('*.jpg'))
metadata=json.loads((root/'clip_metadata.json').read_text()) if (root/'clip_metadata.json').exists() else {'source_frame_start':11790}
records=[]
seconds=0
for index,path in enumerate(frames):
    frame=cv2.imread(str(path))
    tick=time.perf_counter()
    result=model.predict(frame,classes=[0],conf=.10,imgsz=960,device='cpu',verbose=False)[0]
    seconds+=time.perf_counter()-tick
    boxes=result.boxes.xyxy.cpu().numpy()
    scores=result.boxes.conf.cpu().numpy()
    candidates=[]
    for box,score in zip(boxes,scores):
        foot=np.float32([[(box[0]+box[2])/2,box[3]]]).reshape(1,1,2)
        p=cv2.perspectiveTransform(foot,matrix)[0,0]
        if np.isfinite(p).all() and -.18<=p[0]<=1.18 and -.18<=p[1]<=1.18:
            candidates.append([*box.tolist(),float(score)])
    records.append({'frame':index+1,'source_frame':metadata['source_frame_start']+index,'detections':candidates})
    if (index+1)%30==0:
        print('Detected',index+1,flush=True)
(root/'shared_detections.json').write_text(json.dumps({'model':'yolo11n.pt','confidence_floor':.10,
    'imgsz':960,'roi':'same expanded court polygon for every method','detector_seconds':seconds,
    'frames':records},indent=2))
print('Cached',len(records),'frames')
