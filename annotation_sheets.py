"""Raw frames for independent keyframe drawing and visual review."""
from pathlib import Path
import cv2
import numpy as np

paths=sorted(Path('data/occlusion_0633/frames').glob('*.jpg'))
out=Path('outputs/annotation_review/raw')
out.mkdir(parents=True,exist_ok=True)
indices=list(range(0,105,5))+[104]
for page in range(0,len(indices),4):
    tiles=[]
    for index in indices[page:page+4]:
        frame=cv2.imread(str(paths[index]))
        cv2.putText(frame,f'LOCAL {index+1} (zero index {index})',(20,60),0,1.5,(0,255,255),3)
        tiles.append(cv2.resize(frame,(960,540)))
    while len(tiles)<4:
        tiles.append(np.zeros((540,960,3),np.uint8))
    cv2.imwrite(str(out/f'keyframes_{page//4:02d}.jpg'),np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
print('Raw review sheets saved')
