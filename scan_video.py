"""Create timestamped video contact sheets for manual event selection."""
from pathlib import Path
import cv2
import numpy as np

out = Path('outputs/occlusion_scan')
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture('Videos/InPlayBalls_54min_vid.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
tiles = []
sheet = 0
for second in range(0, int(duration), 2):
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(second*fps))
    ok, frame = cap.read()
    if not ok:
        continue
    tile = cv2.resize(frame, (384,216))
    cv2.rectangle(tile,(0,0),(160,25),(0,0,0),-1)
    cv2.putText(tile,f'{second//60:02d}:{second%60:02d}',(6,19),0,.65,(255,255,255),2)
    tiles.append(tile)
    if len(tiles) == 25 or second+2 >= int(duration):
        while len(tiles) < 25:
            tiles.append(np.zeros((216,384,3),np.uint8))
        cv2.imwrite(str(out/f'sheet_{sheet:02d}.jpg'), np.vstack([np.hstack(tiles[r:r+5]) for r in range(0,25,5)]))
        tiles=[]
        sheet += 1
cap.release()
print(f'Saved {sheet} contact sheets; duration {duration:.2f} seconds')
