"""Package the visually selected 06:33.0–06:36.5 event for annotation."""
import csv
import json
import zipfile
from pathlib import Path

import cv2

out = Path('data/occlusion_0633')
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture('Videos/InPlayBalls_54min_vid.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
first, count = round(393*fps), round(3.5*fps)
cap.set(cv2.CAP_PROP_POS_FRAMES,first)
writer = cv2.VideoWriter(str(out/'input_clip.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080))
if not writer.isOpened():
    raise RuntimeError('Cannot create input clip')
try:
    with (out/'frame_mapping.csv').open('w',newline='') as file:
        rows=csv.writer(file)
        rows.writerow(['evaluation_frame_1based','source_frame_0based','source_time_seconds','filename'])
        for i in range(count):
            ok,frame=cap.read()
            if not ok:
                raise RuntimeError('Video ended before selected segment')
            writer.write(frame)
            rows.writerow([i+1,first+i,(first+i)/fps,f'frame_{first+i:08d}.jpg'])
            if i in (0,24,45):
                stage={0:'before_0633_000',24:'overlap_0633_800',45:'after_0634_500'}[i]
                cv2.imwrite(str(out/f'{stage}.jpg'),frame)
finally:
    cap.release()
    writer.release()
with zipfile.ZipFile(out/'frames_for_annotation.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for frame in sorted((out/'frames').glob('*.jpg')):
        archive.write(frame,frame.name)
manifest=dict(start='06:33.000',end_exclusive='06:36.500',fps=fps,frames=count,
              source_frame_range_0based=[first,first+count-1],
              event='Near-side black-clothed player crosses in front of near-side red-clothed player',
              approximate_overlap='06:33.700–06:34.100',
              ground_truth_status='Not annotated; MOTA, AssA and IDSW unavailable',
              identities={'P1':'Near-side red top, red cap', 'P2':'Near-side black outfit',
                          'P3':'Far-side red top', 'P4':'Far-side grey outfit, white cap'})
(out/'event.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
