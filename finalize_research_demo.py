"""Verify artifacts, record review provenance and package the reference draft."""
import csv
import hashlib
import importlib.metadata
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
import cv2
import numpy as np

root=Path('annotations/occlusion_0633')
images=list((root/'annotated_frames').glob('*.jpg'))
rows=list(csv.reader((root/'reference_draft_mot.txt').open()))
assert len(images)==105 and len(rows)==420
assert len({(int(r[0]),int(r[1])) for r in rows})==420
for frame in range(1,106):
    assert sorted(int(r[1]) for r in rows if int(r[0])==frame)==[1,2,3,4]
for r in rows:
    x,y,w,h=map(float,r[2:6])
    assert 0<=x<x+w<=1920 and 0<=y<y+h<=1080
assert len(ET.parse(root/'cvat_reference_draft.xml').findall('.//track/box'))==420
review=dict(reviewer='Codex assistant, not an independent human annotator',
    method='Inspected raw keyframe sheets and all 105 annotated frames in seven contact sheets; full-resolution spot checks',
    checked_local_frame_ranges_1based=[[1,15],[16,30],[31,45],[46,60],[61,75],[76,90],[91,105]],
    corrections=['Expanded P2 body box at local frame 16 to include the right foot',
                 'Expanded P2 final keyframe body boundary'],
    acceptance='Approximate draft identity continuity and body coverage; not pixel-perfect annotation certification',
    human_verification='pending')
(root/'review_log.json').write_text(json.dumps(review,indent=2))
labels=json.loads((root/'annotations.json').read_text())
labels['review_status']='All 105 frames assistant-reviewed via contact sheets; independent human verification pending'
(root/'annotations.json').write_text(json.dumps(labels,indent=2))
(root/'README.md').write_text('# Annotation draft\n\n105 annotated images and 420 player boxes. Published method reference: SportsMOT (ICCV 2023).\n\nRead ../../ANNOTATION_PROTOCOL.md for the source, adaptations and generation procedure. Labels were drawn by the assistant on raw keyframes and interpolated. Independent human verification is pending. CVAT XML uses local zero-based frames; MOT uses local one-based frames.\n',encoding='utf-8')
outputs=Path('outputs/comparison_0633')
experiment=json.loads((outputs/'experiment.json').read_text())
verified=[]
for method in experiment['methods']:
    p=outputs/method/'tracking.mp4'
    cap=cv2.VideoCapture(str(p))
    assert cap.isOpened() and round(cap.get(cv2.CAP_PROP_FRAME_COUNT))==105
    cap.set(cv2.CAP_PROP_POS_FRAMES,104)
    ok,last=cap.read()
    assert ok and last.shape[:2]==(1080,1920)
    cap.release()
    verified.append(str(p))
cap=cv2.VideoCapture(str(outputs/'four_method_comparison.mp4'))
assert cap.isOpened() and round(cap.get(cv2.CAP_PROP_FRAME_COUNT))==105
cap.set(cv2.CAP_PROP_POS_FRAMES,104)
assert cap.read()[0]
cap.release()
audit=json.loads((outputs/'adaptive_osnet'/'association_audit.json').read_text())
weights=np.array([row['weights'] for row in audit])
assert np.allclose(weights.sum(axis=1),1)
assert np.ptp(weights[:,1])>.01
hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
    ['data/occlusion_0633/shared_detections.json','data/occlusion_0633/osnet_features.npz',
     'annotation_keyframes.json','annotations/occlusion_0633/reference_draft_mot.txt']}
report=dict(annotated_frames=105,reference_boxes=420,validated_videos=verified,
    checked='video dimensions/count/final-frame decode; label identities and bounds; CVAT box count; adaptive weights normalized and variable',
    sha256=hashes,versions={name:importlib.metadata.version(name) for name in
    ['ultralytics','torch','torchvision','numpy','scipy','opencv-python','trackeval','filterpy','lap','gdown']})
(outputs/'verification.json').write_text(json.dumps(report,indent=2))
with zipfile.ZipFile(root/'annotation_draft_package.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in root.rglob('*'):
        if path.is_file() and path.suffix!='.zip' and 'review_sheets' not in path.parts:
            z.write(path,path.relative_to(root))
    z.write('ANNOTATION_PROTOCOL.md','ANNOTATION_PROTOCOL.md')
print(json.dumps(report,indent=2))
