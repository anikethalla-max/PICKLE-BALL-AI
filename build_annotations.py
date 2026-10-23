"""Render assistant-drawn keyframes into reviewable reference annotations.

Independent of detector predictions. Interpolation is explicitly documented.
This draft must not be represented as independently human-verified ground truth.
"""
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np

from prototype import COLORS

root=Path('annotations/occlusion_0633')
images=root/'annotated_frames'
review=root/'review_sheets'
images.mkdir(parents=True,exist_ok=True)
review.mkdir(parents=True,exist_ok=True)
config=json.loads(Path('annotation_keyframes.json').read_text())
keys=sorted(map(int,config['boxes']))
paths=sorted(Path('data/occlusion_0633/frames').glob('*.jpg'))
data=[]
for index,path in enumerate(paths):
    left=max(k for k in keys if k<=index)
    right=min(k for k in keys if k>=index)
    a=np.array(config['boxes'][str(left)],float)*config['coordinate_scale']
    b=np.array(config['boxes'][str(right)],float)*config['coordinate_scale']
    boxes=a if left==right else a+(b-a)*(index-left)/(right-left)
    frame=cv2.imread(str(path))
    for identity,box in enumerate(boxes,1):
        occluded=any(start<=index<=end for start,end in config['occluded_local_indices_0based'][str(identity)])
        data.append(dict(frame=index+1,source_frame=11790+index,player_id=identity,
                         box=box.round(2).tolist(),occluded=occluded,
                         keyframe=index in keys,method='assistant_visual_keyframe' if index in keys else 'linear_interpolation'))
        x1,y1,x2,y2=np.rint(box).astype(int)
        cv2.rectangle(frame,(x1,y1),(x2,y2),COLORS[identity-1],2)
        cv2.putText(frame,f'P{identity}'+(' OCCLUDED' if occluded else ''),
                    (x1,max(50,y1-8)),0,.65,COLORS[identity-1],2)
    cv2.rectangle(frame,(0,0),(1920,32),(25,25,25),-1)
    cv2.putText(frame,f'ANNOTATION DRAFT | source 06:{33+index/30:06.3f} | local {index+1}/105 | human verification pending',
                (10,23),0,.65,(255,255,255),1)
    if not cv2.imwrite(str(images/path.name),frame):
        raise IOError('Failed saving annotation')
with (root/'reference_draft_mot.txt').open('w',newline='') as file:
    writer=csv.writer(file)
    for row in data:
        x1,y1,x2,y2=row['box']
        writer.writerow([row['frame'],row['player_id'],x1,y1,round(x2-x1,2),round(y2-y1,2),1,1,-1])
(root/'annotations.json').write_text(json.dumps({'status':config['status'],
    'visibility_note':'MOT visibility=-1 means unknown; occluded is a qualitative annotation, not visible-area measurement',
    'source':'SportsMOT ICCV 2023 annotation guidelines, adapted to pickleball',
    'review_status':'keyframes visually inspected; all-frame review pending',
    'annotations':data},indent=2))
tree=ET.Element('annotations')
ET.SubElement(tree,'version').text='1.1'
meta=ET.SubElement(tree,'meta')
task=ET.SubElement(meta,'task')
ET.SubElement(task,'name').text='PickleMOT-4 occlusion 0633 — assistant reference draft'
ET.SubElement(task,'size').text=str(len(paths))
ET.SubElement(task,'mode').text='interpolation'
labels=ET.SubElement(task,'labels')
label=ET.SubElement(labels,'label')
ET.SubElement(label,'name').text='player'
ET.SubElement(label,'type').text='rectangle'
attributes=ET.SubElement(label,'attributes')
attr=ET.SubElement(attributes,'attribute')
for tag,value in [('name','player_id'),('mutable','False'),('input_type','select'),('default_value','P1'),('values','P1\nP2\nP3\nP4')]:
    ET.SubElement(attr,tag).text=value
for identity in range(1,5):
    track=ET.SubElement(tree,'track',id=str(identity-1),label='player',source='manual')
    for row in data:
        if row['player_id']!=identity:
            continue
        x1,y1,x2,y2=row['box']
        box=ET.SubElement(track,'box',frame=str(row['frame']-1),outside='0',occluded=str(int(row['occluded'])),
                          keyframe=str(int(row['keyframe'])),xtl=str(x1),ytl=str(y1),xbr=str(x2),ybr=str(y2),z_order='0')
        ET.SubElement(box,'attribute',name='player_id').text=f'P{identity}'
ET.indent(tree)
ET.ElementTree(tree).write(root/'cvat_reference_draft.xml',encoding='utf-8',xml_declaration=True)
for start in range(0,len(paths),15):
    tiles=[cv2.resize(cv2.imread(str(images/path.name)),(640,360)) for path in paths[start:start+15]]
    while len(tiles)<15:
        tiles.append(np.zeros((360,640,3),np.uint8))
    cv2.imwrite(str(review/f'frames_{start+1:03d}_{min(start+15,len(paths)):03d}.jpg'),
                np.vstack([np.hstack(tiles[r:r+3]) for r in range(0,15,3)]))
print('Created',len(paths),'annotated frames and',len(data),'reference boxes at',root)
