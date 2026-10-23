"""Independent detector + greedy motion draft labels, never verified ground truth."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def generate(data, court, output):
    torch.set_num_threads(4)
    data, output = Path(data), Path(output)
    marked = output/'annotated_frames'
    marked.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(court).read_text())
    matrix = cv2.getPerspectiveTransform(np.float32(cfg['corners']), np.float32([[0,0],[1,0],[1,1],[0,1]]))
    teacher = YOLO('yolo11s.pt')
    previous, rows, details = {}, [], []
    next_id = 1
    for index, path in enumerate(sorted((data/'frames').glob('*.jpg')), 1):
        image = cv2.imread(str(path))
        result = teacher.predict(image, classes=[0], conf=.25, imgsz=1280, device='cpu', verbose=False)[0]
        candidates = []
        for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            foot = np.float32([[[(box[0]+box[2])/2, box[3]]]])
            p = cv2.perspectiveTransform(foot, matrix)[0,0]
            if -.18 <= p[0] <= 1.18 and -.18 <= p[1] <= 1.18:
                candidates.append((box, float(confidence), p))
        candidates = sorted(candidates, key=lambda x: -x[1])[:4]
        # Greedy nearest-foot association is deliberately separate from evaluated trackers.
        available = set(previous)
        for box, confidence, p in sorted(candidates, key=lambda x: (x[2][1], x[2][0])):
            pairs = [(float(np.linalg.norm(p-old[0])), identity) for identity, old in previous.items()
                     if identity in available and index-old[1] <= 15 and (p[1]>.5)==(old[0][1]>.5)]
            distance, identity = min(pairs, default=(float('inf'), -1))
            if distance > .35:
                identity = next_id
                next_id += 1
            else:
                available.remove(identity)
            previous[identity] = (p, index)
            x1,y1,x2,y2 = map(float,box)
            rows.append([index,identity,x1,y1,x2-x1,y2-y1,1,1,1])
            details.append(dict(frame=index,player_id=identity,box=box.tolist(),confidence=confidence,source='teacher_detection',human_verified=False))
            cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,230,255),2)
            cv2.putText(image,f'DRAFT {identity}',(int(x1),max(20,int(y1)-8)),0,.65,(0,230,255),2)
        cv2.imwrite(str(marked/f'frame_{index:06d}.jpg'),image)
        if index % 15 == 0:
            print('Automatic annotation',index,flush=True)
    if not rows:
        raise ValueError('Automatic teacher found no court players; check calibration.')
    with (output/'automatic_reference_mot.txt').open('w',newline='') as file:
        csv.writer(file).writerows(rows)
    (output/'annotations.json').write_text(json.dumps(dict(reference_kind='automatic',human_verified=False,
        detector='YOLO11s COCO pretrained, confidence .25, image size 1280',
        association='Independent greedy court-foot distance; no baseline or proposed predictions used',
        limitation='Unseen players have no box. Detector misses and greedy ID errors bias scores. Teacher shares detector family/training data with student. SportsMOT-inspired format does not establish annotation quality.',
        annotations=details),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--court',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    generate(args.data_root,args.court,args.output)
