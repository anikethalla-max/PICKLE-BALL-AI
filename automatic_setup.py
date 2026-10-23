"""Register the known demonstration camera; require manual setup for other views."""
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from dashboard_backend import ROOT, PREPARED, first_frame, calibration_valid


def setup(clip):
    if not PREPARED.exists():
        raise ValueError('Known-camera reference is missing. Select court corners and players manually.')
    target=first_frame(clip['path'])
    reference=first_frame(PREPARED)
    detector=cv2.SIFT_create(nfeatures=4000)
    ka,da=detector.detectAndCompute(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),None)
    kb,db=detector.detectAndCompute(cv2.cvtColor(target,cv2.COLOR_BGR2GRAY),None)
    if da is None or db is None:
        raise ValueError('Camera registration failed; use manual setup.')
    matches=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance < .65*b.distance]
    if len(good)<30:
        raise ValueError('This camera is not recognized; use manual setup.')
    transform,mask=cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]),np.float32([kb[m.trainIdx].pt for m in good]),cv2.RANSAC,3)
    if transform is None or mask.sum()<25 or mask.mean()<.45:
        raise ValueError('Camera registration is uncertain; use manual setup.')
    original=json.loads((ROOT/'reference/court_393s.json').read_text())['corners']
    corners=cv2.perspectiveTransform(np.float32(original).reshape(-1,1,2),transform).reshape(-1,2)
    if not calibration_valid(corners) or np.any(corners<0) or np.any(corners[:,0]>=clip['width']) or np.any(corners[:,1]>=clip['height']):
        raise ValueError('Registered court is outside this view; use manual setup.')
    matrix=cv2.getPerspectiveTransform(corners,np.float32([[0,0],[1,0],[1,1],[0,1]]))
    torch.set_num_threads(4)
    boxes=YOLO(str(ROOT/'yolo11n.pt')).predict(target,classes=[0],conf=.25,imgsz=960,device='cpu',verbose=False)[0].boxes
    players=[]
    for box,score in zip(boxes.xyxy.cpu().numpy(),boxes.conf.cpu().numpy()):
        p=cv2.perspectiveTransform(np.float32([[[(box[0]+box[2])/2,box[3]]]]),matrix)[0,0]
        if -.18<=p[0]<=1.18 and -.18<=p[1]<=1.18:
            players.append((box,float(score),p))
    chosen=[]
    for near in (True,False):
        side=sorted([x for x in players if (x[2][1]>.5)==near],key=lambda x:-x[1])[:2]
        if len(side)!=2:
            raise ValueError('Four initial players are not visible; select initial identities manually.')
        chosen.extend(sorted(side,key=lambda x:x[2][0]))
    seeds=[[(float(b[0])+float(b[2]))/2,(float(b[1])+float(b[3]))/2] for b,_,_ in chosen]
    return corners.tolist(),seeds
