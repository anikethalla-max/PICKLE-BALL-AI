"""Replay identical cached detections through three baselines and two ablations.

No annotations are read by this module. Reference labels are used only later
by evaluate_comparison.py. Baselines preserve native tracker IDs.
"""
import csv
import json
import time
import argparse
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import yaml
import ultralytics
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT

from third_party.ocsort.ocsort import OCSort
from adaptive_tracker import AdaptiveCourtTracker
from prototype import COLORS


class CachedEncoder:
    """Resolve BoT-SORT detection crops to their already computed OSNet vectors."""
    def __init__(self):
        self.xywh=None
        self.features=None

    def __call__(self,img,bboxes):
        result=[]
        for box in bboxes:
            # BoT-SORT may call the encoder on high/low-score subsets; use
            # geometry, not subset-relative indices, to find exact cache entries.
            distances=np.abs(self.xywh-box[:4]).sum(axis=1)
            k=int(np.argmin(distances))
            if distances[k]>.05:
                raise ValueError('BoT-SORT crop could not be matched to feature cache')
            result.append(self.features[k])
        return np.asarray(result,dtype=np.float32)


def draw(frame,outputs,title,matrix,frame_number,corners,start_seconds,fps):
    image=frame.copy()
    h,w=image.shape[:2]
    for box,identity,confidence in outputs:
        x1,y1,x2,y2=np.rint(box).astype(int)
        color=COLORS[(int(identity)-1)%4]
        cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
        cv2.putText(image,f'ID {identity}',(x1,max(65,y1-8)),0,.65,color,2)
    cv2.polylines(image,[corners.astype(np.int32)],True,(255,255,0),2)
    bw,bh=max(80,round(w*.078)),max(140,round(h*.25))
    x,y=w-bw-35,h-bh-50
    cv2.rectangle(image,(x-15,y-30),(w-15,h-20),(25,25,25),-1)
    cv2.rectangle(image,(x,y),(x+bw,y+bh),(220,220,220),2)
    cv2.line(image,(x,y+bh//2),(x+bw,y+bh//2),(255,255,255),1)
    cv2.putText(image,'Court mapping',(x-5,y-10),0,.5,(255,255,255),1)
    for box,identity,_ in outputs:
        foot=np.float32([[(box[0]+box[2])/2,box[3]]]).reshape(1,1,2)
        p=cv2.perspectiveTransform(foot,matrix)[0,0]
        if 0<=p[0]<=1 and 0<=p[1]<=1:
            xy=(round(x+p[0]*bw),round(y+p[1]*bh))
            cv2.circle(image,xy,5,COLORS[(int(identity)-1)%4],-1)
            cv2.putText(image,str(identity),(xy[0]+6,xy[1]),0,.45,(255,255,255),1)
    cv2.rectangle(image,(0,0),(w,55),(20,20,20),-1)
    cv2.putText(image,title,(12,23),0,.7,(255,255,255),1)
    seconds=start_seconds+(frame_number-1)/fps
    cv2.putText(image,f'Source {int(seconds)//60:02d}:{seconds%60:06.3f} | shared YOLO11n detections | observed {len(outputs)}',
                (12,46),0,.58,(255,255,255),1)
    return image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',default='data/occlusion_0633')
    parser.add_argument('--output',default='outputs/comparison_0633')
    parser.add_argument('--court',default='reference/court_393s.json')
    parser.add_argument('--seeds',default='reference/seeds_393s.json')
    args=parser.parse_args()
    root=Path(args.data_root)
    meta=json.loads((root/'clip_metadata.json').read_text()) if (root/'clip_metadata.json').exists() else dict(
        fps=30,width=1920,height=1080,start_seconds=393,source_frame_start=11790)
    fps,width,height=meta['fps'],meta['width'],meta['height']
    cache=json.loads((root/'shared_detections.json').read_text())
    records=cache['frames']
    embeddings=np.load(root/'osnet_features.npz')
    frames=sorted((root/'frames').glob('*.jpg'))
    cfgdir=Path(ultralytics.__file__).parent/'cfg'/'trackers'
    bytecfg=yaml.safe_load((cfgdir/'bytetrack.yaml').read_text())
    botcfg=yaml.safe_load((cfgdir/'botsort.yaml').read_text())
    byte=BYTETracker(SimpleNamespace(**bytecfg))
    bot=BOTSORT(SimpleNamespace(**botcfg))
    encoder=CachedEncoder()
    bot.args.with_reid=True
    bot.encoder=encoder
    occonfig=dict(det_thresh=.25,max_age=30,min_hits=3,iou_threshold=.3,delta_t=3,inertia=.2,use_byte=False)
    oc=OCSort(**occonfig)
    cfg=json.loads(Path(args.court).read_text())
    matrix=cv2.getPerspectiveTransform(np.float32(cfg['corners']),np.float32([[0,0],[1,0],[1,1],[0,1]]))
    seeds_xy=json.loads(Path(args.seeds).read_text())
    methods=['bytetrack','ocsort','botsort_osnet','adaptive_osnet','fixed_osnet']
    titles=['ByteTrack (Ultralytics)','OC-SORT (official Python)',
            'BoT-SORT + OSNet (Ultralytics, GMC)',
            'Proposed prototype: adaptive court + OSNet + Hungarian',
            'Ablation: fixed weights, court + OSNet + Hungarian']
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    writers={}
    handles={}
    csvwriters={}
    mot={m:[] for m in methods}
    timings={m:0. for m in methods}
    custom={}
    for name in methods:
        folder=out/name
        folder.mkdir(exist_ok=True)
        writers[name]=cv2.VideoWriter(str(folder/'tracking.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writers[name].isOpened():
            raise RuntimeError('Video writer failed')
        handles[name]=(folder/'tracks.csv').open('w',newline='')
        csvwriters[name]=csv.writer(handles[name])
        csvwriters[name].writerow(['source_frame','time_seconds','player_id','x1','y1','x2','y2','confidence',
                                 'court_x_normalized','court_y_normalized'])
    composite=cv2.VideoWriter(str(out/'four_method_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080))
    try:
        for index,(path,record) in enumerate(zip(frames,records)):
            frame=cv2.imread(str(path))
            raw=np.asarray(record['detections'],np.float32).reshape(-1,5)
            features=embeddings[f'f{index+1:03d}']
            if len(features)!=len(raw):
                raise ValueError('Feature/detection cache mismatch')
            boxes=Boxes(np.column_stack([raw,np.zeros(len(raw))]).astype(np.float32),(height,width)).numpy()
            dets=[]
            for row,feature in zip(raw,features):
                box=row[:4]
                foot=np.float32([[(box[0]+box[2])/2,box[3]]]).reshape(1,1,2)
                p=cv2.perspectiveTransform(foot,matrix)[0,0]
                dets.append(dict(box=box.copy(),p=p,feature=feature,confidence=float(row[4])))
            if index==0:
                seeds=[]
                used=set()
                for x,y in seeds_xy:
                    candidates=[(k,d) for k,d in enumerate(dets) if k not in used and
                                d['box'][0]<=x<=d['box'][2] and d['box'][1]<=y<=d['box'][3]]
                    if not candidates:
                        raise ValueError('Missing first-frame seed detection')
                    k,d=min(candidates,key=lambda item:np.prod(item[1]['box'][2:]-item[1]['box'][:2]))
                    seeds.append(d)
                    used.add(k)
                custom={name:AdaptiveCourtTracker(seeds,fixed=name=='fixed_osnet') for name in methods[-2:]}
            outputs={}
            t=time.perf_counter()
            result=byte.update(boxes,img=frame)
            timings['bytetrack']+=time.perf_counter()-t
            outputs['bytetrack']=[(r[:4],int(r[4]),float(r[5])) for r in result]
            t=time.perf_counter()
            result=oc.update(raw.copy(),(height,width),(height,width))
            timings['ocsort']+=time.perf_counter()-t
            outputs['ocsort']=[(r[:4],int(r[4]),1.) for r in result]
            encoder.xywh=boxes.xywh
            encoder.features=features
            t=time.perf_counter()
            result=bot.update(boxes,img=frame)
            timings['botsort_osnet']+=time.perf_counter()-t
            outputs['botsort_osnet']=[(r[:4],int(r[4]),float(r[5])) for r in result]
            for name,tracker in custom.items():
                t=time.perf_counter()
                matches=tracker.update(dets,index)
                timings[name]+=time.perf_counter()-t
                outputs[name]=[(d['box'],track['id'],d['confidence']) for track,d in matches]
            rendered={}
            for name,title in zip(methods,titles):
                for box,identity,confidence in outputs[name]:
                    x1,y1,x2,y2=map(float,box)
                    foot=np.float32([[(x1+x2)/2,y2]]).reshape(1,1,2)
                    p=cv2.perspectiveTransform(foot,matrix)[0,0]
                    csvwriters[name].writerow([meta['source_frame_start']+index,meta['start_seconds']+index/fps,identity,x1,y1,x2,y2,confidence,*p.tolist()])
                    mot[name].append([index+1,identity,x1,y1,x2-x1,y2-y1,confidence,-1,-1,-1])
                canvas=draw(frame,outputs[name],title,matrix,index+1,np.float32(cfg['corners']),meta['start_seconds'],fps)
                writers[name].write(canvas)
                rendered[name]=cv2.resize(canvas,(960,540))
                if index in (0,24,30,45,104):
                    cv2.imwrite(str(out/name/f'frame_{index+1:03d}.jpg'),canvas)
            grid=np.vstack([np.hstack([rendered[m] for m in methods[:2]]),
                            np.hstack([rendered[m] for m in methods[2:4]])])
            composite.write(grid)
            if index==24:
                cv2.imwrite(str(out/'comparison_overlap.jpg'),grid)
            if (index+1)%30==0:
                print('Compared',index+1,flush=True)
    finally:
        for writer in writers.values():
            writer.release()
        composite.release()
        for handle in handles.values():
            handle.close()
    for name in methods:
        with (out/name/'predictions_mot.txt').open('w',newline='') as file:
            csv.writer(file).writerows(mot[name])
        if name in custom:
            (out/name/'association_audit.json').write_text(json.dumps(custom[name].audit,indent=2))
            (out/name/'recovery_events.json').write_text(json.dumps(custom[name].events,indent=2))
        (out/name/'summary.json').write_text(json.dumps(dict(frames=len(records),source_fps=fps,start_seconds=meta['start_seconds'],
            width=width,height=height,method=name,association_seconds=timings[name],
            timing_note='Association replay only; cached YOLO/OSNet computation and rendering excluded'),indent=2))
    (out/'experiment.json').write_text(json.dumps(dict(
        methods=methods,ultralytics=ultralytics.__version__,byte_config=bytecfg,
        bot_config={**botcfg,'with_reid':True,'model':'external official OSNet-x0.25 MSMT17',
                    'feature_provider':'CachedEncoder, exact current-detection crop geometry'},
        oc_config=occonfig,shared_detection_file=str(root/'shared_detections.json'),
        shared_features=str(root/'osnet_features.npz'),
        initialization='Custom variants use four manually selected first-frame seed points; baselines initialize automatically',
        limitation='Custom persistent four-player prior and manual initialization differ from generic baselines; event is a development demo, not a held-out benchmark',
        baseline_ids='Native tracker IDs preserved; never relabeled using later ground truth',
        court='Same expanded court ROI applied to every detection input. Homography costs only in custom variants; court map drawn for all methods for visualization.'),indent=2))
    print('Comparison complete:',out)


if __name__=='__main__':
    main()
