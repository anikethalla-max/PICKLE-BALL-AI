"""Court Kalman prediction, OSNet memory, risk-adaptive costs and recovery.

This is an experimental heuristic implementation. Parameters are not learned
or validated on held-out matches. Only actual detections are emitted.
"""
from collections import deque
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def overlap_fraction(a,b):
    x1,y1=np.maximum(a[:2],b[:2])
    x2,y2=np.minimum(a[2:],b[2:])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    area=min(np.prod(a[2:]-a[:2]),np.prod(b[2:]-b[:2]))
    return float(intersection/max(area,1))


def adaptive_weights(risk,gap,fixed=False):
    if fixed:
        return np.array([.35,.35,.20,.10])
    if gap>1:
        # Clean returning appearance is useful; corrupted overlapping crops are not.
        return (1-risk)*np.array([.20,.50,.20,.10])+risk*np.array([.45,.15,.30,.10])
    return (1-risk)*np.array([.35,.35,.20,.10])+risk*np.array([.50,.10,.30,.10])


class AdaptiveCourtTracker:
    def __init__(self,seeds,max_gap=60,fixed=False):
        self.fixed=fixed
        self.max_gap=max_gap
        self.tracks=[]
        self.audit=[]
        self.events=[]
        for identity,det in enumerate(seeds,1):
            kf=cv2.KalmanFilter(4,2)
            kf.transitionMatrix=np.float32([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
            kf.measurementMatrix=np.float32([[1,0,0,0],[0,1,0,0]])
            kf.processNoiseCov=np.diag([1e-5,1e-5,1e-4,1e-4]).astype(np.float32)
            kf.measurementNoiseCov=np.eye(2,dtype=np.float32)*.0004
            kf.errorCovPost=np.diag([.001,.001,.01,.01]).astype(np.float32)
            kf.statePost=np.float32([*det['p'],0,0]).reshape(4,1)
            self.tracks.append(dict(id=identity,kf=kf,p=det['p'].copy(),box=det['box'].copy(),
                last=0,side=det['p'][1]>=.5,gallery=deque([det['feature'].copy()],maxlen=20),state='active'))

    def update(self,detections,index):
        predicted=[t['kf'].predict()[:2,0].copy() for t in self.tracks]
        n=len(detections)
        pair_risk=np.zeros(n)
        for j,d in enumerate(detections):
            pair_risk[j]=max([overlap_fraction(d['box'],e['box']) for k,e in enumerate(detections) if j!=k] or [0])
        # Dedicated dummy columns let every identity remain unmatched without
        # forcing an inadmissible detection assignment.
        cost=np.full((4,n+4),1e4)
        details={}
        for i,t in enumerate(self.tracks):
            gap=index-t['last']
            cost[i,n+i]=.65
            if gap>self.max_gap:
                t['state']='lost'
                continue
            for j,d in enumerate(detections):
                displacement=float(np.linalg.norm(d['p']-t['p']))
                motion=float(np.linalg.norm(d['p']-predicted[i]))
                if motion>min(.5,.12+.015*gap):
                    continue
                appearance=float(np.clip(min(1-np.dot(f,d['feature']) for f in t['gallery']),0,2)/2)
                risk=float(max(pair_risk[j],max(0,(.5-d['confidence'])/.5)))
                weights=adaptive_weights(risk,gap,self.fixed)
                terms=np.array([min(1,motion/.20),appearance,min(1,displacement/.30),
                                float((d['p'][1]>=.5)!=t['side'])])
                score=float(weights@terms)
                # A returning detection must have a plausible appearance as well
                # as court position. This is a heuristic threshold, not a result.
                if gap>1 and appearance>.40:
                    continue
                cost[i,j]=score
                details[i,j]=(risk,weights,terms,gap)
        rows,cols=linear_sum_assignment(cost)
        matched=[]
        assigned=set()
        for i,j in zip(rows,cols):
            if j>=n or cost[i,j]>=.65:
                continue
            t,d=self.tracks[i],detections[j]
            risk,weights,terms,gap=details[i,j]
            if gap>1:
                self.events.append({'frame':index+1,'player_id':t['id'],'type':'reconnected',
                                    'gap_frames':gap-1,'previous_state':t['state']})
            # Measurement uncertainty rises with overlap and weak detection score.
            t['kf'].measurementNoiseCov=np.eye(2,dtype=np.float32)*(.0004*(1+4*risk))
            t['kf'].correct(np.float32(d['p']).reshape(2,1))
            if risk<.2 and d['confidence']>=.5:
                t['gallery'].append(d['feature'].copy())
            t.update(p=d['p'].copy(),box=d['box'].copy(),last=index,
                     state='occlusion_risk' if risk>=.2 else 'active')
            self.audit.append({'frame':index+1,'player_id':t['id'],'risk':risk,'state':t['state'],
                'weights':weights.tolist(),'cost_terms':terms.tolist(),'cost':float(cost[i,j]),
                'memory_updated':bool(risk<.2 and d['confidence']>=.5)})
            matched.append((t,d))
            assigned.add(i)
        for i,t in enumerate(self.tracks):
            if i not in assigned and t['state']!='lost':
                if t['state']!='missing':
                    self.events.append({'frame':index+1,'player_id':t['id'],'type':'missing'})
                t['state']='missing'
        return matched
