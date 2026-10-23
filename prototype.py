"""Smart Doubles AI: short, fixed-camera tracking prototype.

Appearance uses a colour histogram until validated OSNet integration is added.
Never reports identity accuracy without ground-truth annotations.
"""
import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

VIDEO = 'Videos/InPlayBalls_54min_vid.mp4'
COLORS = [(255, 150, 40), (50, 70, 255), (60, 220, 60), (20, 200, 255)]


def capture(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    if cap.get(cv2.CAP_PROP_FPS) <= 0:
        cap.release()
        raise ValueError('Video has no usable frame rate')
    return cap


def extract(args):
    cap = capture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    first = round(args.start * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    count = 0
    try:
        for offset in range(round(args.seconds * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            if offset % args.every == 0:
                if not cv2.imwrite(str(out / f'frame_{first + offset:08d}.jpg'), frame):
                    raise IOError('Failed to save frame')
                count += 1
    finally:
        cap.release()
    print(f'Saved {count} frames to {out.resolve()}')


def calibrate(args):
    cap = capture(args.video)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError('No frame at requested start time')
    # These must be physical baseline corners, not the kitchen corners.
    points = []
    title = 'Court: far-left, far-right, near-right, near-left. Enter save; R reset; Esc cancel'

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, click)
    try:
        while True:
            display = frame.copy()
            for i, point in enumerate(points):
                cv2.circle(display, tuple(point), 6, (0, 255, 255), -1)
                cv2.putText(display, str(i + 1), tuple(point), 0, 1, (0, 0, 255), 2)
            cv2.imshow(title, display)
            key = cv2.waitKey(30) & 255
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return
            if key in (ord('r'), ord('R')):
                points.clear()
            if key == 13 and len(points) == 4:
                poly = np.array(points, np.float32)
                if not cv2.isContourConvex(poly) or abs(cv2.contourArea(poly)) < 100:
                    print('Invalid corner order or court polygon. Press R and retry.')
                    continue
                out = Path(args.court)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({'corners': points, 'width': frame.shape[1],
                                           'height': frame.shape[0]}, indent=2))
                print(f'Saved calibration: {out.resolve()}')
                return
    finally:
        cv2.destroyAllWindows()


def appearance(frame, box):
    x1, y1, x2, y2 = map(int, box)
    # Upper body reduces contamination from court and shoes.
    crop = frame[max(0, y1):max(0, y1 + (y2-y1)*2//3), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return np.zeros(128, np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    return hist / max(float(np.linalg.norm(hist)), 1e-8)


class CourtTracker:
    """Four manually initialized identities; gated Hungarian association.

    Coordinates are normalized court coordinates. Track lifetime is bounded.
    Court side is a soft penalty during a continuous rally, never a hard rule.
    """
    def __init__(self, seeds, frame, max_gap):
        self.max_gap = max_gap
        self.tracks = []
        for idx, det in enumerate(seeds):
            self.tracks.append(dict(id=idx+1, p=det['p'], v=np.zeros(2),
                                    a=det['a'], last=frame, side=det['p'][1] >= .5,
                                    trail=deque(maxlen=60)))

    def update(self, detections, frame):
        costs = np.full((4, len(detections)), 1e6)
        for i, track in enumerate(self.tracks):
            gap = frame-track['last']
            if gap > self.max_gap:
                continue
            predicted = track['p'] + track['v'] * min(gap, 10)
            for j, det in enumerate(detections):
                court = float(np.linalg.norm(det['p']-track['p']))
                motion = float(np.linalg.norm(det['p']-predicted))
                app = float(np.clip(1-np.dot(track['a'], det['a']), 0, 1))
                rule = float((det['p'][1] >= .5) != track['side'])
                # Simple adaptive heuristic: trust appearance more after missed frames.
                weights = (.20, .45, .25, .10) if gap > 1 else (.40, .25, .25, .10)
                if motion > min(.55, .12 + .025*gap):
                    continue
                costs[i, j] = np.dot(weights, [motion, app, court, rule])
        matches = []
        if not detections:
            return matches
        rows, cols = linear_sum_assignment(costs)
        for i, j in zip(rows, cols):
            if costs[i, j] > .48:
                continue
            track, det = self.tracks[i], detections[j]
            gap = frame-track['last']
            track['v'] = .5*track['v'] + .5*(det['p']-track['p'])/gap
            track['p'], track['last'] = det['p'], frame
            # Avoid corrupting appearance memory during overlapping boxes.
            if not det['overlap']:
                feature = .9*track['a'] + .1*det['a']
                track['a'] = feature/max(float(np.linalg.norm(feature)), 1e-8)
            track['trail'].append(tuple(map(int, det['foot'])))
            matches.append((track, det))
        return matches


def initialize(frame, detections):
    selected = []
    title = 'Select P1 P2 P3 P4 inside player boxes; Enter confirm; R reset; Esc cancel'

    def click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(selected) == 4:
            return
        candidates = [(k, d) for k, d in enumerate(detections)
                      if k not in selected and d['box'][0] <= x <= d['box'][2]
                      and d['box'][1] <= y <= d['box'][3]]
        if candidates:
            selected.append(min(candidates, key=lambda item:
                (item[1]['box'][2]-item[1]['box'][0])*(item[1]['box'][3]-item[1]['box'][1]))[0])

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, click)
    try:
        while True:
            display = frame.copy()
            for k, det in enumerate(detections):
                x1,y1,x2,y2 = map(int, det['box'])
                cv2.rectangle(display, (x1,y1), (x2,y2), (170,170,170), 2)
                if k in selected:
                    identity = selected.index(k)
                    cv2.putText(display, f'P{identity+1}', (x1,y1), 0, 1, COLORS[identity], 2)
            cv2.imshow(title, display)
            key = cv2.waitKey(30) & 255
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError('Initialization cancelled')
            if key in (ord('r'), ord('R')):
                selected.clear()
            if key == 13 and len(selected) == 4:
                return [detections[k] for k in selected]
    finally:
        cv2.destroyAllWindows()


def run(args):
    from ultralytics import YOLO
    config = json.loads(Path(args.court).read_text())
    corners = np.array(config['corners'], np.float32)
    transform = cv2.getPerspectiveTransform(corners, np.float32([[0,0],[1,0],[1,1],[0,1]]))
    cap = capture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(args.start*fps))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    (out/'homography.json').write_text(json.dumps({
        'image_to_normalized_court': transform.tolist(), 'corners': corners.tolist(),
        'coordinate_system': 'x left-to-right, y far-to-near, normalized court',
        'calibration': str(args.court)}, indent=2))
    tracker, writer = None, None
    processed, visible, total_time = 0, 0, 0.
    started = time.perf_counter()
    try:
        with (out/'tracks.csv').open('w', newline='') as handle:
            csvout = csv.writer(handle)
            csvout.writerow(['source_frame','time_seconds','player_id','x1','y1','x2','y2',
                             'confidence','court_x_normalized','court_y_normalized'])
            for index in range(round(args.seconds*fps)):
                ok, frame = cap.read()
                if not ok:
                    break
                h,w = frame.shape[:2]
                if (w,h) != (config['width'], config['height']):
                    raise ValueError('Calibration resolution differs from video')
                tick = time.perf_counter()
                result = model.predict(frame, classes=[0], conf=args.conf, imgsz=args.imgsz,
                                       device=args.device, verbose=False)[0]
                detections = []
                for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    foot = np.float32([(box[0]+box[2])/2, box[3]])
                    p = cv2.perspectiveTransform(foot.reshape(1,1,2), transform)[0,0]
                    # Margin includes players standing just outside baseline/sideline.
                    if not np.isfinite(p).all() or not (-.18 <= p[0] <= 1.18 and -.18 <= p[1] <= 1.18):
                        continue
                    detections.append(dict(box=box, foot=foot, p=p, confidence=float(confidence),
                                           a=appearance(frame, box), overlap=False))
                for i,d in enumerate(detections):
                    for e in detections[i+1:]:
                        a,b = d['box'],e['box']
                        if min(a[2],b[2]) > max(a[0],b[0]) and min(a[3],b[3]) > max(a[1],b[1]):
                            d['overlap'] = e['overlap'] = True
                if tracker is None:
                    if len(detections) < 4:
                        raise ValueError('Fewer than four court candidates at start. Choose a clearer --start.')
                    if args.seeds:
                        seed_points = json.loads(Path(args.seeds).read_text())
                        if len(seed_points) != 4:
                            raise ValueError('Seeds must contain four [x,y] image points in P1-P4 order')
                        seeds, used = [], set()
                        for x,y in seed_points:
                            candidates = [(k,d) for k,d in enumerate(detections) if k not in used
                                          and d['box'][0] <= x <= d['box'][2]
                                          and d['box'][1] <= y <= d['box'][3]]
                            if not candidates:
                                cv2.imwrite(str(out/'initialization_frame.jpg'), frame)
                                raise ValueError(f'No detected candidate contains seed {(x,y)}. Use interactive initialization.')
                            k,d = min(candidates,key=lambda item:
                                      (item[1]['box'][2]-item[1]['box'][0])*(item[1]['box'][3]-item[1]['box'][1]))
                            used.add(k)
                            seeds.append(d)
                    else:
                        seeds = initialize(frame, detections)
                    tracker = CourtTracker(seeds, index-1, round(args.max_gap*fps))
                matches = tracker.update(detections, index)
                total_time += time.perf_counter()-tick if index else 0
                visible += len(matches)
                canvas = frame.copy()
                for track,det in matches:
                    identity = track['id']
                    color = COLORS[identity-1]
                    x1,y1,x2,y2 = map(int,det['box'])
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                    cv2.putText(canvas,f'P{identity} ({det["p"][0]:.2f}, {det["p"][1]:.2f})',
                                (x1,max(65,y1-8)),0,.55,color,2)
                    if len(track['trail']) > 1:
                        cv2.polylines(canvas,[np.array(track['trail'],np.int32)],False,color,2)
                    source_frame = round(args.start*fps)+index
                    csvout.writerow([source_frame,source_frame/fps,identity,*det['box'].tolist(),
                                     det['confidence'],*det['p'].tolist()])
                cv2.polylines(canvas,[corners.astype(np.int32)],True,(255,255,0),2)
                # Court plot: observed positions only, no invented missing-player markers.
                plot_x, plot_y, plot_w, plot_h = w-200, h-340, 160, 280
                cv2.rectangle(canvas,(plot_x-15,plot_y-25),(w-10,h-20),(30,30,30),-1)
                cv2.rectangle(canvas,(plot_x,plot_y),(plot_x+plot_w,plot_y+plot_h),(220,220,220),2)
                cv2.line(canvas,(plot_x,plot_y+plot_h//2),(plot_x+plot_w,plot_y+plot_h//2),(255,255,255),2)
                cv2.putText(canvas,'Court position',(plot_x-5,plot_y-8),0,.48,(255,255,255),1)
                for track,det in matches:
                    px,py = det['p']
                    if 0 <= px <= 1 and 0 <= py <= 1:
                        xy = (round(plot_x+plot_w*px),round(plot_y+plot_h*py))
                        cv2.circle(canvas,xy,5,COLORS[track['id']-1],-1)
                        cv2.putText(canvas,f'P{track["id"]}',(xy[0]+7,xy[1]),0,.45,COLORS[track['id']-1],1)
                cv2.rectangle(canvas,(0,0),(w,55),(25,25,25),-1)
                cv2.putText(canvas,'PROTOTYPE: YOLO11 + court + Hungarian; colour appearance (not OSNet)',
                            (10,22),0,.52,(255,255,255),1)
                cv2.putText(canvas,f'Observed identities: {len(matches)}/4 | missing players are not drawn',
                            (10,44),0,.52,(255,255,255),1)
                timestamp = args.start+index/fps
                cv2.putText(canvas,f'Source {int(timestamp)//60:02d}:{timestamp%60:06.3f}',
                            (w-280,43),0,.6,(255,255,255),1)
                if writer is None:
                    writer = cv2.VideoWriter(str(out/'prototype.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                    if not writer.isOpened():
                        raise IOError('Video writer failed')
                writer.write(canvas)
                if index % max(1, round(fps*.5)) == 0:
                    cv2.imwrite(str(out/f'demo_{index:06d}.jpg'),canvas)
                processed += 1
                if processed % 30 == 0:
                    print(f'Processed {processed} frames',flush=True)
    finally:
        cap.release()
        if writer:
            writer.release()
    summary = dict(frames=processed, source_fps=fps, start_seconds=args.start,
                   width=config['width'], height=config['height'],
                   wall_seconds=time.perf_counter()-started,
                   processing_fps_excluding_initialization=(processed-1)/total_time if total_time else None,
                   mean_observed_identities=visible/processed if processed else 0,
                   appearance='HSV histogram, NOT OSNet', assignment='gated Hungarian',
                   limitations=['No ground truth: no accuracy or recovery claim',
                                'Fixed-camera continuous rally only; reset after cuts or end changes',
                                'Heuristic motion and adaptive weights; no Kalman/OSNet yet',
                                'Lost IDs expire after max-gap; restart and reinitialize thereafter'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['extract','calibrate','run'])
    parser.add_argument('--video',default=VIDEO)
    parser.add_argument('--start',type=float,default=60)
    parser.add_argument('--seconds',type=float,default=15)
    parser.add_argument('--every',type=int,default=1)
    parser.add_argument('--court',default='data/court.json')
    parser.add_argument('--seeds',help='Optional JSON with four manually chosen pixel points for headless initialization')
    parser.add_argument('--output',default='outputs/demo')
    parser.add_argument('--model',default='yolo11n.pt')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--imgsz',type=int,default=960)
    parser.add_argument('--conf',type=float,default=.25)
    parser.add_argument('--max-gap',type=float,default=2)
    args = parser.parse_args()
    if args.start < 0 or args.seconds <= 0 or args.every < 1 or args.max_gap <= 0:
        parser.error('start must be nonnegative; seconds, every and max-gap must be positive')
    dict(extract=extract,calibrate=calibrate,run=run)[args.command](args)


if __name__ == '__main__':
    main()
