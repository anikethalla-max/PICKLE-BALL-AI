"""Local dashboard I/O and isolated, observable tracking jobs."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT=Path(__file__).resolve().parent
PREPARED=ROOT/'data/occlusion_0633/input_clip.mp4'
SAVED_RUN=ROOT/'outputs/comparison_0633'
JOBS=ROOT/'outputs/dashboard_jobs'
METHODS={'bytetrack':'ByteTrack','ocsort':'OC-SORT','botsort_osnet':'BoT-SORT + OSNet',
         'adaptive_osnet':'Our adaptive method','fixed_osnet':'Fixed-weight ablation'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temp.replace(path)


def video_info(path):
    cap=cv2.VideoCapture(str(path))
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        count=round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok,frame=cap.read()
        if not cap.isOpened() or not ok or fps<=0 or count<1:
            raise ValueError('This file could not be read as a video. Please use an MP4 clip.')
        h,w=frame.shape[:2]
        return dict(fps=fps,frames=count,width=w,height=h,seconds=count/fps)
    finally:
        cap.release()


def first_frame(path):
    cap=cv2.VideoCapture(str(path))
    ok,frame=cap.read()
    cap.release()
    if not ok:
        raise ValueError('Could not read the first frame')
    return frame


def prepared_clip():
    if not PREPARED.is_file():
        raise ValueError('The prepared demo clip is missing. Upload a short clip or rebuild the prepared clip from the local source video.')
    return dict(path=str(PREPARED),sha256=digest(PREPARED),prepared=True,
                name='Prepared occlusion clip',**video_info(PREPARED))


def restore_prepared_clip():
    """Recreate only the prepared video from the local source, atomically."""
    source=ROOT/'Videos/InPlayBalls_54min_vid.mp4'
    if not source.is_file():
        raise ValueError('Source video not found in Videos/. Please upload a short clip instead.')
    cap=cv2.VideoCapture(str(source))
    writer=None
    temp=PREPARED.with_name('input_clip.restoring.mp4')
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or abs(fps-30)>.01:
            raise ValueError('The prepared demo requires the original 30 FPS source video.')
        cap.set(cv2.CAP_PROP_POS_FRAMES,11790)
        PREPARED.parent.mkdir(parents=True,exist_ok=True)
        for index in range(105):
            ok,frame=cap.read()
            if not ok or frame.shape[:2]!=(1080,1920):
                raise ValueError('The source does not contain the expected 105-frame demo at the expected resolution.')
            if writer is None:
                writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),30,(1920,1080))
                if not writer.isOpened():
                    raise ValueError('Could not create the prepared video.')
            writer.write(frame)
        writer.release()
        writer=None
        cap.release()
        info=video_info(temp)
        if info['frames']!=105:
            raise ValueError('The reconstructed clip has an unexpected frame count.')
        temp.replace(PREPARED)
        return prepared_clip()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if temp.exists():
            temp.unlink()


def save_upload(content):
    sha=hashlib.sha256(content).hexdigest()
    if PREPARED.is_file() and sha==digest(PREPARED):
        return prepared_clip()
    folder=ROOT/'data/dashboard_uploads'/sha[:20]
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'input.mp4'
    if not path.exists():
        path.write_bytes(content)
    info=video_info(path)
    if info['seconds']>30 or info['frames']>1800:
        raise ValueError('Please trim the uploaded clip to 30 seconds or less for this local demonstration.')
    if info['frames']<4:
        raise ValueError('Please use a clip containing at least four frames.')
    return dict(path=str(path),sha256=sha,prepared=False,name='Uploaded clip',**info)


def calibration_valid(points):
    p=np.asarray(points,np.float32)
    return p.shape==(4,2) and np.isfinite(p).all() and cv2.isContourConvex(p) and abs(cv2.contourArea(p))>100


def web_video(path):
    """Convert MPEG-4 Part 2 OpenCV output to broadly playable H.264 MP4."""
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    signature=hashlib.sha256(f'{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}'.encode()).hexdigest()[:24]
    target=ROOT/'outputs/web_media'/f'{signature}.mp4'
    if target.exists():
        return target
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name(target.stem+'.encoding.mp4')
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(path),
             '-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p',
             '-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-movflags','+faststart',str(temp)]
    subprocess.run(command,check=True,capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    temp.replace(target)
    return target


def read_status(job):
    path=Path(job)/'status.json'
    return json.loads(path.read_text()) if path.exists() else {'state':'queued','message':'Starting local job','progress':0}


def start_job(clip,corners,seeds,mode):
    JOBS.mkdir(parents=True,exist_ok=True)
    lock=JOBS/'active_job.json'
    if lock.exists():
        import psutil
        active=json.loads(lock.read_text())
        if psutil.pid_exists(active.get('pid',-1)):
            raise RuntimeError('Another tracking job is still running. Wait for it to finish before starting a new run.')
        lock.unlink()
    job=JOBS/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    job.mkdir()
    atomic_json(job/'config.json',dict(clip=clip,corners=corners,seeds=seeds,mode=mode))
    atomic_json(job/'status.json',dict(state='queued',message='Starting local tracking job',progress=0))
    with (job/'worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--job',str(job)],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    atomic_json(lock,dict(pid=process.pid,job=str(job)))
    return str(job)


def run_job(job):
    job=Path(job).resolve()
    config=json.loads((job/'config.json').read_text())
    clip=config['clip']
    run=job/'run'
    def stage(message,progress):
        atomic_json(job/'status.json',dict(state='running',message=message,progress=progress,run=str(run)))
    def script(name,*args):
        subprocess.run([sys.executable,str(ROOT/name),*map(str,args)],cwd=ROOT,check=True,
            env={**os.environ,'YOLO_CONFIG_DIR':str(ROOT/'data/ultralytics')},
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        court=job/'court.json'
        seeds=job/'seeds.json'
        atomic_json(court,dict(corners=config['corners'],width=clip['width'],height=clip['height']))
        atomic_json(seeds,config['seeds'])
        if config['mode']=='cached' and clip['prepared']:
            data=ROOT/'data/occlusion_0633'
            stage('Using the prepared YOLO detections and OSNet features',.15)
        else:
            data=job/'data'
            folder=data/'frames'
            folder.mkdir(parents=True)
            stage('Preparing consecutive video frames',.05)
            if clip['prepared']:
                # Use the exact original source frames that the annotations refer to.
                for path in (ROOT/'data/occlusion_0633/frames').glob('*.jpg'):
                    shutil.copyfile(path,folder/path.name)
                meta=dict(fps=30,width=1920,height=1080,start_seconds=393,source_frame_start=11790)
            else:
                cap=cv2.VideoCapture(clip['path'])
                count=0
                try:
                    while True:
                        ok,frame=cap.read()
                        if not ok:
                            break
                        cv2.imwrite(str(folder/f'frame_{count:08d}.jpg'),frame)
                        count+=1
                finally:
                    cap.release()
                if count!=clip['frames']:
                    raise ValueError('The uploaded video frame count could not be read consistently.')
                meta=dict(fps=clip['fps'],width=clip['width'],height=clip['height'],start_seconds=0,source_frame_start=0)
            atomic_json(data/'clip_metadata.json',meta)
            stage('Running YOLO11 player detection',.15)
            script('cache_detections.py','--data-root',data,'--court',court)
            stage('Extracting pretrained OSNet appearance features',.42)
            script('appearance_osnet.py','--data-root',data)
        stage('Running three baselines, adaptive association and the fixed-weight ablation',.62)
        script('compare_trackers.py','--data-root',data,'--court',court,'--seeds',seeds,'--output',run)
        if not clip['prepared']:
            stage('Generating independent automatic draft annotations with YOLO11s',.80)
            annotation=job/'annotations'
            script('automatic_reference.py','--data-root',data,'--court',court,'--output',annotation)
            stage('Computing agreement scores against automatic draft labels',.88)
            script('evaluate_comparison.py','--run-root',run,'--reference',annotation/'automatic_reference_mot.txt','--reference-kind','automatic')
        if clip['prepared']:
            stage('Evaluating against the prepared reference draft',.84)
            script('evaluate_comparison.py','--run-root',run)
        stage('Preparing videos for browser playback',.92)
        web_video(run/'four_method_comparison.mp4')
        for name in METHODS:
            web_video(run/name/'tracking.mp4')
        atomic_json(job/'status.json',dict(state='complete',message='Tracking completed',progress=1,
                    run=str(run),metrics_available=True,reference_kind='draft' if clip['prepared'] else 'automatic'))
    except Exception as error:
        atomic_json(job/'status.json',dict(state='failed',message=str(error),progress=0,run=str(run)))
        raise
    finally:
        lock=JOBS/'active_job.json'
        if lock.exists() and json.loads(lock.read_text()).get('job')==str(job):
            lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--prepare-media',action='store_true')
    args=parser.parse_args()
    if args.prepare_media:
        for path in [PREPARED,SAVED_RUN/'four_method_comparison.mp4',
                     *[SAVED_RUN/name/'tracking.mp4' for name in METHODS]]:
            print(web_video(path),flush=True)
    elif args.job:
        run_job(args.job)
