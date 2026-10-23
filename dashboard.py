"""Local Smart Doubles AI review dashboard. Run with Streamlit."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from dashboard_backend import (ROOT,PREPARED,SAVED_RUN,METHODS,prepared_clip,save_upload,
    calibration_valid,first_frame,web_video,start_job,read_status,restore_prepared_clip)

st.set_page_config(page_title='Smart Doubles AI | Prototype',page_icon='🎾',layout='wide')
st.markdown('''<style>
.block-container{max-width:1440px;padding-top:2rem;padding-bottom:3rem}
h1{font-size:2.8rem!important;letter-spacing:-.04em;margin-bottom:.1rem!important}
h2,h3{letter-spacing:-.025em} [data-testid="stMetricValue"]{font-size:1.8rem}
[data-testid="stSidebar"]{border-right:1px solid #2c3c37}
[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"]{display:none}
.eyebrow{color:#b8f36b;font-size:.78rem;letter-spacing:.14em;font-weight:700}
.intro{color:#b8c9bf;font-size:1.05rem;max-width:820px;margin-bottom:1.6rem}
.stTabs [data-baseweb="tab-list"]{gap:2rem;border-bottom:1px solid #30403a}
.stTabs [data-baseweb="tab"]{font-size:1rem;padding-bottom:1.1rem}
</style>''',unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def json_data(path,mtime):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_json(path):
    path=Path(path)
    return json_data(str(path),path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def movie_frame(path,index,mtime):
    cap=cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    ok,frame=cap.read()
    cap.release()
    if not ok:
        raise ValueError('Unable to decode selected frame')
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


def show_video(path):
    path=Path(path)
    if path.exists():
        try:
            st.video(str(web_video(path)))
        except Exception as error:
            st.error(f'Video playback could not be prepared: {error}')
    else:
        st.info('Run tracking to generate this video.')


def point_editor(clip):
    sha=clip['sha256'][:20]
    stage=st.radio('Selection step',['Court corners','Player identities'],horizontal=True)
    part='court' if stage=='Court corners' else 'players'
    key=f'points_{sha}_{part}'
    nonce=f'nonce_{key}'
    st.session_state.setdefault(key,[])
    st.session_state.setdefault(nonce,0)
    labels=['Far left','Far right','Near right','Near left'] if part=='court' else ['P1','P2','P3','P4']
    st.write('Click in this order: **'+' → '.join(labels)+'**.')
    st.caption('Court points must be the outer baseline corners. Player points must lie inside each player’s body box.')
    points=st.session_state[key]
    image=first_frame(clip['path']).copy()
    for index,point in enumerate(points):
        cv2.circle(image,tuple(point),9,(0,255,255),-1)
        cv2.putText(image,labels[index],tuple(point),0,1,(0,0,255),2)
    width=min(900,clip['width'])
    height=round(image.shape[0]*width/image.shape[1])
    image=cv2.cvtColor(cv2.resize(image,(width,height)),cv2.COLOR_BGR2RGB)
    value=streamlit_image_coordinates(image,width='content',key=f'click_{key}_{st.session_state[nonce]}',cursor='crosshair')
    if value and value.get('unix_time')!=st.session_state.get('event_'+key) and len(points)<4:
        st.session_state['event_'+key]=value.get('unix_time')
        display_width=value.get('width',width)
        display_height=value.get('height',height)
        point=[round(value['x']/display_width*clip['width']),round(value['y']/display_height*clip['height'])]
        if 0<=point[0]<clip['width'] and 0<=point[1]<clip['height']:
            st.session_state[key]=[*points,point]
            st.rerun()
    a,b=st.columns([1,3])
    if a.button('Reset selection',key='reset_'+key):
        st.session_state[key]=[]
        st.session_state[nonce]+=1
        st.rerun()
    b.caption(f'{len(points)}/4 points selected')
    if points:
        st.dataframe(pd.DataFrame(points,columns=['x','y'],index=labels[:len(points)]),width='stretch')
    corners=st.session_state.get(f'points_{sha}_court',[])
    seeds=st.session_state.get(f'points_{sha}_players',[])
    if len(corners)==4 and not calibration_valid(corners):
        st.error('The court polygon is invalid. Reset and select corners in the specified order.')
    return corners,seeds


with st.sidebar:
    st.markdown('<div class="eyebrow">SMART DOUBLES AI</div>',unsafe_allow_html=True)
    st.title('Review demo')
    st.caption('LOCAL APPLICATION')
    st.write('Video and results stay on this computer.')
    st.caption('Reference annotations follow SportsMOT-inspired rules. Draft labels still need independent human verification.')
    st.markdown('[SportsMOT · ICCV 2023](https://github.com/MCG-NJU/SportsMOT)')

st.markdown('<div class="eyebrow">SINGLE-CAMERA PICKLEBALL DOUBLES</div>',unsafe_allow_html=True)
st.title('Player identities through occlusion')
st.markdown('<div class="intro">Inspect the overlap, compare three baseline trackers, and explore court-aware adaptive association.</div>',unsafe_allow_html=True)
st.subheader('Upload your video')
extra_clip=ROOT/'data/occlusion_cross_court_0247/input_clip_3s.mp4'
if extra_clip.exists():
    st.download_button('Download cross-court occlusion clip · 3 seconds',extra_clip.read_bytes(),file_name='cross_court_occlusion_3s.mp4',mime='video/mp4')
source=st.radio('Video input',['Upload a short clip','Prepared occlusion clip'],horizontal=True)
clip=None
if source=='Upload a short clip':
    uploaded=st.file_uploader('Upload MP4',type=['mp4','mov','m4v'],help='Select Browse files or drag your clip here. Up to 30 seconds; uploads stay on this computer.')
    if uploaded:
        try:
            clip=save_upload(uploaded.getvalue())
            if clip['prepared']:
                st.success('Prepared clip recognized. Calibration and reference draft loaded.')
        except ValueError as error:
            st.error(str(error))
else:
    try:
        clip=prepared_clip()
    except ValueError as error:
        st.warning(str(error))
if PREPARED.is_file():
    st.download_button('Download prepared 3.5 s clip',PREPARED.read_bytes(),file_name='pickleball_occlusion_3_5s.mp4',mime='video/mp4')
elif (ROOT/'Videos/InPlayBalls_54min_vid.mp4').is_file():
    if st.button('Rebuild prepared clip'):
        try:
            with st.spinner('Extracting 06:33.000–06:36.500 from the source video…'):
                restore_prepared_clip()
            st.rerun()
        except ValueError as error:
            st.error(str(error))
if clip is None:
    st.info('Click Browse files above to select your video, or choose Prepared occlusion clip to use the local demo.')
    st.stop()

prepared=clip['prepared']
if prepared:
    corners=read_json(ROOT/'reference/court_393s.json')['corners']
    seeds=read_json(ROOT/'reference/seeds_393s.json')
else:
    automatic_key='automatic_'+clip['sha256']
    if automatic_key not in st.session_state:
        try:
            from automatic_setup import setup
            with st.spinner('Recognizing court and initial players…'):
                st.session_state[automatic_key]=setup(clip)
        except Exception as error:
            st.session_state[automatic_key]=str(error)
    auto_setup=st.session_state[automatic_key]
    corners=st.session_state.get(f'points_{clip["sha256"][:20]}_court',[])
    seeds=st.session_state.get(f'points_{clip["sha256"][:20]}_players',[])
    if isinstance(auto_setup,tuple):
        corners,seeds=auto_setup

job=st.session_state.get('active_job')
job_matches=job and read_json(Path(job)/'config.json')['clip']['sha256']==clip['sha256']
status=read_status(job) if job_matches else None
run=Path(status['run']) if status and status['state']=='complete' else SAVED_RUN if prepared and (SAVED_RUN/'experiment.json').is_file() else None
is_saved=run==SAVED_RUN

a,b,c,d=st.columns(4)
a.metric('Selected sequence','06:33–06:36.5' if prepared else f'{clip["seconds"]:.2f} seconds')
b.metric('Consecutive frames',str(clip['frames']))
c.metric('Frame rate',f'{clip["fps"]:g} FPS')
d.metric('Reference status','Draft labels' if prepared else 'Automatic draft on run')
st.caption('Results below: '+('last completed run' if run and not is_saved else 'saved prepared-clip experiment' if run else 'no tracking run yet'))
input_tab,track_tab,annotation_tab,metric_tab,method_tab=st.tabs(['1 · Input','2 · Tracking','3 · Annotations','4 · Metrics','5 · Method'])

with input_tab:
    left,right=st.columns([1.7,1])
    with left:
        st.subheader('Start with the crossing')
        show_video(clip['path'])
        if prepared:
            st.caption('Near-side black outfit crosses in front of near-side red outfit around 06:33.7–06:34.1. Video has no audio.')
    with right:
        st.subheader('Demonstration procedure')
        st.write('1. Inspect the input and reference labels.\n2. Run the tracker comparison or replay saved outputs.\n3. Inspect identities before and after overlap.\n4. Read preliminary metrics with their limitations.')
        st.caption(f'{clip["width"]} × {clip["height"]} pixels · fixed-camera clip')
        if prepared:
            st.success('Saved court calibration and initial P1–P4 selections are ready.')
        else:
            st.info('Upload → automatic draft labels → four tracking methods → agreement scores. Review labels before reporting accuracy.')
    if not prepared:
        if isinstance(auto_setup,tuple):
            st.success('Known camera registered; court and four initial players selected automatically.')
            with st.expander('Override automatic court and player setup'):
                manual_corners,manual_seeds=point_editor(clip)
                if calibration_valid(manual_corners) and len(manual_seeds)==4:
                    corners,seeds=manual_corners,manual_seeds
        else:
            st.info(auto_setup)
            corners,seeds=point_editor(clip)
    else:
        with st.expander('Inspect saved calibration and initial identities'):
            image=first_frame(clip['path'])
            cv2.polylines(image,[np.int32(corners)],True,(255,255,0),3)
            for index,point in enumerate(seeds,1):
                cv2.circle(image,tuple(point),8,(0,255,255),-1)
                cv2.putText(image,f'P{index}',tuple(point),0,1,(0,255,255),2)
            st.image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),caption='Approximate manually selected baseline corners and initial player points.',width='stretch')

with track_tab:
    st.subheader('Compare the tracking methods')
    st.write('ByteTrack · OC-SORT · BoT-SORT + OSNet · Our adaptive method')
    if prepared:
        mode=st.radio('Execution mode',['Replay associations from cached detections','Run full pipeline: YOLO + OSNet + tracking'],
            help='Cached mode still executes all trackers. It skips the already computed detector and appearance features.')
        cached=mode.startswith('Replay')
    else:
        cached=False
        st.caption('New uploads run frame extraction, YOLO, OSNet and all tracking methods locally. Independent YOLO11s draft annotations provide provisional agreement scores.')
    ready=calibration_valid(corners) and len(seeds)==4
    running=bool(status and status['state'] in ('queued','running'))
    if st.button('Annotate and compare',type='primary',disabled=not ready or running):
        try:
            st.session_state.active_job=start_job(clip,corners,seeds,'cached' if cached else 'full')
            st.session_state.pop('completed_job',None)
            st.rerun()
        except Exception as error:
            st.error(str(error))
    if not ready:
        st.caption('Complete court and player selection on the Input tab first.')

    @st.fragment(run_every='2s')
    def job_panel(active):
        state=read_status(active)
        if state['state'] in ('queued','running'):
            st.progress(state.get('progress',0),text=state['message'])
        elif state['state']=='failed':
            st.error('Tracking could not complete: '+state['message'])
            with st.expander('View local run log'):
                log=Path(active)/'worker.log'
                st.code(log.read_text(encoding='utf-8',errors='replace')[-12000:] if log.exists() else 'No log yet.')
        else:
            st.success('Local tracking run completed.')
            if st.session_state.get('completed_job')!=active:
                st.session_state.completed_job=active
                st.rerun()
    if job_matches:
        job_panel(job)
    if run:
        view=st.selectbox('Output to inspect',['Four-method comparison',*METHODS.values()])
        name=next((name for name,label in METHODS.items() if label==view),None)
        video=run/name/'tracking.mp4' if name else run/'four_method_comparison.mp4'
        show_video(video)
        st.caption('Native baseline IDs may differ from P1–P4. Missing observations are not filled with invented player boxes.')
        with st.expander('Inspect a tracked frame'):
            frame_index=st.slider('Tracking frame',1,clip['frames'],min(25,clip['frames']))
            if video.exists():
                st.image(movie_frame(str(video),frame_index-1,video.stat().st_mtime_ns),width='stretch')

with annotation_tab:
    st.subheader('Reference identities, frame by frame')
    if prepared:
        st.warning('Assistant-drawn reference draft: visually reviewed keyframes and interpolation. Independent human verification is pending.')
        index=st.slider('Reference frame',1,105,25)
        path=ROOT/'data/occlusion_0633/frames'/f'frame_{11789+index:08d}.jpg'
        marked=ROOT/'annotations/occlusion_0633/annotated_frames'/path.name
        a,b=st.columns(2)
        if path.is_file():
            a.image(str(path),caption=f'Raw frame · source {393+(index-1)/30:.3f} s',width='stretch')
        else:
            a.info('Extract the prepared frames to inspect raw images.')
        if marked.is_file():
            b.image(str(marked),caption='Reference draft · P1–P4',width='stretch')
        else:
            b.info('Run build_annotations.py to generate the labeled frame images.')
        st.caption('105 frames · 420 full-athlete boxes · equipment and spectators excluded. Partial occlusion is labeled separately.')
        st.dataframe(pd.DataFrame({'Identity':['P1','P2','P3','P4'],
            'Player':['Near red top / red cap','Near black outfit','Far red top','Far grey outfit / white cap']}),hide_index=True,width='stretch')
        for label,path in [('Download MOT annotation draft',ROOT/'annotations/occlusion_0633/reference_draft_mot.txt'),
                           ('Download editable CVAT XML',ROOT/'annotations/occlusion_0633/cvat_reference_draft.xml')]:
            st.download_button(label,path.read_bytes(),file_name=path.name)
    else:
        annotation=Path(job)/'annotations' if job_matches else None
        if annotation and (annotation/'annotations.json').exists():
            st.warning('Automatic draft labels, not human-verified ground truth. IDs and missed occluded players may be wrong.')
            index=st.slider('Automatic annotation frame',1,clip['frames'],1)
            st.image(str(annotation/'annotated_frames'/f'frame_{index:06d}.jpg'),width='stretch')
            st.caption(f'Annotated frames folder: {annotation / "annotated_frames"}')
            st.download_button('Download automatic MOT labels',(annotation/'automatic_reference_mot.txt').read_bytes(),file_name='automatic_reference_mot.txt')
        else:
            st.info('Run Annotate and compare to generate automatic labels in a separate annotations folder.')
    st.markdown('[SportsMOT annotation reference · ICCV 2023](https://github.com/MCG-NJU/SportsMOT)')
    with st.expander('Read our annotation procedure'):
        st.markdown((ROOT/'ANNOTATION_PROTOCOL.md').read_text(encoding='utf-8'))

with metric_tab:
    st.subheader('Tracking quality on the selected sequence')
    metrics_file=run/'preliminary_metrics.json' if run else None
    if not metrics_file or not metrics_file.exists():
        st.info('Accuracy metrics unavailable: this video has no attached reference annotations. Tracking output remains available after a run.')
    else:
        scores=read_json(metrics_file)
        st.warning(scores['status'])
        if scores.get('reference_kind')=='automatic':
            st.write('These scores measure agreement with independent YOLO11s + greedy motion draft labels. Teacher misses and identity mistakes affect every score. They do not establish tracking accuracy.')
        data=pd.DataFrame(scores['metrics'])
        data['Method']=data['method'].map(METHODS)
        selected=st.selectbox('Method details',list(METHODS.values()),index=3)
        row=data[data['Method']==selected].iloc[0]
        a,b,c,d=st.columns(4)
        a.metric('MOTA',f'{row.MOTA_percent:.2f}%')
        b.metric('AssA',f'{row.AssA_percent:.2f}%')
        c.metric('IDF1',f'{row.IDF1_percent:.2f}%')
        d.metric('ID switches',str(int(row.ID_switches)))
        table=data[['Method','MOTA_percent','HOTA_percent','AssA_percent','IDF1_percent','ID_switches','false_negatives','false_positives']].rename(
            columns={'MOTA_percent':'MOTA (%)','HOTA_percent':'HOTA (%)','AssA_percent':'AssA (%)','IDF1_percent':'IDF1 (%)',
                     'ID_switches':'ID switches','false_negatives':'FN','false_positives':'FP'})
        st.dataframe(table.style.format({name:'{:.2f}' for name in ['MOTA (%)','HOTA (%)','AssA (%)','IDF1 (%)']}),hide_index=True,width='stretch')
        chart=st.selectbox('Compare metric',['AssA (%)','MOTA (%)','HOTA (%)','IDF1 (%)'])
        st.bar_chart(table,x='Method',y=chart,color='#b8f36b',horizontal=True)
        adaptive=data[data.method=='adaptive_osnet'].iloc[0]
        fixed=data[data.method=='fixed_osnet'].iloc[0]
        if all(adaptive[k]==fixed[k] for k in ['MOTA_percent','AssA_percent','IDF1_percent']):
            st.info('Adaptive and fixed-weight variants have identical scores on this clip; an adaptation gain is not demonstrated.')
        with st.expander('Metric definitions and fair-comparison limits'):
            st.write('MOTA counts missed detections, false detections and identity switches. AssA is HOTA’s association component; IDF1 measures identity consistency. All methods receive identical cached detections and the same court-region filter.')
            st.write('Our method uses four selected first-frame identities (automatic when the camera is recognized); the baselines initialize automatically. Reference box interpolation affects localization scores. This is one selected development event, not a held-out test set.')
            st.dataframe(data[['Method','ID_switches_per_video_minute']].rename(columns={'ID_switches_per_video_minute':'Extrapolated switches / minute'}),hide_index=True,width='stretch')
            st.caption('The per-minute rate is extrapolated from this short clip. It is not a whole-match measurement.')
        st.download_button('Download comparison CSV',(run/'preliminary_metrics.csv').read_bytes(),file_name='preliminary_comparison.csv',mime='text/csv')

with method_tab:
    st.subheader('What the prototype implements')
    st.write('Consecutive frames → YOLO11 detections → OSNet appearance and predicted motion → court-aware cost matrix → Hungarian assignment → identity update and recovery.')
    st.write('Homography maps player box foot points to normalized court coordinates. A Kalman filter predicts motion on that court. Overlap and weak detection confidence reduce trust in current appearance crops; clean returning detections can use stored appearance memory.')
    st.write('Hungarian assignment includes an unmatched option. Missing players retain state for a bounded period, and appearance memory updates pause during overlap.')
    audit_file=run/'adaptive_osnet/association_audit.json' if run else None
    if audit_file and audit_file.exists():
        audit=pd.DataFrame(read_json(audit_file))
        player=st.selectbox('Inspect adaptive weights for',['P1','P2','P3','P4'])
        selected=audit[audit.player_id==int(player[-1])]
        weights=pd.DataFrame(selected.weights.tolist(),columns=['Motion','Appearance','Court distance','Court side'],index=selected.frame)
        weights.index.name='Local frame'
        st.line_chart(weights)
        st.caption('Weights shown for accepted observations. No points are invented for missing frames.')
        events=read_json(run/'adaptive_osnet/recovery_events.json')
        with st.expander('Inspect missing and reconnection events'):
            st.dataframe(pd.DataFrame(events),hide_index=True,width='stretch')
    st.caption('Experimental heuristics; weights are not learned or validated on held-out matches. OSNet-x0.25 uses the official Torchreid MSMT17 checkpoint.')
