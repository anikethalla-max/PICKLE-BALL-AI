# Local browser demonstration

## Automatic upload demonstration

Restart Streamlit after updating the code (file watching is disabled). Run:

```powershell
cd D:\PICKLEBALL-AI
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Upload `data/occlusion_cross_court_0247/input_clip_3s.mp4` (source 02:47–02:50,
90 frames at 30 FPS). It contains partial near-player/far-player occlusion.
For this known camera, registration transfers the approximate court calibration
and YOLO selects four starting players. Other cameras require manual setup.
On Tracking click **Annotate and compare**. After completion, Annotations shows
automatic boxes and Metrics shows ByteTrack, OC-SORT, BoT-SORT + OSNet and our
adaptive method. The fixed-weight variant is also included as an ablation.

Automatic reference labels use a separate YOLO11s detector (first use downloads
its pretrained weights) and greedy court-position association, independently of
the evaluated trackers. Frames are saved in
`outputs/dashboard_jobs/<job>/annotations/annotated_frames`; MOT labels and
annotation provenance are alongside that folder. Hidden players are not assigned
invented boxes. These are draft annotations: misses and ID errors require review.

MOTA, HOTA, AssA, IDF1 and ID switches computed against automatic labels are
**agreement scores, not verified accuracy**. The teacher and student detectors
share a model family and COCO training, so their errors may be correlated.
SportsMOT-inspired MOT formatting is the annotation reference; this automated
procedure does not reproduce SportsMOT's human annotation and quality control.
Independent reviewed ground truth is required for research accuracy claims.

Double-click **start_demo.cmd** in `D:\PICKLEBALL-AI`. It opens your browser at
**http://127.0.0.1:8501** once the app is ready. Keep the command window open. Press Ctrl+C there to
stop the application. The app is local; it is not deployed or pushed to GitHub.

The current Python environment already contains its dependencies. For a fresh
environment install `requirements-dashboard.txt` and prepare the official model
files following `COMPARISON_GUIDE.md`.

## Review presentation flow

1. On the main page, use “Upload a short clip” and click **Browse files**, or
   choose “Prepared occlusion clip”. You can download the exact prepared clip
   using the download button below the input. The app recognizes its SHA-256 hash and
   loads its saved calibration, initial identities and draft annotation reference.
2. Show the overlap on **Input**. The sequence is source 06:33.000–06:36.500.
3. On **Annotations**, move the frame slider through approach, overlap and
   separation. Explain that these are assistant-drawn keyframes with interpolated
   reference boxes, pending independent human verification.
4. On **Tracking**, play the saved four-method comparison immediately, or click
   **Annotate and compare**. Cached mode actually reruns every association method using
   saved detector/OSNet features. Full mode reruns YOLO, OSNet and tracking too.
   The progress panel reports which stage is executing. CPU processing may take
   longer than video playback. Saved results remain available during a new run.
5. On **Metrics**, compare MOTA, HOTA, AssA, IDF1 and ID switches. The table is
   measured data from TrackEval, clearly marked preliminary. A completed rerun
   supplies its own outputs and measurements.
6. On **Method**, show the recorded adaptive weights and missing/reconnection
   events. State that fixed weights achieved the same scores on this sequence.

## Another uploaded clip

Upload an MP4/MOV/M4V of at most 30 seconds. On the first frame select court
baseline corners in the order far-left, far-right, near-right, near-left. Then
select one point inside each player's box in P1–P4 order. Clear either selection
to correct it. Run the full comparison when all eight points are ready.

New videos receive new calibration and new processing. They never receive the
prepared clip's reference labels or accuracy scores. Without reference labels,
the app shows tracking output and marks accuracy metrics unavailable. Choose a
clear first frame with all four players detected; a failed initialization is
reported in the run log. The new clip should have a fixed camera and no cuts.

Uploads remain in `data/dashboard_uploads/`. New runs are isolated under
`outputs/dashboard_jobs/`, preserving the original saved experiment. One local
tracking job runs at a time. Closing a browser tab does not stop a running job.

OpenCV videos are converted locally to H.264 MP4 for browser playback. The
application uses the FFmpeg binary bundled by imageio-ffmpeg. No video is sent
to an external service. Paper/code hyperlinks leave the local app only if clicked.

## Troubleshooting

If port 8501 is already in use, stop the previous local Streamlit process or run:

```powershell
.venv/Scripts/python.exe -m streamlit run dashboard.py --server.port 8502
```

Then visit http://127.0.0.1:8502. For an error during processing, expand the local
run log in Tracking. Results and logs stay under the individual job folder.
