# Setup after cloning

This repository contains the local dashboard, tracking methods, tests, annotation
labels and documentation. Videos, model binaries, environments and generated
outputs are excluded from Git.

## Install dependencies (Windows / PowerShell)

Python 3.12 and uv are recommended.

```powershell
git clone https://github.com/anikethalla-max/PICKLE-BALL-AI.git
cd PICKLE-BALL-AI
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-dashboard.txt
.venv/Scripts/python.exe setup_osnet.py
```

Attributed OC-SORT and OSNet source is included under `third_party/`, with
upstream licenses and source hashes. `setup_osnet.py` downloads the official
pretrained OSNet checkpoint. YOLO11n weights download on first detector use if
missing. Installation and model downloads need internet; processing is local.

## Reconstruct the prepared demo

Supply the original video locally as `Videos/InPlayBalls_54min_vid.mp4`.
It is not distributed in this repository. The provided court calibration,
initial identities and reference labels refer to 06:33.000–06:36.500 of that
specific 1920×1080, 30 FPS source. Never apply those labels to an unrelated video.

```powershell
.venv/Scripts/python.exe prototype.py extract --start 393 --seconds 3.5 --output data/occlusion_0633/frames
.venv/Scripts/python.exe prepare_occlusion.py
.venv/Scripts/python.exe build_annotations.py
.venv/Scripts/python.exe cache_detections.py
.venv/Scripts/python.exe appearance_osnet.py
.venv/Scripts/python.exe compare_trackers.py
.venv/Scripts/python.exe evaluate_comparison.py
.venv/Scripts/python.exe launch_demo.py
```

Once these assets exist, `start_demo.cmd` launches the app and opens your
browser at http://127.0.0.1:8501. Keep its window open; Ctrl+C stops the app.
See `DASHBOARD_GUIDE.md` for the review presentation flow.

The dashboard can also process another short upload with new calibration and
player selections. Without its own reference labels, that upload has tracking
output but no accuracy scores. The current dashboard expects the prepared demo
assets to exist at startup; reconstruct them before starting it.

## Research status

The current comparison uses ByteTrack, official OC-SORT, BoT-SORT with actual
OSNet features, and an experimental adaptive court-aware Hungarian tracker.
A fixed-weight ablation is included. All receive the same detector cache and
court-region filter; our method also receives manually selected first-frame
identities, whereas the generic baselines initialize automatically.

The 105-frame, 420-box reference is assistant-drawn and interpolated, pending
independent human verification. It follows SportsMOT-inspired guidelines recorded
in `ANNOTATION_PROTOCOL.md`. It is not an official SportsMOT sequence. Scores on
this development event are preliminary; adaptive and fixed variants had identical
scores, so no accuracy gain caused by adaptation has been established.

The older `prototype.py` uses colour histograms and is separate from the newer
OSNet comparison. Association replay timing excludes detector and feature work;
it must not be described as end-to-end real-time throughput.

## Verification

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Video-dependent tests require the reconstructed local demo assets. Additional
Streamlit and browser integration scripts are in `tests/`. Browser QA uses local
Microsoft Edge and a running dashboard server.

Third-party license files apply to their respective code. Upstream Ultralytics
licensing also applies when using that dependency.
