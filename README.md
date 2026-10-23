# Smart Doubles AI — design-review prototype

**Cloning from GitHub?** Start with [GITHUB_SETUP.md](GITHUB_SETUP.md).
Videos, pretrained weights, generated images and outputs are kept local and are
not included in this repository. References below to already-installed packages
or existing demo outputs describe the original development workspace.

## Browser application

Double-click `start_demo.cmd`, then open http://127.0.0.1:8501.
The local Streamlit dashboard provides video upload, court/player selection,
tracking execution, reference-frame inspection, comparison playback and metrics.
See `DASHBOARD_GUIDE.md` for the presentation flow and execution modes.

## Latest working comparison and annotations

The current complete demonstration is described in `COMPARISON_GUIDE.md`.
It runs ByteTrack, official OC-SORT, BoT-SORT with actual pretrained OSNet,
and the proposed adaptive court-aware Hungarian tracker on shared detections.
A fixed-weight ablation is included. Separate annotated images are in
`annotations/occlusion_0633/annotated_frames/`, with editable CVAT/MOT labels.
Their published-method reference is SportsMOT (ICCV 2023), documented in
`ANNOTATION_PROTOCOL.md`. These are assistant-produced reference drafts pending
independent human verification; metrics are preliminary development results.

Everything below describes the earlier colour-histogram smoke test. It remains
available as a simpler first implementation, not the latest OSNet comparison.

For the selected partial-occlusion sequence at **06:33.0–06:36.5**, see
`OCCLUSION_DEMO.md`. It includes consecutive frames, court homography, Hungarian
assignment, a top-down court display and the ground-truth evaluation procedure.

Goal: demonstrate a short fixed-camera pickleball doubles clip passing through
frame extraction, player detection, court mapping, association and labeled output.
Existing source: `Videos/InPlayBalls_54min_vid.mp4`. Do not extract the full match.
The file was inspected: 1920×1080, 30 FPS, 12,810 frames (427 seconds), despite
the filename. A frame at 60 seconds shows all four players and the court.

## Quick start with the supplied manual calibration

The local environment and pretrained YOLO11n weights have been installed.
From PowerShell in `D:\PICKLEBALL-AI`:

```powershell
.venv/Scripts/python.exe prototype.py run --start 60 --seconds 3 --court reference/court_60s.json --seeds reference/seeds_60s.json --output outputs/my_demo
```

These seed points and court corners were manually selected from the 60-second
frame. They are approximate demo settings, not verified ground truth. The command
runs without GUI interaction. For another timestamp use the interactive procedure
below, or supply new manually verified seed points and court calibration.

## Initial smoke-test evidence

`outputs/smoke_demo/prototype.mp4` was generated from 60–63 seconds of the source.
The output was reopened and verified to contain 90 readable frames at 30 FPS.
The tracker returned four matched identities on 89 frames and three on the final
frame. The near-side grey-clothed player lacks an output box in that final frame.
This is a visible limitation to investigate, not proof of robust occlusion recovery.
Processing throughput excluding first-frame setup was approximately 9.9 FPS on
the installed CPU-only PyTorch build. Five synthetic association tests passed.
No ground-truth identity accuracy has been measured.

## Setup (PowerShell, from this directory)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

The first run downloads pretrained YOLO11n weights if unavailable. CPU inference
is the default. No training or GPU is required. Install size can be substantial
because Ultralytics depends on PyTorch.
`requirements-lock.txt` records the installed versions of the core packages.

## Procedure

1. Watch the source and select a 10–15 second continuous rally. All four players
   should be visible initially, with no camera cuts, zoom or court-end change.
   The example start of 60 seconds is a placeholder, not a verified rally.
2. Extract consecutive frames for your review evidence:

```powershell
.venv/Scripts/python.exe prototype.py extract --start 60 --seconds 15 --output data/frames
```

3. Calibrate using the same clip start:

```powershell
.venv/Scripts/python.exe prototype.py calibrate --start 60
```

Click the four **outer court baseline corners** in order: far-left, far-right,
near-right, near-left. Press Enter to save, R to reset, Esc to cancel. Coordinates
use original video pixels. Both ends must be visible. Recalibrate if the camera changes.

4. Run the proposed association prototype:

```powershell
.venv/Scripts/python.exe prototype.py run --start 60 --seconds 15
```

In the first-frame window click the actual four players in the desired P1, P2,
P3, P4 order, then Enter. This is **manual initialization**, which excludes
spectators rather than assuming the four largest detections are players. Press R
to retry. Choose a different start if four candidates are not detected.

Outputs: `outputs/demo/prototype.mp4`, screenshots, `tracks.csv`, `summary.json`.
Court coordinates are normalized: x left-to-right, y far-to-near. These are not
metric distances. Bottom-center of a box approximates foot position.

## What the prototype implements

YOLO11 person detections feed two branches: colour appearance features and
homography-based court positions. Each identity retains position, smoothed
velocity and appearance memory. Motion, appearance, court displacement and a
soft court-half penalty form a cost matrix. Weights shift toward appearance
after missed frames. Gated Hungarian assignment matches detections to P1–P4.
Unmatched identities retain memory for two seconds by default; missing players
are not drawn. Appearance memory is frozen when detected boxes overlap.

This is a simplified proposed-method demo, **not ByteTrack, OC-SORT or BoT-SORT**.
Those remain separate baselines. Histogram features are **not OSNet embeddings**.
Motion uses a simple velocity estimate, not a Kalman filter. The adaptive weights
and thresholds are initial heuristics, not learned or validated research results.
Expired IDs require restarting on a clear frame; long-gap recovery is not implemented.

## Review explanation

“Our detector locates players. Homography supplies court coordinates. Our cost
matrix compares motion, appearance and court compatibility. Hungarian assignment
selects one-to-one matches. We retain missing-player memory for short gaps.
This demo uses colour features as a placeholder for the planned OSNet model.”

Show input, extracted frames, calibration, labeled output and one manually checked
overlap sequence if present. Do not claim successful occlusion recovery unless
the specific player's identity is checked before and after that event.

## Next research steps

- Integrate pretrained OSNet with verified preprocessing and weights.
- Add a motion filter and explicit occlusion-risk and recovery logic.
- Compare ByteTrack, OC-SORT and BoT-SORT using identical detections.
- Annotate ground truth in CVAT and evaluate HOTA, AssA, IDF1 and ID switches.

Runtime logs are not accuracy measurements. Scene cuts, similar clothing,
merged detections and inaccurate box foot points can cause identity errors.
Court-side compatibility is a soft rally-specific prior; players can stand
outside painted lines and teams can change ends between rallies.
