# Selected occlusion demonstration

The source was reviewed through timestamped contact sheets at two-second
intervals. Candidate sequences were then inspected at quarter-second intervals.
This is manual visual selection, not an exhaustive detection of every event.

Selected segment: **06:33.000 to 06:36.500 (end exclusive)**.
At 30 FPS this is 105 consecutive frames, source frame numbers 11790–11894
(zero based). Last frame time: 06:36.4667.

The near-side player wearing black crosses in front of the near-side player
wearing red at approximately **06:33.700–06:34.100**. This is partial occlusion,
not a claim of complete disappearance. The segment includes approach and separation.
The exact occlusion boundaries still need frame-by-frame annotation.

The 105-frame tracking run completed and its saved video was reopened. Four
identities are output on 97 frames and three on eight frames, at source times
06:33.667–06:33.900 inclusive. This is an observed output gap around overlap,
not a ground-truth false-negative count or an ID-switch measurement. The detailed
audit is in `outputs/occlusion_0633/observation_audit.json`.

## Outputs

- `data/occlusion_0633/input_clip.mp4`: unannotated short clip for review/CVAT.
- `data/occlusion_0633/frames/`: every consecutive image, without subsampling.
- `data/occlusion_0633/frames_for_annotation.zip`: images packaged for annotation.
- `data/occlusion_0633/frame_mapping.csv`: local evaluation frames to source times.
- `outputs/occlusion_0633/prototype.mp4`: proposed-method prototype output.
- `outputs/occlusion_0633/tracks.csv`: predicted identities and image/court positions.
- `outputs/occlusion_0633/homography.json`: actual image-to-court matrix used.

The calibration was manually checked on the selected first frame. The four
approximate baseline corners map to normalized (0,0), (1,0), (1,1), (0,1).
It is a demonstration calibration, not surveyed metric ground truth.

## Reproduce

```powershell
.venv/Scripts/python.exe prototype.py extract --start 393 --seconds 3.5 --output data/occlusion_0633/frames
.venv/Scripts/python.exe prototype.py run --start 393 --seconds 3.5 --court reference/court_393s.json --seeds reference/seeds_393s.json --output outputs/occlusion_0633
.venv/Scripts/python.exe prepare_occlusion.py
.venv/Scripts/python.exe evaluate_tracking.py --run-dir outputs/occlusion_0633
```

## Association procedure

The tracker compares existing P1–P4 identities to current player detections.
Predicted motion, colour appearance, court displacement and court-half penalties
produce a cost matrix. Implausible pairs are gated. Hungarian assignment selects
the lowest-cost one-to-one assignment; high-cost matches are rejected. Missing
identities retain state for a bounded period. This implementation uses heuristic
weights and colour histograms, not the planned OSNet appearance model.

## Ground truth needed for MOTA, AssA and ID-switch frequency

Annotate the raw clip or image ZIP in CVAT with four persistent player tracks:

| Identity | Player in this segment |
|---|---|
| P1 / ID 1 | Near-side red top and red cap |
| P2 / ID 2 | Near-side black outfit |
| P3 / ID 3 | Far-side red top |
| P4 / ID 4 | Far-side grey outfit and white cap |

Use a single `player` label. Draw full-body boxes consistently, including
estimated hidden portions during partial occlusion. Mark occlusion explicitly.
Verify interpolated boxes on every frame, particularly during overlap. Keep
identity tied to the person, not the left/right position. Spectators and officials
are outside this four-player evaluation. Declare a consistent policy for any
fully invisible player; do not invent a box without defensible evidence.

Export MOT ground truth with local frames 1–105 and nine columns:
`frame,id,left,top,width,height,mark,class,visibility`.
For evaluated players use mark=1, class=1. Map CVAT track identifiers to 1–4
if necessary. Annotation numbering must match `frame_mapping.csv`. Keep this
independent of the tracker predictions. Detector or tracker output may help
annotation, but cannot become ground truth without manual correction.

The evaluation helper uses official TrackEval:
https://github.com/JonathonLuiten/TrackEval
MOT conversion and the missing-ground-truth path have been exercised locally.
TrackEval is an optional dependency and has not yet been installed or run on
this event because ground truth is unavailable.

After annotations are available:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-evaluation.txt
.venv/Scripts/python.exe evaluate_tracking.py --run-dir outputs/occlusion_0633 --gt data/occlusion_0633/gt.txt --evaluate
```

Reports include MOTA, HOTA, AssA, IDF1, raw ID switches, and switches per video
minute. AssA is taken from HOTA's association component, averaged over its
localization thresholds. A stable displayed label alone is not proof of high AssA.
MOTA includes false positives, false negatives and identity switches. The helper
will not compute accuracy without ground truth. A rate extrapolated from a
3.5-second event is not an estimate of whole-match performance.
