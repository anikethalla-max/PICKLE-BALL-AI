# Smart Doubles AI: occlusion review demonstration

The implemented demonstration covers **06:33.000–06:36.500**, 105 consecutive
frames from the source video. It contains a near-side player crossing and
partial occlusion around 06:33.7–06:34.1.

## Open these deliverables

1. `annotations/occlusion_0633/annotated_frames/`: 105 independently drawn/interpolated reference images.
2. `annotations/occlusion_0633/cvat_reference_draft.xml`: editable annotation tracks.
3. `outputs/comparison_0633/four_method_comparison.mp4`: four methods side by side.
4. `outputs/comparison_0633/PRELIMINARY_RESULTS.md`: TrackEval comparison table.
5. `ANNOTATION_PROTOCOL.md`: published reference, labeling rules and provenance.

The word “independently” here means drawn from raw images without copying model
predictions. It does **not** mean independently checked by a second human.
The reference is an assistant-produced draft with interpolation. All 105 images
were reviewed in contact sheets, with corrections recorded. Human verification
is still pending. Metrics must retain that qualification.

## What is now implemented

- One shared YOLO11n detection cache (960-pixel inference size, confidence floor .10).
- The same expanded court region applied before every tracker.
- ByteTrack from installed Ultralytics, official Python OC-SORT, and BoT-SORT
  from Ultralytics using sparse optical-flow camera-motion compensation and
  external pretrained OSNet features.
- Real OSNet-x0.25 features from the official Torchreid MSMT17 checkpoint.
  Input is RGB, resized to 256×128, ImageNet-normalized and L2-normalized.
- A proposed-method prototype with court homography, Kalman court prediction,
  OSNet appearance memory, overlap/confidence-dependent costs, soft court-side
  penalties and Hungarian assignment with explicit unmatched alternatives.
- Missing, occlusion-risk and reconnect events; appearance memory updates freeze
  during high risk. Predictions never manufacture a visible box for a missed detection.
- A fixed-weight ablation sharing the same prediction, memory and recovery code.
- Official TrackEval HOTA, AssA, MOTA, IDF1 and ID switches against the reference draft.

The baseline tracker IDs are preserved. Our method's initial P1–P4 seeds are
manually supplied in the first frame, whereas baselines initialize automatically.
This is a meaningful difference to disclose. Court mapping is drawn on baseline
videos for visualization; their association rules do not receive our court costs.

## Reproduce from the local environment

The local model files and dependencies have already been installed. Run from
`D:\PICKLEBALL-AI` in PowerShell:

```powershell
.venv/Scripts/python.exe build_annotations.py
.venv/Scripts/python.exe compare_trackers.py
.venv/Scripts/python.exe evaluate_comparison.py
```

Recompute detections and embeddings only if the input, detector settings or
feature settings change:

```powershell
.venv/Scripts/python.exe cache_detections.py
.venv/Scripts/python.exe appearance_osnet.py
```

For a fresh machine install `requirements-comparison.txt`, then run
`setup_research_dependencies.py` and `setup_osnet.py`. The third-party source
files and checkpoints have URLs and SHA-256 hashes in their PROVENANCE files.
The OC-SORT snapshot was obtained from master because GitHub's commit API was
rate-limited; recorded file hashes identify the exact snapshot used here.

## Explain adaptation during the review

Each track predicts a court position using a Kalman filter. The association
matrix has normalized motion, OSNet appearance, court displacement and court-side
costs. With more box overlap or lower detection confidence, the prototype reduces
the current appearance crop's contribution. For a clean detection returning after
a gap, it relies more on stored appearance. Memory updates are suppressed when
overlap could contaminate them. Hungarian assignment selects one-to-one matches,
including an option to leave an identity unmatched. The event log records when
the same stored identity is reconnected.

These are explicit experimental heuristics, not learned or proven optimal rules.
The adaptive and fixed variants produced identical metrics on this event. This
clip therefore demonstrates execution of adaptation but **does not demonstrate
an accuracy gain caused by adaptation**. More events and a held-out match split
are needed. Do not alter labels or tune on this sequence to manufacture a gain.

## Annotation reference

Cui et al., *SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports
Scenes*, ICCV 2023. See the full citation and adaptations in `ANNOTATION_PROTOCOL.md`.
This pickleball example is not an official SportsMOT sequence or benchmark result.

## Evaluation limits

The printed metrics are preliminary scores on an assistant-annotated development
event, not final accuracy. The full-body reference boxes include limbs, and box
interpolation can change IoU-based scores. Independent human correction is the
next validation step. A 3.5-second event is too short to characterize match-wide
identity-switch frequency. Raw switches are more interpretable here than an
extrapolated per-minute rate. Association replay timings exclude detection,
embedding extraction, initialization and rendering and must not be called
end-to-end real-time performance.
