# PickleMOT-4 demonstration annotation protocol

## Published reference

Cui, Y., Zeng, C., Zhao, X., Yang, Y., Wu, G., and Wang, L. (2023).
**SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes**.
Proceedings of the IEEE/CVF International Conference on Computer Vision,
pp. 9921–9931. See Section 3, “Annotation Pipeline”.

- Paper: https://openaccess.thecvf.com/content/ICCV2023/papers/Cui_SportsMOT_A_Large_Multi-Object_Tracking_Dataset_in_Multiple_Sports_Scenes_ICCV_2023_paper.pdf
- Authors' repository: https://github.com/MCG-NJU/SportsMOT
- Authors' dataset card: https://huggingface.co/datasets/MCG-NJU/SportsMOT/blob/main/README.md

SportsMOT annotates the full athlete, excluding equipment. Under partial
occlusion it estimates the hidden body extent while a body part remains visible.
Its guideline excludes players with half their torso outside the image and
requires one persistent ID per player throughout a clip. Its assisted annotation
workflow includes reviewing and refining proposed boxes.

## Our pickleball adaptation

We retain the source's body-box and identity policy, restrict targets to the four
players, and add qualitative occlusion flags. Rackets, balls, shadows, spectators
and officials are excluded. This does not make the clip part of SportsMOT:
SportsMOT covers other sports. We keep the source video's 1920×1080 resolution
and 30 FPS rather than converting it to SportsMOT's recording settings.

P1: near red top/red cap. P2: near black outfit. P3: far red top.
P4: far grey outfit/white cap. These remain fixed through the crossing.

## How this draft was actually made

The assistant inspected raw images and drew 22 sets of four keyframe boxes in
`annotation_keyframes.json`, without reading detector boxes or tracker IDs.
`build_annotations.py` interpolates the remaining frames. All 105 rendered
frames were examined through seven contact sheets; identified body-boundary
drift was corrected in the keyframes. The assistant reviewed identity continuity
and approximate body coverage, not pixel-perfect boundaries.

The tool was a local Python/OpenCV exporter, **not the SportsMOT authors' tool
and not a completed CVAT labeling session**. The resulting XML is supplied for
further editing in CVAT. Its track source “manual” describes drawn keyframes;
it does not mean a human annotator produced or approved them.

These annotations are an **assistant-drawn, visually reviewed reference draft**,
with interpolated geometry. They are not independently human-verified ground
truth. They are suitable for an honestly labeled preliminary pipeline demo;
research-grade evaluation requires checking every box in the original images,
reviewing occlusion boundaries, and independent approval. There is no claim of
matching SportsMOT's annotation quality or reproducing its benchmark.

## Files and conventions

- `annotations/occlusion_0633/annotated_frames/`: 105 images with drawn labels.
- `annotations/occlusion_0633/reference_draft_mot.txt`: 420 MOT rows.
- `annotations/occlusion_0633/annotations.json`: per-box identity, origin and occlusion flags.
- `annotations/occlusion_0633/cvat_reference_draft.xml`: editable track export.
- `annotations/occlusion_0633/review_sheets/`: all-frame contact sheets.

MOT frames are local and one based (1–105); CVAT XML frames are zero based.
Original source image names remain 11790–11894. Boxes use original image pixels.
MOT columns are `frame,id,left,top,width,height,mark,class,visibility`, with
mark=1 and class=1. Visibility=-1 explicitly means unknown, not a measured
visible fraction. JSON/XML store the qualitative occluded flag separately.
For this draft P1 is flagged partially occluded on local zero-based frames 18–30;
the exact boundaries require independent review. No player is fully invisible
or substantially outside the image in this selected sequence.

Once a human has verified or corrected the XML, retain that version separately
and rerun every method against the same corrected MOT export. Do not replace
reference labels with tracker output or change them to favor a method.
