"""Official TrackEval metrics against the explicitly labeled reference draft.

Never interprets these development-event numbers as independently validated
research results. Human verification of reference annotations remains pending.
"""
import csv
import json
import shutil
import argparse
from pathlib import Path
import numpy as np
import trackeval


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',default='outputs/comparison_0633')
    args=parser.parse_args()
    root=Path(args.run_root).resolve()
    experiment=json.loads((root/'experiment.json').read_text())
    methods=experiment['methods']
    evaluation=root/'evaluation'
    gt=evaluation/'gt'/'occlusion'/'gt'
    gt.mkdir(parents=True,exist_ok=True)
    reference=Path('annotations/occlusion_0633/reference_draft_mot.txt')
    shutil.copyfile(reference,gt/'gt.txt')
    for method in methods:
        target=evaluation/'trackers'/method/'data'
        target.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/method/'predictions_mot.txt',target/'occlusion.txt')
    config=trackeval.Evaluator.get_default_eval_config()
    config.update(USE_PARALLEL=False,PLOT_CURVES=False,PRINT_CONFIG=False,
                  PRINT_ONLY_COMBINED=True,DISPLAY_LESS_PROGRESS=True)
    dataset_config=trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(GT_FOLDER=str(evaluation/'gt'),TRACKERS_FOLDER=str(evaluation/'trackers'),
        OUTPUT_FOLDER=str(evaluation/'results'),TRACKERS_TO_EVAL=methods,SEQ_INFO={'occlusion':105},
        SKIP_SPLIT_FOL=True,DO_PREPROC=False,PRINT_CONFIG=False)
    dataset=trackeval.datasets.MotChallenge2DBox(dataset_config)
    results,messages=trackeval.Evaluator(config).evaluate([dataset],
        [trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()])
    rows=[]
    for method in methods:
        scores=results[dataset.get_name()][method]['occlusion']['pedestrian']
        clear=scores['CLEAR']
        row=dict(method=method,MOTA_percent=float(clear['MOTA'])*100,
            HOTA_percent=float(np.mean(scores['HOTA']['HOTA']))*100,
            AssA_percent=float(np.mean(scores['HOTA']['AssA']))*100,
            IDF1_percent=float(scores['Identity']['IDF1'])*100,
            ID_switches=int(clear['IDSW']),false_negatives=int(clear['CLR_FN']),
            false_positives=int(clear['CLR_FP']),
            ID_switches_per_video_minute=float(clear['IDSW'])/(105/30)*60)
        rows.append(row)
    result=dict(status='PRELIMINARY DEMO ONLY: assistant-drawn reference draft; human verification pending',
        frames=105,seconds=3.5,reference=str(reference),reference_boxes=420,
        evaluation='TrackEval HOTA/CLEAR/Identity; CLEAR IoU threshold 0.5; no GT preprocessing; AssA averaged over HOTA thresholds',
        development_scope='Single deliberately selected occlusion event. No held-out test set; no statistical generalization.',
        initialization_caveat=experiment['initialization'],
        rate_caveat='Per-minute IDSW is extrapolated from 3.5 seconds, not observed over a full minute',
        metrics=rows)
    (root/'preliminary_metrics.json').write_text(json.dumps(result,indent=2))
    with (root/'preliminary_metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text=['# Preliminary comparison on an assistant-annotated reference draft','',
          '**Human verification pending. These are development-demo measurements, not final research results.**','',
          '| Method | MOTA (%) | HOTA (%) | AssA (%) | IDF1 (%) | IDSW | FN | FP |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f'| {row["method"]} | {row["MOTA_percent"]:.2f} | {row["HOTA_percent"]:.2f} | {row["AssA_percent"]:.2f} | {row["IDF1_percent"]:.2f} | {row["ID_switches"]} | {row["false_negatives"]} | {row["false_positives"]} |')
    text+=['','105 frames at 30 FPS, 06:33.000–06:36.500. Reference: 420 full-athlete boxes, drawn on raw keyframes and interpolated independently of detector predictions. Visibility fractions are unknown; partial occlusion is labeled separately.',
           '',experiment['initialization'],experiment['limitation'],
           '','All methods use the same cached detector stream and expanded court ROI. ByteTrack and OC-SORT do not use OSNet. BoT-SORT uses external OSNet features and sparse optical-flow GMC. The adaptive and fixed variants use the same OSNet memory and recovery logic; only the association weights differ.',
           '','Zero ID switches on this short event cannot establish superior identity preservation. Missing detections and box localization also affect these metrics. Parameters were not tuned on held-out data.',
           '','IDSW/minute = IDSW × 60 / 3.5. This is an extrapolated event rate, not a whole-match estimate.']
    (root/'PRELIMINARY_RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
