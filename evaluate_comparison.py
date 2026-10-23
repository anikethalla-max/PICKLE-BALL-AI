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
    parser.add_argument('--reference',default='annotations/occlusion_0633/reference_draft_mot.txt')
    parser.add_argument('--reference-kind',choices=['draft','automatic'],default='draft')
    args=parser.parse_args()
    root=Path(args.run_root).resolve()
    experiment=json.loads((root/'experiment.json').read_text())
    methods=experiment['methods']
    summary=json.loads((root/methods[0]/'summary.json').read_text())
    frames=summary['frames']
    seconds=frames/summary['source_fps']
    evaluation=root/'evaluation'
    gt=evaluation/'gt'/'occlusion'/'gt'
    gt.mkdir(parents=True,exist_ok=True)
    reference=Path(args.reference)
    reference_boxes=sum(bool(line.strip()) for line in reference.read_text().splitlines())
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
        OUTPUT_FOLDER=str(evaluation/'results'),TRACKERS_TO_EVAL=methods,SEQ_INFO={'occlusion':frames},
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
            ID_switches_per_video_minute=float(clear['IDSW'])/seconds*60)
        rows.append(row)
    result=dict(status=('AUTOMATIC REFERENCE AGREEMENT ONLY: not verified accuracy' if args.reference_kind=='automatic' else 'PRELIMINARY DEMO ONLY: reference draft; human verification pending'),
        reference_kind=args.reference_kind,frames=frames,seconds=seconds,reference=str(reference),reference_boxes=reference_boxes,
        evaluation='TrackEval HOTA/CLEAR/Identity; CLEAR IoU threshold 0.5; no GT preprocessing; AssA averaged over HOTA thresholds',
        development_scope='Single deliberately selected occlusion event. No held-out test set; no statistical generalization.',
        initialization_caveat=experiment['initialization'],
        rate_caveat=f'Per-minute IDSW is extrapolated from {seconds:g} seconds',
        metrics=rows)
    (root/'preliminary_metrics.json').write_text(json.dumps(result,indent=2))
    with (root/'preliminary_metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root/'PRELIMINARY_RESULTS.md').write_text(result['status']+'\n\n'+json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
