"""Prepare MOT files and evaluate only with independent ground truth.

Uses official TrackEval HOTA (including AssA), CLEAR (MOTA/IDSW), Identity.
No tracker-generated labels are treated as ground truth.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='outputs/occlusion_0633')
    parser.add_argument('--gt', help='Manually verified MOT gt.txt; local frame indices 1..N')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    meta = json.loads((run/'summary.json').read_text())
    sequence = 'occlusion'
    root = run/'evaluation'
    gt_dir = root/'gt'/sequence/'gt'
    tracker_dir = root/'trackers'/'court_hungarian'/'data'
    gt_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)
    count = int(meta['frames'])
    start = round(meta['start_seconds']*meta['source_fps'])
    with (run/'tracks.csv').open(newline='') as source, (tracker_dir/f'{sequence}.txt').open('w',newline='') as dest:
        writer = csv.writer(dest)
        for row in csv.DictReader(source):
            frame = int(row['source_frame'])-start+1
            if not 1 <= frame <= count:
                raise ValueError('Prediction frame lies outside sequence')
            x1,y1,x2,y2 = [float(row[k]) for k in ('x1','y1','x2','y2')]
            writer.writerow([frame,int(row['player_id']),x1,y1,x2-x1,y2-y1,
                             row['confidence'],-1,-1,-1])
    (gt_dir.parent/'seqinfo.ini').write_text(
        f'[Sequence]\nname={sequence}\nimDir=img1\nframeRate={meta["source_fps"]}\n'
        f'seqLength={count}\nimWidth={meta["width"]}\nimHeight={meta["height"]}\nimExt=.jpg\n')
    if args.gt:
        seen = set()
        with Path(args.gt).open(newline='') as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                if len(row) < 9:
                    raise ValueError('GT requires frame,id,x,y,width,height,mark,class,visibility')
                frame,identity = int(row[0]),int(row[1])
                if not 1 <= frame <= count or not 1 <= identity <= 4:
                    raise ValueError('GT frame must be local 1..N and IDs 1..4')
                if (frame,identity) in seen:
                    raise ValueError('Duplicate GT identity in one frame')
                if float(row[4]) <= 0 or float(row[5]) <= 0 or int(row[6]) != 1 or int(row[7]) != 1:
                    raise ValueError('GT requires positive boxes, mark=1 and person class=1')
                seen.add((frame,identity))
        if not seen or len({f for f,_ in seen}) != count:
            raise ValueError('GT must cover every sequence frame with at least one annotated player')
        target = gt_dir/'gt.txt'
        if Path(args.gt).resolve() != target.resolve():
            shutil.copyfile(args.gt, target)
    if not args.evaluate:
        print(f'Prepared predictions and sequence metadata: {root}')
        print('Metrics pending independent, manually verified ground truth.')
        return
    if not (gt_dir/'gt.txt').exists():
        raise ValueError('No ground truth. Supply --gt path/to/gt.txt; predictions cannot establish accuracy.')
    import numpy as np
    import trackeval
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(USE_PARALLEL=False, PLOT_CURVES=False, PRINT_CONFIG=False)
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(GT_FOLDER=str(root/'gt'), TRACKERS_FOLDER=str(root/'trackers'),
                          OUTPUT_FOLDER=str(root/'results'), TRACKERS_TO_EVAL=['court_hungarian'],
                          SEQ_INFO={sequence:count}, SKIP_SPLIT_FOL=True, DO_PREPROC=False,
                          PRINT_CONFIG=False)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()]
    results,_ = trackeval.Evaluator(eval_config).evaluate([dataset],metrics)
    scores = results[dataset.get_name()]['court_hungarian'][sequence]['pedestrian']
    duration = count/meta['source_fps']
    summary = dict(MOTA_percent=float(scores['CLEAR']['MOTA'])*100,
                   AssA_percent=float(np.mean(scores['HOTA']['AssA']))*100,
                   HOTA_percent=float(np.mean(scores['HOTA']['HOTA']))*100,
                   IDF1_percent=float(scores['Identity']['IDF1'])*100,
                   ID_switches=int(scores['CLEAR']['IDSW']),
                   ID_switches_per_video_minute=float(scores['CLEAR']['IDSW'])/duration*60,
                   evaluated_seconds=duration,
                   note='Short event-only evaluation; per-minute rate is extrapolated, not whole-match performance')
    (root/'metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
