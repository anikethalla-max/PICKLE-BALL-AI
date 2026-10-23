import unittest
import numpy as np
from adaptive_tracker import AdaptiveCourtTracker,adaptive_weights


def det(x,y,k,confidence=.95):
    feature=np.eye(4,dtype=np.float32)[k]
    return dict(p=np.float32([x,y]),box=np.float32([x*100,y*100,x*100+5,y*100+10]),
                feature=feature,confidence=confidence)


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.seeds=[det(.2,.2,0),det(.8,.2,1),det(.2,.8,2),det(.8,.8,3)]

    def test_weights_sum_to_one_and_appearance_falls_with_overlap(self):
        for gap in (1,5):
            clean=adaptive_weights(0,gap)
            overlap=adaptive_weights(1,gap)
            self.assertAlmostEqual(float(clean.sum()),1)
            self.assertAlmostEqual(float(overlap.sum()),1)
            self.assertLess(overlap[1],clean[1])

    def test_fixed_ablation_never_adapts(self):
        np.testing.assert_array_equal(adaptive_weights(0,1,True),adaptive_weights(1,5,True))

    def test_missing_then_returning_detection_reconnects_id(self):
        tracker=AdaptiveCourtTracker(self.seeds)
        self.assertEqual(len(tracker.update(self.seeds,0)),4)
        self.assertEqual(len(tracker.update(self.seeds[1:],1)),3)
        self.assertEqual([t['id'] for t,d in tracker.update(self.seeds[:1],2)],[1])
        self.assertTrue(any(e['type']=='reconnected' and e['player_id']==1 for e in tracker.events))

    def test_empty_frame_emits_no_hallucinated_boxes(self):
        tracker=AdaptiveCourtTracker(self.seeds)
        self.assertEqual(tracker.update([],1),[])

    def test_low_confidence_does_not_update_appearance_memory(self):
        tracker=AdaptiveCourtTracker(self.seeds)
        tracker.update([det(.2,.2,0,.15)],1)
        self.assertEqual(len(tracker.tracks[0]['gallery']),1)


if __name__=='__main__':
    unittest.main()
