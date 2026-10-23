"""Synthetic association checks, independent of detector weights or video UI."""
import unittest

import numpy as np

from prototype import CourtTracker


def detection(x, y, feature):
    return dict(p=np.array([x, y]), a=np.array(feature, dtype=float),
                foot=np.array([x*100, y*100]), overlap=False)


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.seeds = [detection(.2,.2,[1,0,0,0]), detection(.8,.2,[0,1,0,0]),
                      detection(.2,.8,[0,0,1,0]), detection(.8,.8,[0,0,0,1])]

    def test_detection_order_does_not_change_identity(self):
        tracker = CourtTracker(self.seeds, 0, 30)
        matches = tracker.update(list(reversed(self.seeds)), 1)
        self.assertEqual(len(matches), 4)
        for track, det in matches:
            np.testing.assert_array_equal(det['a'], self.seeds[track['id']-1]['a'])

    def test_missing_player_retains_identity_when_returning(self):
        tracker = CourtTracker(self.seeds, 0, 30)
        self.assertEqual(len(tracker.update(self.seeds[1:], 1)), 3)
        matches = tracker.update([self.seeds[0]], 3)
        self.assertEqual([track['id'] for track, _ in matches], [1])

    def test_impossible_jump_is_unmatched(self):
        tracker = CourtTracker(self.seeds, 0, 30)
        self.assertEqual(tracker.update([detection(4,4,[1,0,0,0])], 1), [])

    def test_expired_identity_is_not_silently_reassigned(self):
        tracker = CourtTracker(self.seeds, 0, 2)
        self.assertEqual(tracker.update(self.seeds, 3), [])

    def test_empty_frame_is_supported(self):
        tracker = CourtTracker(self.seeds, 0, 30)
        self.assertEqual(tracker.update([], 1), [])


if __name__ == '__main__':
    unittest.main()
