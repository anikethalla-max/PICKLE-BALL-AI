import unittest
from pathlib import Path
import tempfile
import cv2
import numpy as np

from dashboard_backend import ROOT,PREPARED,calibration_valid,prepared_clip,save_upload,video_info


class DashboardBackendTests(unittest.TestCase):
    def test_prepared_clip_is_bound_to_its_reference(self):
        clip=save_upload(PREPARED.read_bytes())
        self.assertTrue(clip['prepared'])
        self.assertEqual(clip['frames'],105)
        self.assertEqual(clip['sha256'],prepared_clip()['sha256'])

    def test_other_clip_never_receives_prepared_reference(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            path=Path(folder)/'other.mp4'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30,(64,48))
            for i in range(6):
                writer.write(np.full((48,64,3),i*30,np.uint8))
            writer.release()
            clip=save_upload(path.read_bytes())
            self.assertFalse(clip['prepared'])
            self.assertEqual(clip['frames'],6)

    def test_invalid_video_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            path=Path(folder)/'invalid.mp4'
            path.write_bytes(b'not a video')
            with self.assertRaises(ValueError):
                video_info(path)

    def test_court_requires_convex_ordered_corners(self):
        self.assertTrue(calibration_valid([[100,100],[300,100],[400,400],[20,400]]))
        self.assertFalse(calibration_valid([[100,100],[400,400],[300,100],[20,400]]))
        self.assertFalse(calibration_valid([[100,100]]))


if __name__=='__main__':
    unittest.main()
