import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from dashboard_backend import ROOT,PREPARED


class DashboardInputTests(unittest.TestCase):
    def test_upload_is_visible_on_first_load(self):
        app=AppTest.from_file(str(ROOT/'dashboard.py'),default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(next(r for r in app.radio if r.label=='Video input').value,'Upload a short clip')
        self.assertEqual(len(app.get('file_uploader')),1)
        self.assertEqual(len(app.main.get('file_uploader')),1)

    def test_missing_prepared_clip_keeps_upload_accessible(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            with patch('dashboard_backend.PREPARED',Path(folder)/'missing.mp4'):
                app=AppTest.from_file(str(ROOT/'dashboard.py'),default_timeout=60).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.main.get('file_uploader')),1)
                next(r for r in app.radio if r.label=='Video input').set_value('Prepared occlusion clip').run()
                self.assertFalse(app.exception)
                self.assertTrue(any('prepared demo clip is missing' in x.value for x in app.warning))


if __name__=='__main__':
    unittest.main()
