"""Exercise Streamlit UI widgets and submit one real cached tracking job."""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard.py'),default_timeout=90).run()
assert not app.exception,[x.message for x in app.exception]
assert app.get('file_uploader'), 'Upload control must be visible on first load'
next(x for x in app.radio if x.label=='Video input').set_value('Prepared occlusion clip').run()
assert len(app.tabs)==5
assert any(x.label=='MOTA' for x in app.metric)
next(x for x in app.selectbox if x.label=='Method details').set_value('ByteTrack').run()
assert not app.exception
assert next(x for x in app.metric if x.label=='MOTA').value=='96.43%'
next(x for x in app.slider if x.label=='Reference frame').set_value(45).run()
assert not app.exception
next(x for x in app.radio if x.label=='Video input').set_value('Upload a short clip').run()
assert not app.exception
assert not app.metric  # No clip selected: never show somebody else's scores.
next(x for x in app.radio if x.label=='Video input').set_value('Prepared occlusion clip').run()
next(x for x in app.button if x.label=='Run comparison').click().run()
assert not app.exception,[x.message for x in app.exception]
job=app.session_state['active_job']
Path('outputs/dashboard_ui_test.json').write_text(json.dumps({'job':job,'checks':[
    'five views render','metric method selector','annotation frame slider',
    'upload mode clears unrelated scores','run button starts actual cached tracking job']},indent=2))
print('Dashboard integration checks passed; real job:',job,flush=True)
