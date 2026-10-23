"""Headless Edge DevTools check against the local running dashboard."""
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from websockets.sync.client import connect

root=Path(__file__).resolve().parents[1]
browser=Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')
process=subprocess.Popen([str(browser),'--headless=new','--disable-gpu','--no-first-run',
    '--no-default-browser-check','--disable-background-networking','--remote-debugging-port=9223',
    '--user-data-dir='+str(root/'outputs/edge_cdp_profile'),'about:blank'],
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
try:
    for attempt in range(40):
        try:
            request=urllib.request.Request('http://127.0.0.1:9223/json/new?http://127.0.0.1:8501',method='PUT')
            with urllib.request.urlopen(request,timeout=2) as response:
                target=json.load(response)
            break
        except OSError:
            time.sleep(.5)
    else:
        raise RuntimeError('Headless browser did not start')
    with connect(target['webSocketDebuggerUrl'],open_timeout=10) as socket:
        identifier=0
        def command(method,params=None):
            global identifier
            identifier+=1
            current=identifier
            socket.send(json.dumps(dict(id=current,method=method,params=params or {})))
            while True:
                message=json.loads(socket.recv(timeout=20))
                if message.get('id')==current:
                    if 'error' in message:
                        raise RuntimeError(message['error'])
                    return message.get('result',{})
        def evaluate(expression):
            return command('Runtime.evaluate',dict(expression=expression,returnByValue=True)).get('result',{}).get('value')
        command('Page.enable')
        command('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1100,deviceScaleFactor=1,mobile=False))
        for attempt in range(80):
            body=evaluate('document.body.innerText') or ''
            if 'Player identities through occlusion' in body and 'Start with the crossing' in body:
                break
            time.sleep(.5)
        else:
            raise RuntimeError('Dashboard did not finish rendering: '+str(body)[:300])
        time.sleep(1)
        for label,match in [('input','1 · Input'),('tracking','2 · Tracking'),('metrics','4 · Metrics')]:
            clicked=evaluate(f"Array.from(document.querySelectorAll('[role=tab]')).find(x=>x.innerText.includes({json.dumps(match)}))?.click(); true")
            time.sleep(1)
            result=command('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
            (root/'outputs'/f'dashboard_{label}.png').write_bytes(base64.b64decode(result['data']))
        video=evaluate("Array.from(document.querySelectorAll('video')).map(v=>({ready:v.readyState,error:v.error?.message||null}))")
        report=dict(rendered=True,tabs=evaluate("Array.from(document.querySelectorAll('[role=tab]')).map(x=>x.innerText)"),videos=video)
        (root/'outputs/dashboard_browser_check.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        command('Browser.close')
finally:
    if process.poll() is None:
        process.terminate()
