"""User launcher: run the local app and open the default browser when ready."""
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

root=Path(__file__).resolve().parent
url='http://127.0.0.1:8501'
process=subprocess.Popen([sys.executable,'-m','streamlit','run',str(root/'dashboard.py'),
    '--server.address','127.0.0.1','--server.port','8501','--server.headless','true'],cwd=root)
try:
    for attempt in range(60):
        if process.poll() is not None:
            raise RuntimeError('Dashboard could not start. Check the message above, including whether port 8501 is in use.')
        try:
            with urllib.request.urlopen(url+'/_stcore/health',timeout=1) as response:
                if response.status==200:
                    break
        except OSError:
            time.sleep(.5)
    else:
        raise RuntimeError('Dashboard did not become ready in 30 seconds.')
    print('\nDashboard ready:',url,'\nKeep this window open; press Ctrl+C to stop.\n',flush=True)
    webbrowser.open(url)
    process.wait()
except KeyboardInterrupt:
    print('\nStopping the local dashboard.',flush=True)
finally:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
