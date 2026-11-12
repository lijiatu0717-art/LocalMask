import json,subprocess,time,sys,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent;url='http://127.0.0.1:8765'
def running():
 try:
  with urllib.request.urlopen(url+'/api/bootstrap',timeout=2) as r:data=json.load(r)
  return isinstance(data.get('projects'),list) and 'token' in data
 except Exception:return False
if not running():
 log=(root/'server.log').open('a');subprocess.Popen([sys.executable,str(root/'server.py'),'--port','8765'],stdout=log,stderr=log,start_new_session=True)
 for _ in range(20):
  time.sleep(.3)
  if running():break
 else:raise SystemExit('工作台启动失败。请查看 workbench/server.log（8765端口可能被占用）。')
subprocess.run(['open',url],check=True)
print('LocalMask 已打开：'+url)
