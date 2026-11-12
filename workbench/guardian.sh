#!/bin/zsh
# 看门程序：每 5 秒检查一次工作台。没在运行就启动；在运行但连续约 20 秒没反应（卡死）就结束它再重启。
ROOT="${0:A:h:h}"
cd "$ROOT/workbench" || exit 1
echo $$ > guardian.pid
fails=0
codestamp() { stat -f %m server.py ai.py 2>/dev/null | tr "\n" "-"; }
last=$(codestamp)
while true; do
  now=$(codestamp)
  if [ "$now" != "$last" ]; then
    last=$now
    P=$(/usr/sbin/lsof -ti tcp:8765 -sTCP:LISTEN 2>/dev/null)
    if [ -n "$P" ]; then echo "[guardian $(date '+%F %T')] 程序更新了，自动重启工作台" >> server.log; sleep 3; kill $P 2>/dev/null; sleep 2; fi
  fi
  if /usr/bin/curl -s -m 4 http://127.0.0.1:8765/api/bootstrap >/dev/null 2>&1; then
    fails=0
  else
    PIDS=$(/usr/sbin/lsof -ti tcp:8765 -sTCP:LISTEN 2>/dev/null)
    if [ -n "$PIDS" ]; then
      fails=$((fails+1))
      if [ $fails -ge 4 ]; then
        echo "[guardian $(date '+%F %T')] 工作台卡住没有响应，结束进程 $PIDS 并重启" >> server.log
        kill $PIDS 2>/dev/null; sleep 2; kill -9 $PIDS 2>/dev/null
        fails=0
      fi
    else
      fails=0
      echo "[guardian $(date '+%F %T')] 工作台不在运行，重新启动" >> server.log
      "$ROOT/.venv/bin/python" server.py --port 8765 < /dev/null >> server.log 2>&1 &!
      sleep 8
    fi
  fi
  sleep 5
done
