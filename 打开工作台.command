#!/bin/zsh
# 打开 LocalMask。会在后台启动"看门程序"：工作台停了或卡住会自动重启，关掉这个终端窗口也没关系。
ROOT="${0:A:h}"
cd "$ROOT/workbench" || exit 1
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "还没有安装。请先按项目说明运行一次安装命令。"; exit 1
fi
PID=$(cat guardian.pid 2>/dev/null)
[ -n "$PID" ] && kill "$PID" 2>/dev/null
nohup /bin/zsh "$ROOT/workbench/guardian.sh" >/dev/null 2>&1 &!
for i in {1..60}; do
  curl -s -m 2 http://127.0.0.1:8765/api/bootstrap >/dev/null 2>&1 && break
  sleep 1
done
open http://127.0.0.1:8765
echo "LocalMask 已打开：http://127.0.0.1:8765 （这个窗口可以关掉）"
