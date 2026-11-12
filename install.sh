#!/bin/bash
# LocalMask 本地 AI 视频打码 · 一键安装（macOS / Apple 芯片）
# 用法：在「终端」里粘贴 README 上那一行命令。再运行一次 = 升级到最新版，已有视频和打码记录会保留。
set -eo pipefail

REPO="lijiatu0717-art/LocalMask"
APP="$HOME/LocalMask"

say()  { printf "\n\033[1;35m▸ %s\033[0m\n" "$1"; }
fail() { printf "\n\033[1;31m✗ %s\033[0m\n\n" "$1"; exit 1; }

[ "$(uname)" = "Darwin" ] || fail "目前只支持 Mac。"
[ "$(uname -m)" = "arm64" ] || fail "需要 Apple 芯片（M1/M2/M3/M4）的 Mac。"

echo ""
echo "  LocalMask 本地 AI 视频打码 · 安装"
echo "  全部在你自己的电脑上运行，视频不会上传到任何地方。"
echo "  第一次安装要下载约 1.5GB，大概 5–15 分钟，请保持联网。"

# 1. 程序本体（在项目文件夹里直接运行时，就用当前文件夹）
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -n "$HERE" ] && [ -f "$HERE/workbench/server.py" ]; then
  APP="$HERE"
else
  say "1/5 下载程序"
  TMP="$(mktemp -d)"
  curl -fsSL "https://github.com/$REPO/archive/refs/heads/main.tar.gz" | tar -xz -C "$TMP" \
    || fail "下载失败，请检查网络后再试一次。"
  mkdir -p "$APP"
  cp -R "$TMP"/*/. "$APP"/
  rm -rf "$TMP"
fi
cd "$APP"

# 2. 独立的 Python 运行环境（只装在这个文件夹里，不影响电脑上其他软件）
say "2/5 准备运行环境"
export UV_PYTHON_INSTALL_DIR="$APP/.uv/python" UV_CACHE_DIR="$APP/.uv/cache" UV_LINK_MODE=copy
UV="$APP/.uv/uv"
if [ ! -x "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$APP/.uv" INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null 2>&1 \
    || fail "运行环境下载失败，请检查网络后再试一次。"
fi
[ -x "$APP/.venv/bin/python" ] || "$UV" venv --python 3.12 "$APP/.venv" -q || fail "Python 准备失败，请再试一次。"
PY="$APP/.venv/bin/python"

say "3/5 安装组件（AI 抠图、视频处理）"
"$UV" pip install --python "$PY" -q -r requirements.txt \
  || "$UV" pip install --python "$PY" -q -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  || fail "组件安装失败，请检查网络后再试一次。"
"$PY" -c "from static_ffmpeg import run; run.get_or_fetch_platform_executables_else_raise()" >/dev/null 2>&1 \
  || fail "FFmpeg 下载失败，请检查网络后再试一次。"

say "4/5 下载 AI 模型（只在第一次下载）"
"$PY" - <<'EOF' || fail "模型下载失败，请检查网络后再试一次（再运行一次会接着下载）。"
import hashlib, os, sys, urllib.request
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from pathlib import Path
root = Path('.cache')
faces = root / 'face-models'; faces.mkdir(parents=True, exist_ok=True)
zoo = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/'
for name, url, sha in [
    ('yunet.onnx', zoo + 'face_detection_yunet/face_detection_yunet_2023mar.onnx', '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
    ('sface.onnx', zoo + 'face_recognition_sface/face_recognition_sface_2021dec.onnx', '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79'),
]:
    f = faces / name
    if f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest() == sha:
        continue
    print('  人脸模型', name)
    urllib.request.urlretrieve(url, f)
    if hashlib.sha256(f.read_bytes()).hexdigest() != sha:
        f.unlink(); sys.exit('人脸模型校验失败')
from huggingface_hub import snapshot_download
for repo, name in [('facebook/sam2.1-hiera-tiny', 'sam2.1-hiera-tiny'), ('IDEA-Research/grounding-dino-tiny', 'grounding-dino-tiny')]:
    dest = root / 'ai-models' / name
    if (dest / 'model.safetensors').is_file():
        continue
    print('  AI 模型', repo)
    for endpoint in [None, 'https://hf-mirror.com']:
        try:
            if endpoint: os.environ['HF_ENDPOINT'] = endpoint
            snapshot_download(repo, local_dir=str(dest), allow_patterns=['*.json', '*.txt', 'model.safetensors'], endpoint=endpoint)
            break
        except Exception as e:
            print('  换一个下载线路重试…', type(e).__name__)
    else:
        sys.exit('AI 模型下载失败')
EOF

say "5/5 检查并打开"
rm -rf "$APP/.uv/cache"
"$PY" -c "import sys; sys.path.insert(0,'workbench'); import cv2, torch, ai; s=ai.status(); assert s['ready'], s" \
  || fail "安装检查没通过，请把这个窗口截图发到项目的 Issues。"
chmod +x "$APP"/*.command "$APP/workbench/guardian.sh"
ln -sf "$APP/打开工作台.command" "$HOME/Desktop/LocalMask.command" 2>/dev/null || true

printf "\n\033[1;32m✓ 安装完成！\033[0m 以后双击桌面上的「LocalMask」就能打开。\n\n"
/bin/zsh "$APP/打开工作台.command"
