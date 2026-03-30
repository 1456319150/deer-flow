#!/usr/bin/env bash
# Quick start: ./run.sh

set -euo pipefail

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

pyenv local 3.12.4

if [ ! -d ".venv" ]; then
    echo ">>> 首次运行，创建虚拟环境..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo ">>> 安装依赖..."
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

echo "Python: $(python3 --version)"
echo "Path:   $(which python3)"
echo ">>> 前台启动 gateway.py"

exec python3 gateway.py
