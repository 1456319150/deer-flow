#!/usr/bin/env bash
# Quick start: ./run.sh
# Or with env vars: FEISHU_APP_ID=xxx FEISHU_APP_SECRET=yyy ./run.sh

set -euo pipefail

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)"

cd "$(dirname "$0")"


# Ensure ttadk is in PATH (adjust to your install location)
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

python3 gateway.py
