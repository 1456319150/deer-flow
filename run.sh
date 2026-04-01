#!/usr/bin/env bash
# Quick start: ./run.sh

set -euo pipefail

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init -)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

pyenv local 3.12.4

REQUIREMENTS_HASH_FILE=".venv/.requirements.sha256"
CURRENT_REQUIREMENTS_HASH="$(shasum -a 256 requirements.txt | awk '{print $1}')"

if [ ! -d ".venv" ]; then
    echo ">>> 首次运行，创建虚拟环境..."
    python3 -m venv .venv
fi

source .venv/bin/activate

if [ ! -f "$REQUIREMENTS_HASH_FILE" ] || [ "$(cat "$REQUIREMENTS_HASH_FILE")" != "$CURRENT_REQUIREMENTS_HASH" ]; then
    echo ">>> 安装依赖..."
    python3 -m pip install -r requirements.txt
    printf '%s' "$CURRENT_REQUIREMENTS_HASH" > "$REQUIREMENTS_HASH_FILE"
fi

echo "Python: $(python3 --version)"
echo "Path:   $(which python3)"
echo ">>> 前台启动 gateway.py"

exec python3 gateway.py
