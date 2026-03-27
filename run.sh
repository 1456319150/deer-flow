#!/usr/bin/env bash
# Quick start: ./run.sh
# Or with env vars: FEISHU_APP_ID=xxx FEISHU_APP_SECRET=yyy ./run.sh

set -euo pipefail
cd "$(dirname "$0")"

# Ensure ttadk is in PATH (adjust to your install location)
export PATH="$HOME/.npm-global/bin:$PATH"

python3 gateway.py
