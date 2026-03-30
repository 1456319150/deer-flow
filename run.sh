#!/usr/bin/env bash
# Quick start: ./run.sh
# Restart and continue current topic: ./run.sh --current-topic --initial-prompt "继续处理刚才的问题"
# Or with explicit topic: ./run.sh --initial-topic-id topic-123 --initial-prompt "restart-and-continue"

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

INITIAL_TOPIC_ID="${GATEWAY_INITIAL_TOPIC_ID:-}"
INITIAL_PROMPT="${GATEWAY_INITIAL_PROMPT:-}"
ARG_INITIAL_TOPIC_ID=""
ARG_INITIAL_PROMPT=""
HAS_ARG_INITIAL_TOPIC_ID=0
HAS_ARG_INITIAL_PROMPT=0
USE_CURRENT_TOPIC=0
DETACH_AFTER_READY=1
RESTART_HELPER=0
GATEWAY_LOG_PATH="logs/gateway.log"
GATEWAY_STDOUT_LOG_PATH="logs/gateway.stdout.log"
SELF_RESTART_LOG_PATH=".run/self-restart.log"
GATEWAY_READY_MARKER="Gateway running. Ctrl+C to stop."
GATEWAY_READY_TIMEOUT_SECONDS=20
GATEWAY_READY_TIMEOUT_WITH_INIT_SECONDS=90

usage() {
    cat <<'EOF' >&2
用法:
  ./run.sh
  ./run.sh --current-topic --initial-prompt "继续处理"
  ./run.sh --initial-topic-id <topic_id> --initial-prompt "继续处理"
  ./run.sh --restart-helper --initial-topic-id <topic_id> --initial-prompt "继续处理"
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --initial-topic-id)
            if [ $# -lt 2 ]; then
                echo "缺少参数: --initial-topic-id <topic_id>" >&2
                exit 1
            fi
            ARG_INITIAL_TOPIC_ID="$2"
            HAS_ARG_INITIAL_TOPIC_ID=1
            shift 2
            ;;
        --initial-prompt)
            if [ $# -lt 2 ]; then
                echo "缺少参数: --initial-prompt <prompt>" >&2
                exit 1
            fi
            ARG_INITIAL_PROMPT="$2"
            HAS_ARG_INITIAL_PROMPT=1
            shift 2
            ;;
        --current-topic)
            USE_CURRENT_TOPIC=1
            shift
            ;;
        --restart-helper)
            RESTART_HELPER=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "未知参数: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [ "$HAS_ARG_INITIAL_TOPIC_ID" -eq 1 ]; then
    INITIAL_TOPIC_ID="$ARG_INITIAL_TOPIC_ID"
fi

if [ "$HAS_ARG_INITIAL_PROMPT" -eq 1 ]; then
    INITIAL_PROMPT="$ARG_INITIAL_PROMPT"
fi

if [ "$USE_CURRENT_TOPIC" -eq 1 ] && [ "$HAS_ARG_INITIAL_TOPIC_ID" -eq 1 ]; then
    echo "--current-topic 和 --initial-topic-id 不能同时提供" >&2
    exit 1
fi

if [ "$USE_CURRENT_TOPIC" -eq 1 ]; then
    INITIAL_TOPIC_ID=""
fi

if [ -n "$INITIAL_PROMPT" ] && [ -z "$INITIAL_TOPIC_ID" ] && [ "$USE_CURRENT_TOPIC" -eq 0 ]; then
    echo "提供 --initial-prompt 时，必须同时提供 --initial-topic-id 或 --current-topic" >&2
    exit 1
fi

if [ "$USE_CURRENT_TOPIC" -eq 1 ] && [ -z "$INITIAL_PROMPT" ]; then
    echo "提供 --current-topic 时，必须同时提供 --initial-prompt" >&2
    exit 1
fi

if [ "$USE_CURRENT_TOPIC" -eq 0 ] && [ -n "$INITIAL_TOPIC_ID" ] && [ -z "$INITIAL_PROMPT" ]; then
    echo "--initial-topic-id 和 --initial-prompt 必须同时提供" >&2
    exit 1
fi

load_feishu_app_id() {
    if [ -n "${FEISHU_APP_ID:-}" ]; then
        return
    fi

    if [ ! -f ".env" ]; then
        return
    fi

    FEISHU_APP_ID="$(python3 - <<'PY'
from pathlib import Path

for raw_line in Path('.env').read_text(encoding='utf-8').splitlines():
    line = raw_line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    key, value = line.split('=', 1)
    if key.strip() != 'FEISHU_APP_ID':
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    print(value)
    break
PY
)"

    if [ -n "$FEISHU_APP_ID" ]; then
        export FEISHU_APP_ID
    fi
}

pid_is_running() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null
}

pid_looks_like_gateway() {
    local pid="$1"
    local cmd
    cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    case "$cmd" in
        *"python3 gateway.py"*|*"python gateway.py"*|*"/gateway.py"*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_current_topic_id() {
    python3 - <<'PY'
from pathlib import Path
import re

current_topic_path = Path('.run/current_feishu_topic.txt')
if current_topic_path.exists():
    topic_id = current_topic_path.read_text(encoding='utf-8', errors='ignore').strip()
    if topic_id:
        print(topic_id)
        raise SystemExit(0)

log_path = Path('logs/gateway.log')
if not log_path.exists():
    raise SystemExit(1)

pattern = re.compile(r"\btopic=([^\s]+)")
for raw_line in reversed(log_path.read_text(encoding='utf-8', errors='ignore').splitlines()):
    match = pattern.search(raw_line)
    if match:
        print(match.group(1).strip())
        raise SystemExit(0)

raise SystemExit(1)
PY
}

print_log_tail() {
    if [ ! -f "$GATEWAY_LOG_PATH" ]; then
        return
    fi

    echo ">>> 最近的 gateway 日志:"
    python3 - <<'PY'
from pathlib import Path

log_path = Path('logs/gateway.log')
for line in log_path.read_text(encoding='utf-8', errors='ignore').splitlines()[-80:]:
    print(line)
PY
}

stop_existing_gateway() {
    if [ ! -f "$GATEWAY_PID_FILE" ]; then
        echo ">>> 未发现旧 gateway pid 文件，跳过停止"
        return
    fi

    local old_pid
    old_pid="$(tr -d '[:space:]' < "$GATEWAY_PID_FILE")"
    if [ -z "$old_pid" ]; then
        rm -f "$GATEWAY_PID_FILE"
        echo ">>> gateway pid 文件为空，已清理"
        return
    fi

    if ! pid_is_running "$old_pid"; then
        rm -f "$GATEWAY_PID_FILE"
        echo ">>> 旧 gateway 进程不存在，已清理陈旧 pid 文件"
        return
    fi

    if ! pid_looks_like_gateway "$old_pid"; then
        echo ">>> pid 文件指向的进程不是 gateway，停止操作以免误杀: pid=$old_pid" >&2
        exit 1
    fi

    echo ">>> 停止旧 gateway 进程: pid=$old_pid"
    kill "$old_pid"

    for _ in 1 2 3 4 5 6 7 8 9 10; do
        if ! pid_is_running "$old_pid"; then
            rm -f "$GATEWAY_PID_FILE"
            echo ">>> 旧 gateway 已停止"
            return
        fi
        sleep 1
    done

    echo ">>> 旧 gateway 未在 10 秒内退出，发送 SIGKILL"
    kill -9 "$old_pid"
    rm -f "$GATEWAY_PID_FILE"
}

start_gateway_detached() {
    mkdir -p logs
    touch "$GATEWAY_STDOUT_LOG_PATH"

    echo ">>> 启动新 gateway 进程"
    nohup python3 gateway.py >> "$GATEWAY_STDOUT_LOG_PATH" 2>&1 < /dev/null &
    GATEWAY_PID=$!
    disown "$GATEWAY_PID" 2>/dev/null || true
    printf '%s\n' "$GATEWAY_PID" > "$GATEWAY_PID_FILE"
}

wait_for_gateway_ready() {
    local pid="$1"
    local timeout="$GATEWAY_READY_TIMEOUT_SECONDS"

    if [ -n "${GATEWAY_INITIAL_TOPIC_ID:-}" ] && [ -n "${GATEWAY_INITIAL_PROMPT:-}" ]; then
        timeout="$GATEWAY_READY_TIMEOUT_WITH_INIT_SECONDS"
    fi

    echo ">>> 等待 gateway 完成渠道初始化..."

    for _ in $(seq 1 "$timeout"); do
        if ! pid_is_running "$pid"; then
            echo ">>> gateway 在完成初始化前已退出" >&2
            print_log_tail >&2
            return 1
        fi

        if [ -f "$GATEWAY_LOG_PATH" ] && python3 - <<'PY'
from pathlib import Path

log_path = Path('logs/gateway.log')
text = log_path.read_text(encoding='utf-8', errors='ignore') if log_path.exists() else ''
raise SystemExit(0 if 'Gateway running. Ctrl+C to stop.' in text else 1)
PY
        then
            echo ">>> gateway 已就绪: pid=$pid"
            return 0
        fi

        sleep 1
    done

    echo ">>> 在 ${timeout} 秒内未看到 ready 日志" >&2
    print_log_tail >&2
    return 1
}

schedule_async_restart() {
    mkdir -p .run
    touch "$SELF_RESTART_LOG_PATH"

    nohup "$SCRIPT_DIR/run.sh" \
        --restart-helper \
        --initial-topic-id "$INITIAL_TOPIC_ID" \
        --initial-prompt "$INITIAL_PROMPT" \
        >> "$SELF_RESTART_LOG_PATH" 2>&1 < /dev/null &
    local helper_pid=$!
    disown "$helper_pid" 2>/dev/null || true

    echo ">>> 已安排后台自重启: helper_pid=$helper_pid"
    echo ">>> 自重启日志: $SELF_RESTART_LOG_PATH"
}

load_feishu_app_id
mkdir -p .run logs
GATEWAY_PID_FILE=".run/gateway.${FEISHU_APP_ID:-default}.pid"

if [ "$USE_CURRENT_TOPIC" -eq 1 ]; then
    if ! INITIAL_TOPIC_ID="$(resolve_current_topic_id)"; then
        echo "无法从 .run/current_feishu_topic.txt 或 logs/gateway.log 中解析当前 topic；请手动提供 --initial-topic-id" >&2
        exit 1
    fi
fi

if [ -n "$INITIAL_TOPIC_ID" ] && [ -n "$INITIAL_PROMPT" ]; then
    export GATEWAY_INITIAL_TOPIC_ID="$INITIAL_TOPIC_ID"
    export GATEWAY_INITIAL_PROMPT="$INITIAL_PROMPT"
else
    unset GATEWAY_INITIAL_TOPIC_ID
    unset GATEWAY_INITIAL_PROMPT
fi

echo "Python: $(python3 --version)"
echo "Path:   $(which python3)"
if [ -n "${FEISHU_APP_ID:-}" ]; then
    echo "Feishu App: $FEISHU_APP_ID"
else
    echo "Feishu App: 未解析到 FEISHU_APP_ID"
fi
if [ -n "${GATEWAY_INITIAL_TOPIC_ID:-}" ]; then
    echo "Initial Topic: $GATEWAY_INITIAL_TOPIC_ID"
fi
if [ -n "${GATEWAY_INITIAL_PROMPT:-}" ]; then
    echo "Initial Prompt: 已设置 (${#GATEWAY_INITIAL_PROMPT} chars)"
fi

if [ "$RESTART_HELPER" -eq 0 ] && [ -n "$INITIAL_PROMPT" ]; then
    schedule_async_restart
    exit 0
fi

if [ "$RESTART_HELPER" -eq 1 ]; then
    sleep 1
fi

stop_existing_gateway
start_gateway_detached
wait_for_gateway_ready "$GATEWAY_PID"

echo ">>> gateway 已重启完成"
echo ">>> pid 文件: $GATEWAY_PID_FILE"
echo ">>> 日志文件: $GATEWAY_LOG_PATH"
