# Feishu → Claude Code Gateway

Minimal relay: Feishu messages → ttadk Claude Code → Feishu reply.

## Architecture

```
User (Feishu) ──WebSocket──→ FeishuBot ──subprocess──→ ttadk code (Claude Code)
                    ↑                                         │
                    └─────── card reply ◀─── result text ─────┘
```

**One file. Two classes. Zero frameworks.**

| Component | Role |
|-----------|------|
| `FeishuBot` | lark-oapi WebSocket listener, card reply/update |
| `ClaudeCodeBridge` | ttadk subprocess wrapper, session management |

## vs DeerFlow

| | DeerFlow | This Gateway |
|---|---|---|
| Python files | 100+ | 1 |
| Dependencies | ~25 packages | 2 (lark-oapi, pyyaml) |
| Framework | LangGraph + LangChain | None |
| Frontend | Next.js | None |
| Agent loop | DeerFlow orchestration (redundant) | Claude Code built-in |
| Session | LangGraph checkpointer + SQLite | In-memory dict |

## Quick Start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Set env vars
export FEISHU_APP_ID=cli_xxxx
export FEISHU_APP_SECRET=xxxx

# 3. Run
python gateway.py
```

## Message Flow

1. User sends message in Feishu
2. Bot adds ✅ emoji reaction
3. Bot replies "🤔 Thinking..." card in thread
4. Claude Code processes (may use Bash/Read/Write/Edit internally)
5. Bot updates card with result
6. Bot adds ✅ DONE reaction

## Multi-turn

Thread replies share the same `topic_id` (Feishu `root_id`), mapped to ttadk `session_id` via `--resume`. Claude Code retains full context within a thread.

## Config

Edit `config.yaml` or use env vars. See config.yaml for all options.
