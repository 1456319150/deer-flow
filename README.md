# Feishu / WeChat → Claude Code Gateway

Minimal relay: Feishu or WeChat messages → ttadk Claude Code → channel reply.

## Architecture

```text
Feishu user  ──WebSocket──→ FeishuBot  ──┐
                                         ├──→ ClaudeCodeBridge ──subprocess──→ ttadk code
WeChat user ──HTTP long-poll──→ WeixinBot ─┘
```

| Component | Role |
|-----------|------|
| `FeishuBot` | lark-oapi WebSocket listener, card reply/update |
| `WeixinBot` | iLink long-poll listener, plain-text reply |
| `ClaudeCodeBridge` | ttadk subprocess wrapper, session management |
| `weixin_channel.py` | standalone iLink HTTP/JSON protocol client |

## vs DeerFlow

| | DeerFlow | This Gateway |
|---|---|---|
| Python files | 100+ | a few focused channel files |
| Dependencies | ~25 packages | small runtime set |
| Framework | LangGraph + LangChain | None |
| Frontend | Next.js | None |
| Agent loop | DeerFlow orchestration (redundant) | Claude Code built-in |
| Session | LangGraph checkpointer + SQLite | local session mapping file |

## Quick Start

推荐直接使用前台启动脚本：

```bash
# 1. Set env vars for Feishu (optional if only using WeChat)
export FEISHU_APP_ID=cli_xxxx
export FEISHU_APP_SECRET=xxxx

# 2. Start gateway in foreground
./run.sh
```

`run.sh` 现在只负责准备本地 Python 环境并前台执行 `python gateway.py`。
如果你已经自己管理好了虚拟环境，也可以直接运行：

```bash
python gateway.py
```

## Channel Behavior

### Feishu

1. User sends message in Feishu
2. Bot adds `OK` reaction
3. Claude Code processes the message
4. Bot replies in thread with cards / stream events
5. Bot adds `DONE` reaction

Multi-turn behavior:
- Thread replies share the same `topic_id`
- `topic_id = root_id or msg_id`
- `topic_id` is mapped to ttadk `session_id` via `--resume`

### WeChat

1. `WeixinBot` long-polls iLink `getupdates`
2. Inbound text is routed to `ClaudeCodeBridge`
3. Bot sends typing status during processing
4. Reply is converted to plain text and sent with `sendmessage`

Multi-turn behavior:
- `topic_id = wx_{from_user}`
- Each WeChat user maps to one Claude session
- iLink `context_token` is persisted locally so replies stay attached to the same WeChat conversation context

## Configuration

Edit `config.yaml` or use env vars.

### Example config

```yaml
feishu:
  app_id: ${FEISHU_APP_ID}
  app_secret: ${FEISHU_APP_SECRET}

claude:
  ttadk_cmd: ttadk
  model: gpt-5.4
  target: claude
  timeout: 600
  session_store_path: .gateway-sessions.json
  allowed_tools: "Bash,Read,Write,Edit,Glob,WebSearch,WebFetch"

weixin:
  enabled: true
  auto_login: false
  account_path: .weixin-account.json
  context_tokens_path: .weixin-context-tokens.json
  sync_cursor_path: .weixin-sync-cursor.json
```

## WeChat Setup

### Option A: one-time QR login, then reuse saved token

Recommended for local use.

```bash
python weixin_login.py
python gateway.py
```

What happens:
- `weixin_login.py` calls iLink QR-login endpoints
- after you scan and confirm, the returned `bot_token` is saved to `.weixin-account.json`
- later, `WeixinBot` loads that file and authenticates each API call with `Authorization: Bearer <bot_token>`

### Option B: auto login on gateway startup

```yaml
weixin:
  enabled: true
  auto_login: true
```

Behavior:
- if `account_path` does not exist, the gateway performs QR login on startup
- after login succeeds, the token is saved locally
- future restarts reuse the saved token until it expires or is revoked

### State files used by WeChat

These files are local runtime state and are already ignored by git:

- `.gateway-sessions.json`: Claude session mapping used for cross-message continuity
- `.weixin-account.json`: saved login credential (`bot_token`)
- `.weixin-context-tokens.json`: latest `context_token` per user
- `.weixin-sync-cursor.json`: last `get_updates_buf` cursor for crash recovery
- `logs/gateway.log`: the only runtime log file, truncated on each startup

## WeChat Authorization Modes

This repository currently supports **QR login based bot-token authorization**.

### 1. QR login → bot token

Implemented in `weixin_channel.py:155` and `weixin_login.py:19`.

Principle:
- client calls `GET /ilink/bot/get_bot_qrcode?bot_type=3`
- WeChat/iLink returns a QR code URL plus a QR key
- the user scans and confirms in WeChat
- client polls `GET /ilink/bot/get_qrcode_status?qrcode=...`
- once approved, iLink returns a `bot_token`
- subsequent API requests send:
  - `AuthorizationType: ilink_bot_token`
  - `Authorization: Bearer <bot_token>`

What it authorizes:
- long-polling inbound messages (`getupdates`)
- sending replies (`sendmessage`)
- typing status (`sendtyping`)

Lifetime:
- the codebase itself does **not** encode a fixed TTL
- effective lifetime is determined by the server side token policy
- in practice, the token remains usable until one of these happens:
  - server expires it
  - session is revoked
  - the account is re-bound / invalidated
- when the server marks the session expired, the code treats `errcode = -14` as session expiry and requires re-login (`weixin_channel.py:43`, `weixin_channel.py:396-398`, `weixin_bot.py:100-103`)

### 2. Reuse saved credential from disk

Implemented in `weixin_channel.py:229-245` and used by `weixin_bot.py:67-84`.

Principle:
- after a successful QR login, the returned token is written to `account_path`
- on startup, the gateway reads that file and skips the QR flow
- every request still uses the same bearer token mechanism above

This is not a different server-side auth type; it is just **local credential persistence** for the same bot-token authorization flow.

### 3. OpenClaw plugin installer / host-assisted login

Relevant if you use `@tencent-weixin/openclaw-weixin` outside this repo.

Principle:
- OpenClaw installs the plugin and guides the same WeChat login flow
- after login, the host/plugin stores and uses the credential for its own gateway runtime

In other words, the auth primitive is still the same idea: **WeChat QR approval → iLink-issued bot token**. The difference is only who manages the token lifecycle: this Python repo or the OpenClaw host.

## Effective Validity and Renewal

For this repo, the practical rules are:

- First use: run `python weixin_login.py` or set `auto_login: true`
- Normal restart: reuse `.weixin-account.json`
- If replies/polling start failing with session-expired behavior (`errcode = -14`), delete the saved account file and log in again

```bash
rm .weixin-account.json
python weixin_login.py
```

## Notes

- WeChat replies are plain text; markdown is stripped before sending
- Feishu and WeChat can run at the same time in one gateway process
- startup truncates `logs/gateway.log` intentionally to save disk space
- shell 层的 `current-topic` / `initial-prompt` / `restart-helper` 自重启链路已移除
- Claude 会话连续性仍由 `.gateway-sessions.json` 保持，不依赖 shell 续接
