"""WeixinBot — WeChat channel adapter for DeerFlow gateway.

Mirrors FeishuBot's role: receives WeChat messages via iLink long-poll,
routes them to ClaudeCodeBridge, and sends replies back as plain text.

Integration:
    In gateway.py main():
        weixin_bot = WeixinBot(cfg["weixin"], bridge)
        await weixin_bot.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from weixin_channel import (
    WeixinAccount,
    WeixinChannel,
    WeixinMessage,
    ContextTokenStore,
    SyncCursor,
    load_account,
    save_account,
    qr_login,
    strip_markdown,
)

# Import from gateway — these are the shared types
from gateway import ClaudeCodeBridge, StreamResult, _preview_text

log = logging.getLogger("weixin_bot")


class WeixinBot:
    """WeChat Bot: iLink long-poll → ClaudeCodeBridge → text reply.

    Design decisions:
    - Plain text replies (WeChat doesn't render Markdown)
    - Session keyed by from_user (1:1 conversation = 1 session)
    - Typing indicator while Claude is processing
    - Auto-reconnect on transient failures
    - Re-login prompt on session expiry
    """

    MSG_MAX_AGE = 120     # Skip messages older than 2 min (offline replay)

    def __init__(self, cfg: dict, bridge: ClaudeCodeBridge):
        self.bridge = bridge
        self._enabled = cfg.get("enabled", False)
        self._account_path = cfg.get("account_path", ".weixin-account.json")
        self._ctx_store_path = cfg.get("context_tokens_path", ".weixin-context-tokens.json")
        self._cursor_path = cfg.get("sync_cursor_path", ".weixin-sync-cursor.json")
        self._auto_login = cfg.get("auto_login", False)
        self._channel: WeixinChannel | None = None
        self._seen_msgs: set[str] = set()

    async def start(self) -> None:
        """Initialize and start the WeChat polling loop."""
        if not self._enabled:
            log.info("[WeixinBot] disabled in config (weixin.enabled=false)")
            return

        # Load or create account
        account = load_account(self._account_path)
        if not account:
            if self._auto_login:
                log.info("[WeixinBot] No saved account, starting QR login...")
                account = await qr_login()
                save_account(account, self._account_path)
            else:
                log.warning(
                    "[WeixinBot] No account found at %s. "
                    "Set weixin.auto_login=true or run: python -m weixin_login",
                    self._account_path,
                )
                return

        ctx_store = ContextTokenStore(self._ctx_store_path)
        sync_cursor = SyncCursor(self._cursor_path)
        self._channel = WeixinChannel(account, ctx_store, sync_cursor)

        # Start polling in background task
        asyncio.create_task(self._poll_loop())
        log.info("✅ WeixinBot started (baseurl=%s)", account.baseurl)

    async def _poll_loop(self) -> None:
        """Main message processing loop."""
        assert self._channel is not None

        try:
            async for msg in self._channel.poll():
                try:
                    await self._handle_message(msg)
                except Exception:
                    log.exception("[WeixinBot] Error handling message from %s", msg.from_user)
        except RuntimeError as e:
            if "expired" in str(e).lower():
                log.error("[WeixinBot] Session expired! Delete %s and restart to re-login.", self._account_path)
            else:
                log.exception("[WeixinBot] Fatal error in poll loop")

    async def _handle_message(self, msg: WeixinMessage) -> None:
        """Process a single inbound message."""
        # Dedup
        if msg.msg_id:
            if msg.msg_id in self._seen_msgs:
                return
            self._seen_msgs.add(msg.msg_id)
            if len(self._seen_msgs) > 10000:
                self._seen_msgs.clear()

        # Skip stale messages
        if msg.timestamp:
            age = time.time() - msg.timestamp
            if age > self.MSG_MAX_AGE:
                log.info("[WeixinBot] skip stale msg=%s age=%.0fs", msg.msg_id, age)
                return

        # Skip empty
        if msg.is_empty:
            return

        text = msg.text.strip()
        from_user = msg.from_user
        ctx_token = msg.context_token

        log.info("[WeixinBot] 收到消息 from=%s text=%r", from_user, text[:100])

        # Use from_user as topic_id (each WeChat user = one conversation thread)
        topic_id = f"wx_{from_user}"

        # Show typing
        if ctx_token:
            await self._channel.send_typing(from_user, ctx_token, typing=True)

        # --- Route to Claude Code ---
        try:
            result = await self._process_with_streaming(topic_id, text, from_user, ctx_token)
        except Exception as e:
            log.exception("[WeixinBot] Bridge error")
            result = StreamResult(assistant_texts=[f"处理出错: {e}"])

        # --- Send reply ---
        reply = self._format_reply(result)
        if reply:
            plain = strip_markdown(reply)
            log.info("[WeixinBot] 发送回复 to=%s len=%d preview=%r",
                     from_user, len(plain), _preview_text(plain))
            await self._channel.send_text(from_user, ctx_token, plain)
        else:
            await self._channel.send_text(
                from_user, ctx_token, "(Claude Code 已执行操作但未生成文字回复)")

        # Cancel typing
        if ctx_token:
            await self._channel.send_typing(from_user, ctx_token, typing=False)

    async def _process_with_streaming(
        self, topic_id: str, text: str, from_user: str, ctx_token: str
    ) -> StreamResult:
        """Call Bridge with streaming, send intermediate typing keepalive."""
        result: StreamResult | None = None
        last_typing_time = time.time()

        async for event in self.bridge.stream_ask(topic_id, text):
            if event["type"] == "final":
                result = event["result"]
            elif event["type"] == "stream_event":
                # Keepalive typing every 5s during processing
                now = time.time()
                if now - last_typing_time > 5 and ctx_token:
                    await self._channel.send_typing(from_user, ctx_token, typing=True)
                    last_typing_time = now

        return result or StreamResult()

    @classmethod
    def _format_reply(cls, result: StreamResult) -> str:
        """Format StreamResult for WeChat (plain text, no cards).

        Unlike Feishu which uses rich cards, WeChat gets a simpler format:
        - Tool calls summarized briefly
        - Thinking omitted (too verbose for mobile)
        - Focus on the actual reply text
        """
        sections: list[str] = []

        # Brief tool summary (collapsed)
        if result.tool_calls:
            tool_names = [tc.name for tc in result.tool_calls[:5]]
            summary = ", ".join(tool_names)
            remaining = len(result.tool_calls) - 5
            if remaining > 0:
                summary += f" 等{remaining + 5}个工具"
            sections.append(f"[执行了: {summary}]")

        # Main reply
        reply = result.reply_text
        if reply:
            sections.append(reply)

        return "\n\n".join(sections)
