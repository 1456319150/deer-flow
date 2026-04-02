"""Feishu Bot — Feishu WebSocket channel adapter for DeerFlow gateway.

Receives Feishu messages via lark-oapi WebSocket, routes them to
ClaudeCodeBridge, and sends replies as interactive cards with rich
Markdown formatting.

Extracted from gateway.py for architectural symmetry with WeixinBot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

from gateway import ClaudeCodeBridge, StreamResult, StreamEvent, UsageSummary, _preview_text

log = logging.getLogger("feishu")


class FeishuBot:
    """Feishu WebSocket bot with card-based responses."""

    # Truncation limits
    MAX_REPLY = 50000
    MAX_THINKING = 2000
    MAX_TOOL_INPUT = 500
    MAX_TOOL_OUTPUT = 1000
    MAX_TOOL_CALLS_SHOWN = 10
    MSG_MAX_AGE = 120  # Ignore messages older than 2 minutes (offline replay protection)

    def __init__(self, cfg: dict, bridge: ClaudeCodeBridge):
        self.app_id: str = cfg["app_id"]
        self.app_secret: str = cfg["app_secret"]
        self.bridge = bridge
        self._api_client: Any = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._ws_thread: threading.Thread | None = None
        self._sdk: dict[str, Any] = {}
        self._seen_msgs: set[str] = set()  # msg_id dedup

    async def start(self) -> None:
        """Initialize API client and start WebSocket listener."""
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import (
            CreateMessageReactionRequest,
            CreateMessageReactionRequestBody,
            Emoji,
            PatchMessageRequest,
            PatchMessageRequestBody,
            ReplyMessageRequest,
            ReplyMessageRequestBody,
        )

        self._api_client = lark.Client.builder().app_id(self.app_id).app_secret(self.app_secret).build()
        self._main_loop = asyncio.get_event_loop()
        self._lark = lark
        self._sdk = {
            "ReactReq": CreateMessageReactionRequest,
            "ReactBody": CreateMessageReactionRequestBody,
            "Emoji": Emoji,
            "PatchReq": PatchMessageRequest,
            "PatchBody": PatchMessageRequestBody,
            "ReplyReq": ReplyMessageRequest,
            "ReplyBody": ReplyMessageRequestBody,
        }

        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()
        log.info("✅ Feishu bot started (app_id=%s)", self.app_id)

    def _run_ws(self) -> None:
        """Run lark WS client in a thread with its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            import lark_oapi as lark
            import lark_oapi.ws.client as ws_mod

            ws_mod.loop = loop

            handler = (
                lark.EventDispatcherHandler.builder("", "")
                .register_p2_im_message_receive_v1(self._on_message)
                .register_p2_im_message_reaction_created_v1(lambda e: None)
                .build()
            )
            ws = lark.ws.Client(
                app_id=self.app_id,
                app_secret=self.app_secret,
                event_handler=handler,
                log_level=lark.LogLevel.INFO,
            )
            ws.start()
        except Exception:
            log.exception("Feishu WebSocket error")

    def _on_message(self, event: Any) -> None:
        """Handle incoming message (runs in lark thread)."""
        try:
            msg = event.event.message
            chat_id = msg.chat_id
            msg_id = msg.message_id
            root_id = getattr(msg, "root_id", None) or None
            topic_id = root_id or msg_id

            # --- Dedup: skip already-processed messages ---
            if msg_id in self._seen_msgs:
                log.debug("[MSG] skip duplicate msg=%s", msg_id)
                return
            self._seen_msgs.add(msg_id)
            # Cap set size to prevent unbounded growth
            if len(self._seen_msgs) > 10000:
                self._seen_msgs.clear()

            # --- Staleness: skip messages sent while bot was offline ---
            create_time_ms = int(getattr(msg, "create_time", "0") or "0")
            if create_time_ms:
                age = time.time() - create_time_ms / 1000
                if age > self.MSG_MAX_AGE:
                    log.info("[MSG] skip stale msg=%s age=%.0fs", msg_id, age)
                    return

            content = json.loads(msg.content)
            text = self._extract_text(content).strip()

            log.info("[MSG] chat=%s msg=%s topic=%s text=%r", chat_id, msg_id, topic_id, text[:100])

            if not text:
                return

            if self._main_loop and self._main_loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(
                    self._handle(chat_id, msg_id, topic_id, text), self._main_loop
                )
                fut.add_done_callback(lambda f, mid=msg_id: self._log_err(f, mid))
        except Exception:
            log.exception("Error processing message")

    async def _handle(self, chat_id: str, msg_id: str, topic_id: str, text: str) -> None:
        """Full message lifecycle: react -> stream cards -> done."""
        await self._react(msg_id, "OK")
        result = StreamResult()
        emitted_event_keys: set[tuple[str, str]] = set()
        streamed_texts: list[str] = []
        tool_cards: dict[str, dict[str, str | None]] = {}
        saw_stream_event = False

        # --- Streaming card state ---
        _thinking_card_id: str | None = None
        _thinking_acc: list[str] = []
        _result_card_id: str | None = None
        _result_acc: list[str] = []
        _thinking_dirty = False  # new content since last flush
        _result_dirty = False
        _FLUSH_INTERVAL = 0.5  # seconds between card updates (was 1.0)

        async def _flush_cards():
            """Background flusher: periodically update cards with accumulated text."""
            nonlocal _thinking_dirty, _result_dirty, _thinking_card_id, _result_card_id
            while True:
                await asyncio.sleep(_FLUSH_INTERVAL)
                if _thinking_dirty and _thinking_acc:
                    _thinking_dirty = False
                    full = "💭 **Thinking...**\n\n" + "".join(_thinking_acc)
                    if _thinking_card_id:
                        try:
                            await self._update_card(_thinking_card_id, full)
                        except Exception:
                            log.debug("flush thinking card update failed")
                    else:
                        _thinking_card_id = await self._reply_card(msg_id, full)
                if _result_dirty and _result_acc:
                    _result_dirty = False
                    full = "".join(_result_acc)
                    if _result_card_id:
                        try:
                            await self._update_card(_result_card_id, full)
                        except Exception:
                            log.debug("flush result card update failed")
                    else:
                        _result_card_id = await self._reply_card(msg_id, full)

        flush_task = asyncio.create_task(_flush_cards())

        try:
            async for event in self.bridge.stream_ask(topic_id, text):
                if event["type"] == "stream_event":
                    saw_stream_event = True
                    stream_event = event["event"]
                    log.info(
                        "[RenderEvent] kind=%s tool=%s text_len=%d preview=%r",
                        stream_event.kind,
                        stream_event.tool_name or "-",
                        len(stream_event.text),
                        _preview_text(stream_event.text),
                    )
                    # -- Streaming delta handling (Mira bridge) --
                    if stream_event.kind == "thinking" and stream_event.text:
                        _thinking_acc.append(stream_event.text)
                        _thinking_dirty = True
                        continue

                    if stream_event.kind == "result" and stream_event.text:
                        _result_acc.append(stream_event.text)
                        _result_dirty = True
                        continue

                    if stream_event.kind == "text":
                        streamed_texts.append(stream_event.text)
                    if stream_event.kind == "result":
                        current_reply = "\n\n".join(streamed_texts).strip()
                        if not stream_event.text or stream_event.text == current_reply:
                            continue
                    if stream_event.kind == "tool_result" and stream_event.tool_use_id in tool_cards:
                        tool_card = tool_cards[stream_event.tool_use_id]
                        merged = self._merge_tool_stream_blocks(str(tool_card["tool_use_block"]), stream_event)
                        if not merged or merged == tool_card.get("content"):
                            continue
                        if tool_card.get("card_id"):
                            log.info(
                                "[RenderCard] mode=update content_len=%d tool_use_id=%s",
                                len(merged),
                                stream_event.tool_use_id,
                            )
                            await self._update_card(str(tool_card["card_id"]), merged)
                        else:
                            log.info(
                                "[RenderCard] mode=reply reason=missing_card_id_fallback tool=%s tool_use_id=%s",
                                stream_event.tool_name or "-",
                                stream_event.tool_use_id,
                            )
                            await self._reply_card(msg_id, merged)
                        tool_card["content"] = merged
                        continue
                    if stream_event.kind == "tool_result" and stream_event.tool_use_id:
                        log.info(
                            "[RenderCard] mode=reply reason=unbound_tool_result tool=%s tool_use_id=%s",
                            stream_event.tool_name or "-",
                            stream_event.tool_use_id,
                        )
                    block = self._format_stream_event(stream_event)
                    if not block:
                        continue
                    event_key = self._event_dedup_key(stream_event)
                    if event_key and event_key in emitted_event_keys:
                        log.info(
                            "[RenderSkip] reason=dedup kind=%s key=%r",
                            stream_event.kind,
                            event_key,
                        )
                        continue
                    if event_key:
                        emitted_event_keys.add(event_key)
                    log.info(
                        "[RenderCard] mode=reply content_len=%d last_kind=%s",
                        len(block),
                        stream_event.kind,
                    )
                    card_id = await self._reply_card(msg_id, block)
                    if stream_event.kind == "tool_use" and stream_event.tool_use_id:
                        tool_cards[stream_event.tool_use_id] = {
                            "card_id": card_id,
                            "tool_use_block": block,
                            "content": block,
                        }
                        if card_id:
                            log.info(
                                "[RenderCard] mode=bind tool=%s tool_use_id=%s card_id=%s",
                                stream_event.tool_name or "-",
                                stream_event.tool_use_id,
                                card_id,
                            )
                        else:
                            log.info(
                                "[RenderCard] mode=bind_skipped reason=missing_card_id tool=%s tool_use_id=%s",
                                stream_event.tool_name or "-",
                                stream_event.tool_use_id,
                            )
                elif event["type"] == "final":
                    result = event["result"]
        except Exception as e:
            log.exception("Bridge error")
            result = StreamResult(assistant_texts=[f"❌ Error: {e}"])

        # -- Cancel background flusher --
        flush_task.cancel()
        try:
            await flush_task
        except asyncio.CancelledError:
            pass

        # -- Final flush -- ensure all accumulated content is displayed --
        if _thinking_acc:
            full = "💭 **Thinking**\n\n" + "".join(_thinking_acc)
            if _thinking_card_id:
                try:
                    await self._update_card(_thinking_card_id, full)
                except Exception:
                    pass
            else:
                await self._reply_card(msg_id, full)

        if _result_acc:
            full = "".join(_result_acc)
            if _result_card_id:
                try:
                    await self._update_card(_result_card_id, full)
                except Exception:
                    log.warning("[FinalFlush] update result card %s failed, will retry", _result_card_id)
                    # Retry once after a short delay
                    await asyncio.sleep(1.0)
                    try:
                        await self._update_card(_result_card_id, full)
                    except Exception:
                        log.error("[FinalFlush] retry update result card %s also failed", _result_card_id)
            elif full.strip():
                card_id = await self._reply_card(msg_id, full)
                if card_id is None and full.strip():
                    # Card creation failed -- retry after delay
                    log.warning("[FinalFlush] reply_card returned None for result (%d chars), retrying in 2s", len(full))
                    await asyncio.sleep(2.0)
                    card_id = await self._reply_card(msg_id, full)
                    if card_id is None:
                        log.error("[FinalFlush] retry reply_card also returned None -- user will NOT see result")

        if not saw_stream_event:
            card_content = self._format_result(result)
            log.info("[RenderFallback] content_len=%d", len(card_content))
            await self._reply_card(msg_id, card_content)
        elif result.reply_text and not _result_acc:
            final_text = result.reply_text
            reply_so_far = "\n\n".join(streamed_texts).strip()
            if final_text != reply_so_far:
                final_key = ("result", final_text)
                if final_key not in emitted_event_keys:
                    log.info("[RenderFinalText] text_len=%d preview=%r", len(final_text), _preview_text(final_text))
                    await self._reply_card(msg_id, final_text)

        usage_text = self._format_usage_summary(result.usage)
        if usage_text:
            try:
                log.info("[RenderUsage] content_len=%d", len(usage_text))
                await self._reply_card(msg_id, usage_text)
            except Exception:
                log.exception("Reply usage card failed for %s", msg_id)

        await self._react(msg_id, "DONE")

    @staticmethod
    def _event_dedup_key(event: StreamEvent) -> tuple[str, str] | None:
        if event.kind in {"text", "result"}:
            text = event.text.strip()
            return (event.kind, text) if text else None
        return None

    @classmethod
    def _format_stream_event(cls, event: StreamEvent) -> str:
        text = event.text.strip()
        if event.kind == "thinking":
            if len(text) > cls.MAX_THINKING:
                text = text[:cls.MAX_THINKING] + "\n\n... (thinking truncated)"
            return f"**💭 Thinking**\n> {cls._blockquote(text)}" if text else ""
        if event.kind == "tool_use":
            if len(text) > cls.MAX_TOOL_INPUT:
                text = text[:cls.MAX_TOOL_INPUT] + "..."
            title = event.tool_name or "unknown"
            return f"**🔧 Tool Use: {title}**\n`{text}`" if text else f"**🔧 Tool Use: {title}**"
        if event.kind == "tool_result":
            if len(text) > cls.MAX_TOOL_OUTPUT:
                text = text[:cls.MAX_TOOL_OUTPUT] + "..."
            title = event.tool_name or "unknown"
            return f"**📦 Tool Result: {title}**\n> {cls._blockquote(text)}" if text else f"**📦 Tool Result: {title}**"
        if event.kind == "result":
            if len(text) > cls.MAX_REPLY:
                text = text[:cls.MAX_REPLY] + "\n\n... (truncated, response too long)"
            return f"**✅ Result**\n{text}" if text else ""
        if event.kind == "text":
            if len(text) > cls.MAX_REPLY:
                text = text[:cls.MAX_REPLY] + "\n\n... (truncated, response too long)"
            return text
        return ""

    @classmethod
    def _merge_tool_stream_blocks(cls, tool_use_block: str, tool_result_event: StreamEvent) -> str:
        result_block = cls._format_stream_event(tool_result_event)
        if not tool_use_block:
            return result_block
        if not result_block:
            return tool_use_block
        return f"{tool_use_block}\n\n{result_block}"

    @classmethod
    def _format_stream_transcript(cls, blocks: list[str]) -> str:
        content = "\n\n---\n\n".join(block for block in blocks if block)
        if len(content) > cls.MAX_REPLY:
            return content[:cls.MAX_REPLY] + "\n\n... (truncated, response too long)"
        return content

    @classmethod
    def _format_result(cls, result: StreamResult) -> str:
        """Format StreamResult into rich Markdown for Feishu card.

        Layout:
        1. 💭 Thinking (collapsed if long) -- shows Claude's reasoning
        2. 🔧 Tool Calls -- shows what commands/files were accessed
        3. 📝 Reply -- the actual response text
        """
        sections: list[str] = []

        # --- Thinking section ---
        if result.thinking:
            combined_thinking = "\n\n".join(result.thinking)
            if len(combined_thinking) > cls.MAX_THINKING:
                combined_thinking = combined_thinking[:cls.MAX_THINKING] + "\n\n... (thinking truncated)"
            sections.append(f"**💭 Thinking**\n> {cls._blockquote(combined_thinking)}")

        # --- Tool calls section ---
        if result.tool_calls:
            tool_lines: list[str] = []
            shown = result.tool_calls[:cls.MAX_TOOL_CALLS_SHOWN]
            for i, tc in enumerate(shown, 1):
                input_display = tc.input_text
                if len(input_display) > cls.MAX_TOOL_INPUT:
                    input_display = input_display[:cls.MAX_TOOL_INPUT] + "..."

                tool_lines.append(f"**{i}. {tc.name}**")
                tool_lines.append(f"`{input_display}`")

                if tc.output_text:
                    output_display = tc.output_text
                    if len(output_display) > cls.MAX_TOOL_OUTPUT:
                        output_display = output_display[:cls.MAX_TOOL_OUTPUT] + "..."
                    tool_lines.append(f"> {cls._blockquote(output_display)}")

            remaining = len(result.tool_calls) - cls.MAX_TOOL_CALLS_SHOWN
            if remaining > 0:
                tool_lines.append(f"*... and {remaining} more tool calls*")

            sections.append(f"**🔧 Tool Calls ({len(result.tool_calls)})**\n" + "\n".join(tool_lines))

        # --- Reply section ---
        reply = result.reply_text
        if reply:
            if len(reply) > cls.MAX_REPLY:
                reply = reply[:cls.MAX_REPLY] + "\n\n... (truncated, response too long)"
            # If there are process sections above, add a separator
            if sections:
                sections.append("---")
            sections.append(reply)
        elif result.is_empty:
            sections.append("(Claude Code 已执行操作但未生成文字回复，请尝试更具体的提问)")

        return "\n\n".join(sections)

    @staticmethod
    def _format_usage_summary(usage: UsageSummary | None) -> str:
        if not usage or not usage.has_values:
            return ""

        lines = ["**Usage**"]
        if usage.input_tokens is not None:
            lines.append(f"- Input: {usage.input_tokens:,}")
        if usage.output_tokens is not None:
            lines.append(f"- Output: {usage.output_tokens:,}")
        if usage.cache_creation_input_tokens is not None:
            lines.append(f"- Cache Create: {usage.cache_creation_input_tokens:,}")
        if usage.cache_read_input_tokens is not None:
            lines.append(f"- Cache Read: {usage.cache_read_input_tokens:,}")
        if usage.total_tokens is not None:
            lines.append(f"- Total: {usage.total_tokens:,}")
        if usage.cost_usd is not None:
            lines.append(f"- Cost: ${usage.cost_usd:.4f}")
        return "\n".join(lines)

    @staticmethod
    def _blockquote(text: str) -> str:
        """Convert multi-line text to blockquote format."""
        return "\n> ".join(text.split("\n"))

    # --- Feishu API helpers ---

    async def _react(self, msg_id: str, emoji: str) -> None:
        try:
            req = (
                self._sdk["ReactReq"]
                .builder()
                .message_id(msg_id)
                .request_body(
                    self._sdk["ReactBody"]
                    .builder()
                    .reaction_type(self._sdk["Emoji"].builder().emoji_type(emoji).build())
                    .build()
                )
                .build()
            )
            await asyncio.to_thread(self._api_client.im.v1.message_reaction.create, req)
        except Exception:
            log.debug("React %s failed for %s", emoji, msg_id, exc_info=True)

    async def _reply_card(self, msg_id: str, text: str) -> str | None:
        try:
            req = (
                self._sdk["ReplyReq"]
                .builder()
                .message_id(msg_id)
                .request_body(
                    self._sdk["ReplyBody"]
                    .builder()
                    .msg_type("interactive")
                    .content(self._card(text))
                    .reply_in_thread(True)
                    .build()
                )
                .build()
            )
            resp = await asyncio.to_thread(self._api_client.im.v1.message.reply, req)
            # Enhanced error logging: check response status
            code = getattr(resp, "code", None)
            resp_msg = getattr(resp, "msg", None)
            if code and code != 0:
                log.error(
                    "[ReplyCard] API error for msg=%s code=%s msg=%s text_len=%d",
                    msg_id, code, resp_msg, len(text),
                )
                return None
            data = getattr(resp, "data", None)
            card_id = getattr(data, "message_id", None) if data else None
            if card_id is None:
                log.warning(
                    "[ReplyCard] no message_id in response for msg=%s code=%s resp_msg=%s text_len=%d",
                    msg_id, code, resp_msg, len(text),
                )
            return card_id
        except Exception:
            log.exception("Reply card failed for %s (text_len=%d)", msg_id, len(text))
            return None

    async def _update_card(self, card_id: str, text: str) -> None:
        try:
            req = (
                self._sdk["PatchReq"]
                .builder()
                .message_id(card_id)
                .request_body(self._sdk["PatchBody"].builder().content(self._card(text)).build())
                .build()
            )
            resp = await asyncio.to_thread(self._api_client.im.v1.message.patch, req)
            # Enhanced error logging: check response status
            code = getattr(resp, "code", None)
            resp_msg = getattr(resp, "msg", None)
            if code and code != 0:
                log.error(
                    "[UpdateCard] API error for card=%s code=%s msg=%s text_len=%d",
                    card_id, code, resp_msg, len(text),
                )
        except Exception:
            log.exception("Update card failed for %s (text_len=%d)", card_id, len(text))

    @staticmethod
    def _card(text: str) -> str:
        return json.dumps({
            "config": {"wide_screen_mode": True, "update_multi": True},
            "elements": [{"tag": "markdown", "content": text}],
        })

    @staticmethod
    def _extract_text(content: dict) -> str:
        if "text" in content:
            return content["text"]
        if "content" in content and isinstance(content["content"], list):
            paras = []
            for para in content["content"]:
                if isinstance(para, list):
                    parts = [
                        el.get("text", "")
                        for el in para
                        if isinstance(el, dict) and el.get("tag") in ("text", "at")
                    ]
                    joined = " ".join(p for p in parts if p)
                    if joined:
                        paras.append(joined)
            return "\n\n".join(paras)
        return ""

    @staticmethod
    def _log_err(fut: Any, msg_id: str) -> None:
        try:
            if fut.exception():
                log.error("Handle failed for %s: %s", msg_id, fut.exception())
        except Exception:
            pass
