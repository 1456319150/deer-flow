"""Feishu Bot — Feishu WebSocket channel adapter for DeerFlow gateway.

Receives Feishu messages via lark-oapi WebSocket, routes them to
ClaudeCodeBridge, and sends replies as interactive cards with rich
Markdown formatting.

Extracted from gateway.py for architectural symmetry with WeixinBot.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.request
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

    _INBOUND_DIR = "/tmp/deerflow_inbound"

    def __init__(self, cfg: dict, bridge: ClaudeCodeBridge):
        self.app_id: str = cfg["app_id"]
        self.app_secret: str = cfg["app_secret"]
        self.bridge = bridge
        self._api_client: Any = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._ws_thread: threading.Thread | None = None
        self._sdk: dict[str, Any] = {}
        self._seen_msgs: set[str] = set()  # msg_id dedup
        self._image_cache: dict[str, str | None] = {}  # url -> image_key cache

    async def start(self) -> None:
        """Initialize API client and start WebSocket listener."""
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import (
            CreateImageRequest,
            CreateImageRequestBody,
            CreateMessageReactionRequest,
            CreateMessageReactionRequestBody,
            Emoji,
            GetMessageResourceRequest,
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
            "CreateImageReq": CreateImageRequest,
            "CreateImageBody": CreateImageRequestBody,
            "GetMsgResourceReq": GetMessageResourceRequest,
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

    # --- Inbound attachment helpers ---

    async def _download_feishu_image(self, msg_id: str, image_key: str) -> bytes | None:
        """Download an image from Feishu via message_resource API.

        The im.v1.image.get API only allows downloading images uploaded by the
        bot itself.  For user-sent images we must use message_resource.get with
        the originating message_id + image_key + type=image.
        """
        try:
            req = (
                self._sdk["GetMsgResourceReq"]
                .builder()
                .message_id(msg_id)
                .file_key(image_key)
                .type("image")
                .build()
            )
            resp = await asyncio.to_thread(self._api_client.im.v1.message_resource.get, req)
            if getattr(resp, "code", None) == 0 and resp.file:
                data = resp.file.read()
                log.info("[ImageDownload] ok key=%s msg=%s size=%d", image_key, msg_id, len(data))
                return data
            log.warning("[ImageDownload] API error key=%s msg=%s code=%s", image_key, msg_id, getattr(resp, "code", "?"))
        except Exception:
            log.warning("[ImageDownload] failed key=%s msg=%s", image_key, msg_id, exc_info=True)
        return None

    async def _download_feishu_file(self, msg_id: str, file_key: str, file_type: str = "file") -> bytes | None:
        """Download a file attachment from Feishu by file_key."""
        try:
            req = (
                self._sdk["GetMsgResourceReq"]
                .builder()
                .message_id(msg_id)
                .file_key(file_key)
                .type(file_type)
                .build()
            )
            resp = await asyncio.to_thread(self._api_client.im.v1.message_resource.get, req)
            if getattr(resp, "code", None) == 0 and resp.file:
                data = resp.file.read()
                log.info("[FileDownload] ok key=%s size=%d", file_key, len(data))
                return data
            log.warning("[FileDownload] API error key=%s code=%s", file_key, getattr(resp, "code", "?"))
        except Exception:
            log.warning("[FileDownload] failed key=%s", file_key, exc_info=True)
        return None

    @staticmethod
    def _guess_image_ext(data: bytes) -> str:
        """Guess image extension from magic bytes."""
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return ".png"
        if data[:2] == b'\xff\xd8':
            return ".jpg"
        if data[:4] == b'GIF8':
            return ".gif"
        if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return ".webp"
        return ".bin"

    @staticmethod
    def _cleanup_inbound_attachments(paths: list[str]) -> None:
        """Remove temporary inbound attachment files."""
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass

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

            msg_type = getattr(msg, "message_type", "text") or "text"
            content = json.loads(msg.content)

            # --- Build attachment metadata ---
            attachment_meta: list[dict[str, str]] = []

            if msg_type == "image":
                image_key = content.get("image_key", "")
                if image_key:
                    attachment_meta.append({"type": "image", "key": image_key, "name": "", "msg_id": msg_id})
                text = ""  # image-only message, no text
                log.info("[MSG] chat=%s msg=%s topic=%s type=image key=%s", chat_id, msg_id, topic_id, image_key)
            elif msg_type == "file":
                file_key = content.get("file_key", "")
                file_name = content.get("file_name", "attachment")
                if file_key:
                    attachment_meta.append({"type": "file", "key": file_key, "name": file_name, "msg_id": msg_id})
                text = ""
                log.info("[MSG] chat=%s msg=%s topic=%s type=file key=%s name=%s", chat_id, msg_id, topic_id, file_key, file_name)
            elif msg_type in ("audio", "video", "media"):
                text = f"[不支持的消息类型: {msg_type}，请发送文字、图片或文件]"
                log.info("[MSG] chat=%s msg=%s topic=%s type=%s (unsupported)", chat_id, msg_id, topic_id, msg_type)
            else:
                text = self._extract_text(content).strip()
                log.info("[MSG] chat=%s msg=%s topic=%s text=%r", chat_id, msg_id, topic_id, text[:100])

            # For image/file-only messages, provide a description as prompt
            if not text and attachment_meta:
                types = [a["type"] for a in attachment_meta]
                names = [a.get("name") or a["key"] for a in attachment_meta]
                text = f"[用户发送了附件: {', '.join(names)}]"

            if not text:
                return

            if self._main_loop and self._main_loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(
                    self._handle(chat_id, msg_id, topic_id, text, attachment_meta=attachment_meta or None), self._main_loop
                )
                fut.add_done_callback(lambda f, mid=msg_id: self._log_err(f, mid))
        except Exception:
            log.exception("Error processing message")

    async def _handle(self, chat_id: str, msg_id: str, topic_id: str, text: str, attachment_meta: list[dict[str, str]] | None = None) -> None:
        """Full message lifecycle: react -> stream cards -> done.

        Card ordering strategy (2-card model):
          1. 💭 Thinking card — thinking text + tool calls combined
          2. ✅ Result card   — final answer

        Tool calls are appended to the thinking card rather than a separate
        process card, giving the user a single "behind the scenes" view.
        """
        await self._react(msg_id, "OK")

        image_paths: list[str] = []
        file_paths: list[str] = []
        all_temp_paths: list[str] = []

        if attachment_meta:
            os.makedirs(self._INBOUND_DIR, exist_ok=True)
            for att in attachment_meta:
                if att["type"] == "image":
                    att_msg_id = att.get("msg_id", msg_id)
                    data = await self._download_feishu_image(att_msg_id, att["key"])
                    if data:
                        ext = self._guess_image_ext(data)
                        path = os.path.join(self._INBOUND_DIR, f"{att['key']}{ext}")
                        with open(path, "wb") as f:
                            f.write(data)
                        image_paths.append(path)
                        all_temp_paths.append(path)
                        log.info("[Inbound] saved image %s (%d bytes)", path, len(data))
                elif att["type"] == "file":
                    att_msg_id = att.get("msg_id", msg_id)
                    data = await self._download_feishu_file(att_msg_id, att["key"])
                    if data:
                        safe_name = att.get("name", att["key"]).replace("/", "_").replace("..", "_")
                        path = os.path.join(self._INBOUND_DIR, f"{att['key']}_{safe_name}")
                        with open(path, "wb") as f:
                            f.write(data)
                        file_paths.append(path)
                        all_temp_paths.append(path)
                        log.info("[Inbound] saved file %s (%d bytes)", path, len(data))

        try:
            result = StreamResult()
            emitted_event_keys: set[tuple[str, str]] = set()
            streamed_texts: list[str] = []
            saw_stream_event = False

            # --- Streaming card state (2 ordered cards) ---
            _thinking_card_id: str | None = None
            _thinking_acc: list[str] = []       # thinking text chunks
            _tool_blocks: list[str] = []        # formatted tool blocks
            _tool_block_index: dict[str, int] = {}  # tool_use_id -> index in _tool_blocks
            _thinking_dirty = False             # covers both thinking and tool updates

            _result_card_id: str | None = None
            _result_acc: list[str] = []
            _result_dirty = False

            _FLUSH_INTERVAL = 0.5  # seconds between card updates

            def _build_thinking_content() -> str:
                """Build combined thinking + tool calls card content."""
                sections: list[str] = []
                if _thinking_acc:
                    sections.append("💭 **Thinking...**\n\n" + "".join(_thinking_acc))
                if _tool_blocks:
                    n = len(_tool_blocks)
                    header = f"**🔧 Tool Calls ({n})**"
                    body = "\n\n---\n\n".join(_tool_blocks)
                    sections.append(f"{header}\n\n{body}")
                content = "\n\n---\n\n".join(sections)
                if len(content) > cls.MAX_REPLY:
                    content = content[:cls.MAX_REPLY] + "\n\n... (truncated)"
                return content

            cls = self.__class__

            async def _ensure_thinking_card():
                """Ensure thinking card exists (create if needed)."""
                nonlocal _thinking_card_id
                if _thinking_card_id is None and (_thinking_acc or _tool_blocks):
                    _thinking_card_id = await self._reply_card(msg_id, _build_thinking_content())

            async def _ensure_result_card():
                """Ensure result card exists, creating thinking card first if needed."""
                nonlocal _result_card_id
                if _result_card_id is None and _result_acc:
                    await _ensure_thinking_card()
                    full = "".join(_result_acc)
                    _result_card_id = await self._reply_card(msg_id, full)

            async def _flush_cards():
                """Background flusher: periodically update cards with accumulated text."""
                nonlocal _thinking_dirty, _result_dirty
                nonlocal _thinking_card_id, _result_card_id
                while True:
                    await asyncio.sleep(_FLUSH_INTERVAL)

                    # Flush thinking card (includes tool calls)
                    if _thinking_dirty and (_thinking_acc or _tool_blocks):
                        _thinking_dirty = False
                        full = _build_thinking_content()
                        if _thinking_card_id:
                            try:
                                await self._update_card(_thinking_card_id, full)
                            except Exception:
                                log.debug("flush thinking card update failed")
                        else:
                            _thinking_card_id = await self._reply_card(msg_id, full)

                    # Flush result card
                    if _result_dirty and _result_acc:
                        _result_dirty = False
                        await _ensure_thinking_card()
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
                async for event in self.bridge.stream_ask(topic_id, text, image_paths=image_paths or None, file_paths=file_paths or None):
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

                        # -- Thinking: accumulate into thinking card --
                        if stream_event.kind == "thinking" and stream_event.text:
                            _thinking_acc.append(stream_event.text)
                            _thinking_dirty = True
                            continue

                        # -- Result: accumulate into result card --
                        if stream_event.kind == "result" and stream_event.text:
                            _result_acc.append(stream_event.text)
                            _result_dirty = True
                            continue

                        # -- Tool use: add block to thinking card --
                        if stream_event.kind == "tool_use":
                            block = self._format_stream_event(stream_event)
                            if block:
                                idx = len(_tool_blocks)
                                _tool_blocks.append(block)
                                if stream_event.tool_use_id:
                                    _tool_block_index[stream_event.tool_use_id] = idx
                                _thinking_dirty = True
                                log.info(
                                    "[RenderProcess] action=add_tool_use tool=%s id=%s block_idx=%d",
                                    stream_event.tool_name or "-",
                                    stream_event.tool_use_id or "-",
                                    idx,
                                )
                            continue

                        # -- Tool result: merge into existing tool block in thinking card --
                        if stream_event.kind == "tool_result":
                            if stream_event.tool_use_id and stream_event.tool_use_id in _tool_block_index:
                                idx = _tool_block_index[stream_event.tool_use_id]
                                old_block = _tool_blocks[idx]
                                merged = self._merge_tool_stream_blocks(old_block, stream_event)
                                if merged and merged != old_block:
                                    _tool_blocks[idx] = merged
                                    _thinking_dirty = True
                                    log.info(
                                        "[RenderProcess] action=merge_tool_result tool=%s id=%s block_idx=%d",
                                        stream_event.tool_name or "-",
                                        stream_event.tool_use_id,
                                        idx,
                                    )
                            else:
                                # Unbound tool result — add as standalone block
                                block = self._format_stream_event(stream_event)
                                if block:
                                    _tool_blocks.append(block)
                                    _thinking_dirty = True
                                    log.info(
                                        "[RenderProcess] action=add_unbound_tool_result tool=%s id=%s",
                                        stream_event.tool_name or "-",
                                        stream_event.tool_use_id or "-",
                                    )
                            continue

                        # -- Text events: accumulate for dedup tracking --
                        if stream_event.kind == "text":
                            streamed_texts.append(stream_event.text)

                        # Skip duplicate text/result events
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
                        await self._reply_card(msg_id, block)
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
            # Order matters: thinking (with tools) -> result
            if _thinking_acc or _tool_blocks:
                # Change header from "Thinking..." to "Thinking" for final state
                full = _build_thinking_content().replace("Thinking...", "Thinking", 1)
                if _thinking_card_id:
                    try:
                        await self._update_card(_thinking_card_id, full)
                    except Exception:
                        pass
                else:
                    _thinking_card_id = await self._reply_card(msg_id, full)

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
                    # Ensure thinking card exists before creating result card
                    if (_thinking_acc or _tool_blocks) and not _thinking_card_id:
                        pass  # Already handled above
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
        finally:
            if all_temp_paths:
                self._cleanup_inbound_attachments(all_temp_paths)

    @staticmethod
    def _event_dedup_key(event: StreamEvent) -> tuple[str, str] | None:
        if event.kind in {"text", "result"}:
            text = event.text.strip()
            return (event.kind, text) if text else None
        return None

    @classmethod
    def _build_process_card(cls, blocks: list[str]) -> str:
        """Build the unified process card content from accumulated tool blocks.

        Shows all tool calls in a single card with a header indicating count.
        """
        if not blocks:
            return ""
        n = len(blocks)
        header = f"**🔧 Tool Calls ({n})**\n\n"
        # Join blocks with separator
        body = "\n\n---\n\n".join(blocks)
        content = header + body
        if len(content) > cls.MAX_REPLY:
            content = content[:cls.MAX_REPLY] + "\n\n... (truncated)"
        return content

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
            return f"**{title}**\n`{text}`" if text else f"**{title}**"
        if event.kind == "tool_result":
            if len(text) > cls.MAX_TOOL_OUTPUT:
                text = text[:cls.MAX_TOOL_OUTPUT] + "..."
            title = event.tool_name or "unknown"
            return f"📦 **{title}** result\n> {cls._blockquote(text)}" if text else f"📦 **{title}** result"
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
            text = await self._resolve_images(text)
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
            text = await self._resolve_images(text)
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

    # ── Image upload / resolution ──────────────────────────────────

    async def _upload_image_to_feishu(self, url: str) -> str | None:
        """Download image from URL and upload to Feishu. Returns image_key or None."""
        if url in self._image_cache:
            return self._image_cache[url]
        try:
            req_ = urllib.request.Request(url, headers={"User-Agent": "DeerFlow/1.0"})
            data = await asyncio.to_thread(
                lambda: urllib.request.urlopen(req_, timeout=10).read()
            )
            if not data or len(data) < 100:  # skip tiny / empty responses
                self._image_cache[url] = None
                return None

            body = (
                self._sdk["CreateImageBody"]
                .builder()
                .image_type("message")
                .image(io.BytesIO(data))
                .build()
            )
            req = self._sdk["CreateImageReq"].builder().request_body(body).build()
            resp = await asyncio.to_thread(self._api_client.im.v1.image.create, req)
            code = getattr(resp, "code", None)
            if code == 0 and resp.data:
                key = resp.data.image_key
                log.info("[ImageUpload] ok url=%s key=%s size=%d", url[:80], key, len(data))
                self._image_cache[url] = key
                return key
            else:
                log.warning("[ImageUpload] API error url=%s code=%s", url[:80], code)
        except Exception:
            log.debug("[ImageUpload] failed url=%s", url[:80], exc_info=True)
        self._image_cache[url] = None
        return None

    async def _resolve_images(self, text: str) -> str:
        """Try to upload images in text to Feishu, replacing URLs with image_keys.

        Successfully uploaded images become inline Feishu images.
        Failed uploads are left unchanged — _sanitize_for_card() converts them
        to plain links as a safety net.
        """
        if not text:
            return text
        matches = list(self._RE_MD_IMAGE.finditer(text))
        if not matches:
            return text

        # Collect unique URLs
        urls = list({m.group(2) for m in matches})

        # Upload concurrently (with short timeout per image)
        results = await asyncio.gather(
            *(self._upload_image_to_feishu(u) for u in urls),
            return_exceptions=True,
        )
        url_to_key: dict[str, str] = {}
        for url, res in zip(urls, results):
            if isinstance(res, str) and res:
                url_to_key[url] = res

        if not url_to_key:
            return text  # no successful uploads

        # Replace URLs with image_keys for successful uploads
        def _replacer(m: re.Match) -> str:
            alt = m.group(1)
            url = m.group(2)
            if url in url_to_key:
                return f"![{alt}]({url_to_key[url]})"
            return m.group(0)  # leave unchanged

        return self._RE_MD_IMAGE.sub(_replacer, text)

    # ── Regex patterns for card content sanitization ──────────────
    # Markdown image: ![alt](url) or ![alt](url "title")
    _RE_MD_IMAGE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
    # HTML img tag
    _RE_HTML_IMG = re.compile(r'<img\b[^>]*src=["\x27]([^"\x27]+)["\x27][^>]*/?\s*>', re.IGNORECASE)
    # Feishu image token pattern (used in Mira content)
    _RE_FEISHU_IMAGE_TOKEN = re.compile(r'<image\s+token=["\x27]([^"\x27]+)["\x27][^/]*/?\s*>')

    @classmethod
    def _sanitize_for_card(cls, text: str) -> str:
        """Sanitize markdown content for Feishu card compatibility.

        Feishu interactive cards require uploaded image_keys for any image
        content.  Raw image URLs / markdown images cause API error 11310.
        This method converts image references to plain-text links.
        """
        if not text:
            return text

        # ![alt](url) -> [alt](url)  or  [image](url)
        def _replace_md_img(m: re.Match) -> str:
            alt = m.group(1).strip() or "image"
            url = m.group(2)
            return f"[{alt}]({url})"

        text = cls._RE_MD_IMAGE.sub(_replace_md_img, text)

        # <img src="url" ...> -> [image](url)
        def _replace_html_img(m: re.Match) -> str:
            url = m.group(1)
            return f"[image]({url})"

        text = cls._RE_HTML_IMG.sub(_replace_html_img, text)

        # <image token="xxx" .../> (Feishu internal) -> [image:xxx]
        def _replace_feishu_img(m: re.Match) -> str:
            token = m.group(1)
            return f"[image:{token}]"

        text = cls._RE_FEISHU_IMAGE_TOKEN.sub(_replace_feishu_img, text)

        return text

    @staticmethod
    def _card(text: str) -> str:
        text = FeishuBot._sanitize_for_card(text)
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
