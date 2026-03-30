"""Feishu → Claude Code Gateway

Minimal relay: Feishu messages in, ttadk Claude Code out.
No LangGraph, no multi-agent, no frontend — just a message bridge.

Architecture:
    Feishu WebSocket (lark-oapi)
        → FeishuBot._on_message()
            → ClaudeCodeBridge.ask(topic_id, text)
                → ttadk code -t claude (subprocess, stream-json mode)
            ← StreamResult (thinking + tool_use + text)
        ← update Feishu card with rich result

Session continuity:
    topic_id (root_id or msg_id) → ttadk session_id via --resume

stream-json workaround:
    Claude Code's --output-format json has a known bug where `result` is
    often empty despite the model generating a full response. We use
    --output-format stream-json --verbose instead, and extract the actual
    text from {"type":"assistant"} events in the stream.
    See: https://github.com/anthropics/claude-code/issues/38706
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import yaml

log = logging.getLogger("gateway")


# ===========================================================================
# Config
# ===========================================================================

def load_dotenv(path: str = ".env") -> None:
    """Load .env file into os.environ. No extra dependency needed."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    for q in ('"', "'"):
                        if value.startswith(q) and value.endswith(q):
                            value = value[1:-1]
                            break
                    os.environ.setdefault(key, value)
    except FileNotFoundError:
        pass


def load_config(path: str = "config.yaml") -> dict:
    load_dotenv()
    with open(path) as f:
        raw = f.read()
    resolved = re.sub(r"\$\{(\w+)}", lambda m: os.environ.get(m.group(1), ""), raw)
    return yaml.safe_load(resolved)


def _preview_text(text: str, limit: int = 120) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


def _reset_log_file(path: str) -> None:
    with open(path, "w", encoding="utf-8"):
        pass


# ===========================================================================
# Stream Result
# ===========================================================================

@dataclass
class ToolCall:
    """A single tool invocation: what was called and its result."""
    name: str
    input_text: str
    output_text: str = ""


@dataclass
class StreamEvent:
    """A single user-visible event parsed from Claude Code stream-json."""
    kind: str
    text: str = ""
    tool_name: str = ""
    tool_use_id: str = ""


@dataclass
class StreamResult:
    """Structured result from parsing stream-json events."""
    thinking: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_texts: list[str] = field(default_factory=list)
    result_text: str = ""
    session_id: str | None = None

    @property
    def reply_text(self) -> str:
        """Primary text to show: result if present, else combined assistant texts."""
        if self.result_text:
            return self.result_text
        if self.assistant_texts:
            return "\n\n".join(self.assistant_texts)
        return ""

    @property
    def is_empty(self) -> bool:
        return not self.reply_text and not self.tool_calls


@dataclass
class StreamState:
    """Mutable parser state shared by full and incremental stream parsing."""
    result: StreamResult = field(default_factory=StreamResult)
    pending_tools: dict[str, ToolCall] = field(default_factory=dict)


# ===========================================================================
# Claude Code Bridge
# ===========================================================================

class ClaudeCodeBridge:
    """Wraps ttadk CLI to talk to Claude Code."""

    def __init__(self, cfg: dict):
        self.ttadk_cmd: str = cfg.get("ttadk_cmd", "ttadk")
        self.model: str = cfg.get("model", "gpt-5.4")
        self.target: str = cfg.get("target", "claude")
        self.timeout: int = cfg.get("timeout", 600)
        self.allowed_tools: str = cfg.get("allowed_tools", "")
        self.instruction: str = cfg.get("instruction", "")
        self.session_store_path: str = cfg.get("session_store_path", ".gateway-sessions.json")
        self._session_lock = asyncio.Lock()
        self._sessions: dict[str, str] = self._load_sessions()

    async def ask(self, topic_id: str, prompt: str) -> StreamResult:
        """Send prompt to Claude Code, return structured StreamResult."""
        result: StreamResult | None = None
        async for event in self.stream_ask(topic_id, prompt):
            if event["type"] == "final":
                result = event["result"]
        return result or StreamResult()

    async def stream_ask(self, topic_id: str, prompt: str) -> AsyncIterator[dict[str, Any]]:
        """Stream Claude Code output as immediate events plus one final result."""
        session_id = self._sessions.get(topic_id)
        full_prompt = f"{self.instruction}\n\n{prompt}" if self.instruction else prompt
        cmd = self._build_cmd(full_prompt, session_id)

        log.info("[Bridge] topic=%s session=%s prompt=%d chars", topic_id, session_id, len(prompt))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            r = StreamResult(assistant_texts=[f"❌ Command not found: {self.ttadk_cmd}. Is ttadk installed and in PATH?"])
            yield {"type": "final", "result": r}
            return

        state = StreamState()
        raw_lines: list[str] = []

        try:
            async with asyncio.timeout(self.timeout):
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace")
                    raw_lines.append(decoded)
                    for event in self._consume_stream_line(state, decoded):
                        if event["type"] == "session" and event["session_id"]:
                            await self._remember_session(topic_id, event["session_id"])
                        else:
                            yield event
                await proc.wait()
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            r = StreamResult(assistant_texts=[f"⏰ Claude Code timed out after {self.timeout}s"])
            yield {"type": "final", "result": r}
            return

        raw = "".join(raw_lines)
        log.info("[Bridge] exit=%d output=%d chars", proc.returncode, len(raw))

        if state.result.session_id:
            await self._remember_session(topic_id, state.result.session_id)

        yield {"type": "final", "result": state.result}

    @staticmethod
    def _event_to_lines(event: dict) -> list[str]:
        return [json.dumps(event, ensure_ascii=False)]

    @classmethod
    def _consume_stream_line(cls, state: StreamState, line: str) -> list[dict[str, Any]]:
        stripped = line.strip()
        if not stripped.startswith("{"):
            return []
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        return cls._consume_event(state, event)

    @classmethod
    def _emit_stream_event(
        cls,
        emitted: list[dict[str, Any]],
        kind: str,
        text: str = "",
        tool_name: str = "",
        tool_use_id: str = "",
    ) -> None:
        log.info(
            "[BridgeEvent] kind=%s tool=%s tool_use_id=%s text_len=%d preview=%r",
            kind,
            tool_name or "-",
            tool_use_id or "-",
            len(text),
            _preview_text(text),
        )
        emitted.append({
            "type": "stream_event",
            "event": StreamEvent(kind=kind, text=text, tool_name=tool_name, tool_use_id=tool_use_id),
        })

    @classmethod
    def _consume_event(cls, state: StreamState, event: dict[str, Any]) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        result = state.result
        event_type = event.get("type")

        if event_type == "assistant":
            message = event.get("message", {})
            content = message.get("content", [])
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")

                if block_type == "text":
                    text = block.get("text", "").strip()
                    if text:
                        result.assistant_texts.append(text)
                        cls._emit_stream_event(emitted, "text", text=text)

                elif block_type == "thinking":
                    thinking = block.get("thinking", "").strip()
                    if thinking:
                        result.thinking.append(thinking)
                        cls._emit_stream_event(emitted, "thinking", text=thinking)

                elif block_type == "tool_use":
                    tool_id = block.get("id", "")
                    name = block.get("name", "unknown")
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        if "command" in inp:
                            input_text = inp["command"]
                        elif "file_path" in inp:
                            input_text = inp["file_path"]
                        elif "query" in inp:
                            input_text = inp["query"]
                        else:
                            input_text = json.dumps(inp, ensure_ascii=False)
                    else:
                        input_text = str(inp)
                    tc = ToolCall(name=name, input_text=input_text)
                    result.tool_calls.append(tc)
                    if tool_id:
                        state.pending_tools[tool_id] = tc
                    cls._emit_stream_event(emitted, "tool_use", text=input_text, tool_name=name, tool_use_id=tool_id)

                elif block_type == "tool_result":
                    tool_use_id = block.get("tool_use_id", "")
                    content_val = block.get("content", "")
                    if isinstance(content_val, list):
                        parts = []
                        for item in content_val:
                            if isinstance(item, dict) and item.get("type") == "text":
                                parts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                parts.append(item)
                        output = "\n".join(parts)
                    elif isinstance(content_val, str):
                        output = content_val
                    else:
                        output = str(content_val)
                    output = output.strip()
                    tool_name = ""
                    if tool_use_id in state.pending_tools:
                        state.pending_tools[tool_use_id].output_text = output
                        tool_name = state.pending_tools[tool_use_id].name
                    cls._emit_stream_event(emitted, "tool_result", text=output, tool_name=tool_name, tool_use_id=tool_use_id)

        elif event_type == "system" and event.get("subtype") == "init":
            session_id = event.get("session_id")
            if session_id:
                result.session_id = session_id
                emitted.append({"type": "session", "session_id": session_id})

        elif event_type == "result":
            result.result_text = (event.get("result") or "").strip()
            log.info(
                "[BridgeResult] result_len=%d preview=%r session=%s",
                len(result.result_text),
                _preview_text(result.result_text),
                event.get("session_id") or "-",
            )
            if result.result_text:
                cls._emit_stream_event(emitted, "result", text=result.result_text)
            session_id = event.get("session_id")
            if session_id:
                result.session_id = session_id
                emitted.append({"type": "session", "session_id": session_id})

        return emitted

    @staticmethod
    def _parse_stream(raw: str) -> StreamResult:
        """Parse stream-json output from Claude Code CLI."""
        state = StreamState()
        for line in raw.splitlines():
            ClaudeCodeBridge._consume_stream_line(state, line)
        return state.result

    def _load_sessions(self) -> dict[str, str]:
        try:
            with open(self.session_store_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError):
            log.warning("[Bridge] failed to load session store: %s", self.session_store_path, exc_info=True)
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if k and v}

    async def _remember_session(self, topic_id: str, session_id: str) -> None:
        if not topic_id or not session_id or self._sessions.get(topic_id) == session_id:
            return
        async with self._session_lock:
            self._sessions[topic_id] = session_id
            await asyncio.to_thread(self._save_sessions)
        log.info("[Bridge] session: topic=%s → %s", topic_id, session_id)

    def _save_sessions(self) -> None:
        directory = os.path.dirname(self.session_store_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = f"{self.session_store_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._sessions, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, self.session_store_path)

    def _build_cmd(self, prompt: str, session_id: str | None) -> list[str]:
        """Build ttadk CLI command list."""
        safe = lambda s: "'" + s.replace("\n", "\\n").replace("\r", "").replace("'", "'\"'\"'") + "'"
        parts = [
            f"-p {safe(prompt)}",
            "--output-format stream-json",
            "--verbose",
        ]
        if session_id:
            parts.insert(0, f"--resume {session_id}")
        if self.allowed_tools:
            parts.append(f"--allowedTools {self.allowed_tools}")
        return [self.ttadk_cmd, "code", "-t", self.target, "-m", self.model, "-a", " ".join(parts)]


# ===========================================================================
# Feishu Bot
# ===========================================================================

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
        """Full message lifecycle: react → incremental card updates → done."""
        await self._react(msg_id, "OK")
        card_id = await self._reply_card(msg_id, "🤔 Thinking...")
        rendered_blocks: list[str] = []
        streamed_texts: list[str] = []
        seen_result_block = False
        result = StreamResult()

        try:
            async for event in self.bridge.stream_ask(topic_id, text):
                if event["type"] == "stream_event":
                    stream_event = event["event"]
                    log.info(
                        "[RenderEvent] kind=%s tool=%s text_len=%d preview=%r",
                        stream_event.kind,
                        stream_event.tool_name or "-",
                        len(stream_event.text),
                        _preview_text(stream_event.text),
                    )
                    if stream_event.kind == "text":
                        streamed_texts.append(stream_event.text)
                    if stream_event.kind == "result":
                        current_reply = "\n\n".join(streamed_texts).strip()
                        if not stream_event.text or stream_event.text == current_reply:
                            seen_result_block = True
                            continue
                        seen_result_block = True
                    block = self._format_stream_event(stream_event)
                    if not block:
                        continue
                    rendered_blocks.append(block)
                    content = self._format_stream_transcript(rendered_blocks)
                    log.info(
                        "[RenderCard] blocks=%d content_len=%d last_kind=%s",
                        len(rendered_blocks),
                        len(content),
                        stream_event.kind,
                    )
                    if card_id:
                        await self._update_card(card_id, content)
                    else:
                        card_id = await self._reply_card(msg_id, content)
                elif event["type"] == "final":
                    result = event["result"]
        except Exception as e:
            log.exception("Bridge error")
            result = StreamResult(assistant_texts=[f"❌ Error: {e}"])

        if not rendered_blocks:
            card_content = self._format_result(result)
            log.info("[RenderFallback] content_len=%d", len(card_content))
            if card_id:
                await self._update_card(card_id, card_content)
            else:
                await self._reply_card(msg_id, card_content)
        elif result.reply_text and not any(block == result.reply_text for block in rendered_blocks):
            final_text = result.reply_text
            reply_so_far = "\n\n".join(streamed_texts).strip()
            if final_text != reply_so_far:
                log.info("[RenderFinalText] text_len=%d preview=%r", len(final_text), _preview_text(final_text))
                rendered_blocks.append(final_text)
                content = self._format_stream_transcript(rendered_blocks)
                if card_id:
                    await self._update_card(card_id, content)
                else:
                    await self._reply_card(msg_id, content)
        elif result.result_text and not seen_result_block:
            final_block = self._format_stream_event(StreamEvent(kind="result", text=result.result_text))
            if final_block:
                reply_so_far = "\n\n".join(result.assistant_texts).strip()
                if result.result_text != reply_so_far:
                    rendered_blocks.append(final_block)
                    content = self._format_stream_transcript(rendered_blocks)
                    if card_id:
                        await self._update_card(card_id, content)
                    else:
                        await self._reply_card(msg_id, content)

        await self._react(msg_id, "DONE")

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
    def _format_stream_transcript(cls, blocks: list[str]) -> str:
        content = "\n\n---\n\n".join(block for block in blocks if block)
        if len(content) > cls.MAX_REPLY:
            return content[:cls.MAX_REPLY] + "\n\n... (truncated, response too long)"
        return content

    @classmethod
    def _format_result(cls, result: StreamResult) -> str:
        """Format StreamResult into rich Markdown for Feishu card.

        Layout:
        1. 💭 Thinking (collapsed if long) — shows Claude's reasoning
        2. 🔧 Tool Calls — shows what commands/files were accessed
        3. 📝 Reply — the actual response text
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
            data = getattr(resp, "data", None)
            return getattr(data, "message_id", None) if data else None
        except Exception:
            log.exception("Reply card failed for %s", msg_id)
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
            await asyncio.to_thread(self._api_client.im.v1.message.patch, req)
        except Exception:
            log.exception("Update card failed for %s", card_id)

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


# ===========================================================================
# Main
# ===========================================================================

async def main() -> None:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "gateway.log")
    _reset_log_file(log_path)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )

    cfg = load_config()
    bridge = ClaudeCodeBridge(cfg.get("claude", {}))
    bot = FeishuBot(cfg["feishu"], bridge)

    log.info("[Init] file logging enabled at %s", log_path)

    await bot.start()

    log.info("🚀 Gateway running. Ctrl+C to stop.")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
