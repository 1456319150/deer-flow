"""Feishu → Claude Code Gateway

Minimal relay: Feishu/WeChat messages in, ttadk Claude Code out.
No LangGraph, no multi-agent, no frontend — just a message bridge.

Architecture:
    gateway.py        — Config, Bridge, StreamResult types, main()
    feishu_bot.py     — Feishu WebSocket channel adapter
    weixin_bot.py     — WeChat iLink channel adapter
    weixin_channel.py — iLink protocol layer (login, poll, send)

Session continuity:
    topic_id → ttadk session_id via --resume
    Feishu: topic_id = root_id or msg_id
    WeChat: topic_id = wx_{from_user}

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


def _get_initial_message() -> tuple[str, str] | None:
    topic_id = os.environ.get("GATEWAY_INITIAL_TOPIC_ID", "").strip()
    prompt = os.environ.get("GATEWAY_INITIAL_PROMPT", "").strip()
    if not topic_id or not prompt:
        return None
    return topic_id, prompt


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
                limit=10 * 1024 * 1024,  # 10MB; default 64KB too small for Claude Code output
            )
        except FileNotFoundError:
            r = StreamResult(assistant_texts=[f"❌ Command not found: {self.ttadk_cmd}. Is ttadk installed and in PATH?"])
            yield {"type": "final", "result": r}
            return

        state = StreamState()
        output_len = 0

        try:
            async with asyncio.timeout(self.timeout):
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace")
                    output_len += len(decoded)
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

        log.info("[Bridge] exit=%d output=%d chars", proc.returncode, output_len)

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

    @staticmethod
    def _stringify_tool_result_content(content_val: Any) -> str:
        if isinstance(content_val, list):
            parts = []
            for item in content_val:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()
        if isinstance(content_val, str):
            return content_val.strip()
        return str(content_val).strip()

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
                    output = cls._stringify_tool_result_content(block.get("content", ""))
                    tool_name = ""
                    if tool_use_id in state.pending_tools:
                        state.pending_tools[tool_use_id].output_text = output
                        tool_name = state.pending_tools[tool_use_id].name
                    cls._emit_stream_event(emitted, "tool_result", text=output, tool_name=tool_name, tool_use_id=tool_use_id)

        elif event_type == "user":
            message = event.get("message", {})
            content = message.get("content", [])
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_use_id = block.get("tool_use_id", "")
                output = cls._stringify_tool_result_content(block.get("content", ""))
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

    log.info("[Init] file logging enabled at %s", log_path)

    initial_message = _get_initial_message()
    if initial_message:
        initial_topic_id, initial_prompt = initial_message
        log.info("[Init] sending initial prompt to topic=%s prompt=%r", initial_topic_id, _preview_text(initial_prompt))
        result = await bridge.ask(initial_topic_id, initial_prompt)
        if result.reply_text:
            log.info("[Init] initial reply: %s", _preview_text(result.reply_text))
        else:
            log.info("[Init] initial prompt finished with empty reply")

    # Feishu channel
    from feishu_bot import FeishuBot
    feishu_bot = FeishuBot(cfg["feishu"], bridge)
    await feishu_bot.start()

    # WeChat channel (optional)
    weixin_cfg = cfg.get("weixin", {})
    if weixin_cfg.get("enabled"):
        from weixin_bot import WeixinBot
        weixin_bot = WeixinBot(weixin_cfg, bridge)
        await weixin_bot.start()

    log.info("🚀 Gateway running. Ctrl+C to stop.")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
