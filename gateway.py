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

# Max auto-retries when CLI exits with stop_reason=tool_use and empty result
_MAX_TOOL_USE_RETRIES = 2


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
    resolved = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), raw)
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
class UsageSummary:
    """Token and billing summary from the final Claude Code result event."""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def has_values(self) -> bool:
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.cache_creation_input_tokens,
                self.cache_read_input_tokens,
                self.total_tokens,
                self.cost_usd,
            )
        )


@dataclass
class StreamResult:
    """Structured result from parsing stream-json events."""
    thinking: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_texts: list[str] = field(default_factory=list)
    result_text: str = ""
    stop_reason: str | None = None
    session_id: str | None = None
    usage: UsageSummary | None = None

    @property
    def reply_text(self) -> str:
        """Primary text to show: result if present, else combined assistant texts.

        When stop_reason is tool_use and no text is available, synthesize a
        fallback message so the user is never left with a blank reply.
        """
        if self.result_text:
            return self.result_text
        if self.assistant_texts:
            return "\n\n".join(self.assistant_texts)
        # Fallback: model requested tool use but CLI exited before producing text
        if self.tool_calls:
            names = ", ".join(tc.name for tc in self.tool_calls[:5])
            remaining = len(self.tool_calls) - 5
            suffix = f" 等 {remaining + 5} 个工具" if remaining > 0 else ""
            return (
                f"⚠️ CLI 请求执行工具 ({names}{suffix}) 但未返回最终回复。\n"
                f"这通常是因为 CLI 在工具执行前退出。请重试，或检查网关日志获取详情。"
            )
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
        self.provider: str = self._resolve_provider(cfg.get("provider"), self.target)
        self.timeout: int = cfg.get("timeout", 600)
        self.allowed_tools: str = cfg.get("allowed_tools", "")
        self.instruction: str = cfg.get("instruction", "")
        self.session_store_path: str = cfg.get("session_store_path", ".gateway-sessions.json")
        self._session_lock = asyncio.Lock()
        self._sessions: dict[str, str] = self._load_sessions()

    @staticmethod
    def _resolve_provider(provider: Any, target: str) -> str:
        if isinstance(provider, str) and provider.strip():
            return provider.strip().lower()
        return "codex" if str(target).strip().lower() == "codex" else "claude"

    def _build_full_prompt(self, prompt: str) -> str:
        return f"{self.instruction}\n\n{prompt}" if self.instruction else prompt

    def _retry_prompt(self, _result: StreamResult) -> str:
        return "continue"

    async def ask(self, topic_id: str, prompt: str) -> StreamResult:
        """Send prompt to Claude Code, return structured StreamResult."""
        result: StreamResult | None = None
        async for event in self.stream_ask(topic_id, prompt):
            if event["type"] == "final":
                result = event["result"]
        return result or StreamResult()

    async def stream_ask(self, topic_id: str, prompt: str) -> AsyncIterator[dict[str, Any]]:
        """Stream Claude Code output with auto-retry on tool_use exit.

        When the CLI exits with stop_reason=tool_use and empty result, it means
        the model requested a tool but the CLI stopped before executing it.
        This wrapper auto-retries by resuming the session, giving the model
        a continuation prompt so it can proceed.
        """
        for attempt in range(_MAX_TOOL_USE_RETRIES + 1):
            final_result: StreamResult | None = None
            async for event in self._stream_ask_once(topic_id, prompt):
                if event["type"] == "final":
                    final_result = event["result"]
                else:
                    yield event

            if final_result is None:
                return

            # Check if we should retry: stop_reason=tool_use with empty result
            should_retry = (
                final_result.stop_reason == "tool_use"
                and not final_result.result_text
                and not final_result.assistant_texts
                and final_result.session_id
                and attempt < _MAX_TOOL_USE_RETRIES
            )

            if should_retry:
                log.warning(
                    "[Bridge] AUTO-RETRY %d/%d: stop_reason=tool_use with empty result, "
                    "resuming session %s",
                    attempt + 1, _MAX_TOOL_USE_RETRIES, final_result.session_id,
                )
                prompt = self._retry_prompt(final_result)
                continue

            # No retry needed — yield the final result
            yield {"type": "final", "result": final_result}
            return

        # Exhausted retries — yield whatever we have
        log.error(
            "[Bridge] exhausted %d tool_use retries — returning last result",
            _MAX_TOOL_USE_RETRIES,
        )
        if final_result is not None:
            yield {"type": "final", "result": final_result}

    async def _stream_ask_once(self, topic_id: str, prompt: str) -> AsyncIterator[dict[str, Any]]:
        """Single attempt: stream Claude Code output as events plus one final result."""
        session_id = self._sessions.get(topic_id)
        full_prompt = self._build_full_prompt(prompt)
        cmd = self._build_cmd(full_prompt, session_id)

        log.info("[Bridge] topic=%s session=%s prompt=%d chars", topic_id, session_id, len(prompt))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,  # separate pipe
                limit=10 * 1024 * 1024,  # 10MB; default 64KB too small for Claude Code output
            )
        except FileNotFoundError:
            r = StreamResult(assistant_texts=[f"❌ Command not found: {self.ttadk_cmd}. Is ttadk installed and in PATH?"])
            yield {"type": "final", "result": r}
            return

        state = StreamState()
        output_len = 0

        # DIAG: line counters + raw tail buffer for empty-result diagnostics
        json_line_count = 0
        non_json_line_count = 0
        event_types_seen: list[str] = []
        raw_tail: list[str] = []  # last 20 lines (truncated to 300 chars each)

        stderr_task: asyncio.Task[None] | None = None
        try:
            async with asyncio.timeout(self.timeout):
                assert proc.stdout is not None
                # Drain stderr concurrently — log each line with [STDERR] prefix
                stderr_lines: list[str] = []

                async def _drain_stderr():
                    assert proc.stderr is not None
                    while True:
                        line = await proc.stderr.readline()
                        if not line:
                            break
                        text = line.decode("utf-8", errors="replace").strip()
                        if text:
                            stderr_lines.append(text)
                            log.warning("[Bridge] [STDERR] %s", text)

                stderr_task = asyncio.create_task(_drain_stderr())
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace")
                    output_len += len(decoded)

                    # DIAG: classify and buffer raw lines
                    stripped = decoded.strip()
                    if stripped.startswith("{"):
                        json_line_count += 1
                        # DIAG: track event type for summary
                        try:
                            peek = json.loads(stripped)
                            etype = peek.get("type", "?")
                            event_types_seen.append(etype)
                        except json.JSONDecodeError:
                            event_types_seen.append("json_err")
                    else:
                        non_json_line_count += 1
                        if stripped:
                            log.debug(f"[Bridge] non-JSON stdout line: {stripped!r}")
                    raw_tail.append(stripped[:2000])
                    if len(raw_tail) > 20:
                        raw_tail.pop(0)

                    for event in self._consume_stream_line(state, decoded, provider=self.provider):
                        if event["type"] == "session" and event["session_id"]:
                            await self._remember_session(topic_id, event["session_id"])
                        else:
                            yield event
                await proc.wait()
                if stderr_task is not None:
                    await stderr_task
                log.warning(f"[Bridge] process exited: returncode={proc.returncode}")
        except TimeoutError:
            log.error(f"[Bridge] TIMEOUT after {self.timeout}s, killing process")
            if stderr_task is not None:
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
            proc.kill()
            await proc.wait()
            r = StreamResult(assistant_texts=[f"⏰ Claude Code timed out after {self.timeout}s"])
            yield {"type": "final", "result": r}
            return

        # DIAG: enhanced exit log with line counters and event type summary
        log.info(
            "[Bridge] exit=%d output=%d chars json_lines=%d non_json_lines=%d stderr_lines=%d event_types=%s",
            proc.returncode, output_len, json_line_count, non_json_line_count, len(stderr_lines), event_types_seen,
        )

        # DIAG: dump raw tail when result is empty — this is the key diagnostic
        if state.result.is_empty:
            log.warning(
                "[Bridge] EMPTY RESULT diagnostic — assistant_texts=%d tool_calls=%d "
                "result_text_len=%d thinking=%d stop_reason=%s",
                len(state.result.assistant_texts), len(state.result.tool_calls),
                len(state.result.result_text), len(state.result.thinking),
                state.result.stop_reason,
            )
            log.warning("[Bridge] EMPTY RESULT — dumping last %d raw output lines:", len(raw_tail))
            for i, rl in enumerate(raw_tail):
                log.warning("[Bridge] raw_tail[%02d]: %s", i, rl)
            if stderr_lines:
                log.warning("[Bridge] EMPTY RESULT — stderr output (%d lines):", len(stderr_lines))
                for i, sl in enumerate(stderr_lines[-20:]):
                    log.warning("[Bridge] stderr[%02d]: %s", i, sl[:500])

        if state.result.session_id:
            await self._remember_session(topic_id, state.result.session_id)

        yield {"type": "final", "result": state.result}

    @staticmethod
    def _event_to_lines(event: dict) -> list[str]:
        return [json.dumps(event, ensure_ascii=False)]

    @staticmethod
    def _parse_json_event(line: str) -> dict[str, Any] | None:
        stripped = line.strip()
        if not stripped.startswith("{"):
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            log.debug(f"[Bridge] non-JSON line in stream: {stripped!r}")
            return None

    @classmethod
    def _consume_stream_line(
        cls,
        state: StreamState,
        line: str,
        provider: str = "claude",
    ) -> list[dict[str, Any]]:
        if provider == "codex":
            return cls._consume_stream_line_codex(state, line)
        return cls._consume_stream_line_claude(state, line)

    @classmethod
    def _consume_stream_line_claude(cls, state: StreamState, line: str) -> list[dict[str, Any]]:
        event = cls._parse_json_event(line)
        if event is None:
            return []
        return cls._consume_event_claude(state, event)

    @classmethod
    def _consume_stream_line_codex(cls, state: StreamState, line: str) -> list[dict[str, Any]]:
        event = cls._parse_json_event(line)
        if event is None:
            return []
        return cls._consume_event_codex(state, event)

    @classmethod
    def _consume_event(
        cls,
        state: StreamState,
        event: dict[str, Any],
        provider: str = "claude",
    ) -> list[dict[str, Any]]:
        if provider == "codex":
            return cls._consume_event_codex(state, event)
        return cls._consume_event_claude(state, event)

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

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return int(stripped)
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    @classmethod
    def _first_coerced_value(
        cls,
        sources: tuple[dict[str, Any], ...],
        keys: tuple[str, ...],
        coercer: Any,
    ) -> Any:
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in keys:
                if key not in source:
                    continue
                value = coercer(source.get(key))
                if value is not None:
                    return value
        return None

    @classmethod
    def _extract_usage_summary(cls, event: dict[str, Any]) -> UsageSummary | None:
        usage = event.get("usage")
        usage_dict = usage if isinstance(usage, dict) else {}

        model_usage = event.get("modelUsage")
        model_usage_dict: dict[str, Any] = {}
        if isinstance(model_usage, dict):
            for value in model_usage.values():
                if isinstance(value, dict):
                    model_usage_dict = value
                    break

        input_tokens = cls._first_coerced_value(
            (usage_dict, model_usage_dict),
            ("input_tokens", "prompt_tokens", "inputTokens", "promptTokens"),
            cls._coerce_int,
        )
        output_tokens = cls._first_coerced_value(
            (usage_dict, model_usage_dict),
            ("output_tokens", "completion_tokens", "outputTokens", "completionTokens"),
            cls._coerce_int,
        )
        cache_creation_input_tokens = cls._first_coerced_value(
            (usage_dict, model_usage_dict),
            ("cache_creation_input_tokens", "cacheCreationInputTokens"),
            cls._coerce_int,
        )
        cache_read_input_tokens = cls._first_coerced_value(
            (usage_dict, model_usage_dict),
            ("cache_read_input_tokens", "cacheReadInputTokens"),
            cls._coerce_int,
        )
        total_tokens = cls._first_coerced_value(
            (usage_dict, model_usage_dict),
            ("total_tokens", "totalTokens"),
            cls._coerce_int,
        )
        cost_usd = cls._first_coerced_value(
            (event, usage_dict, model_usage_dict),
            ("total_cost_usd", "cost_usd", "totalCostUsd", "costUSD", "cost"),
            cls._coerce_float,
        )

        if total_tokens is None:
            token_values = [
                input_tokens,
                output_tokens,
                cache_creation_input_tokens,
                cache_read_input_tokens,
            ]
            if any(value is not None for value in token_values):
                total_tokens = sum(value or 0 for value in token_values)

        summary = UsageSummary(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )
        return summary if summary.has_values else None

    @classmethod
    def _consume_event_claude(cls, state: StreamState, event: dict[str, Any]) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        result = state.result
        event_type = event.get("type")

        if event_type == "assistant":
            message = event.get("message", {})
            content = message.get("content", [])
            # DIAG: event-level summary — block count and all types at a glance
            block_types_summary = []
            for b in content:
                if isinstance(b, dict):
                    block_types_summary.append(b.get("type", f"NO_TYPE(keys={list(b.keys())[:4]})"))
                else:
                    block_types_summary.append(f"non-dict({type(b).__name__})")
            log.info("[Bridge] assistant event: %d content blocks, types=%s", len(content), block_types_summary)
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")

                # DIAG: log each content block for tool-use debugging
                log.info("[Bridge] content block: type=%s keys=%s", block_type, list(block.keys())[:8])
                if block_type == "tool_use":
                    log.info("[Bridge] tool_use requested: name=%s id=%s", block.get("name"), block.get("id"))


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

                elif block_type == "redacted_thinking":
                    # Extended thinking encrypted block — safe to skip silently
                    log.debug("[Bridge] redacted_thinking block (encrypted extended thinking) — skipped")

                elif block_type is None and "signature" in block:
                    # Encrypted extended thinking block — no type field, just {"signature":"enc_..."}
                    log.debug("[Bridge] encrypted thinking block (signature-only format) — skipped")

                else:
                    log.warning("[Bridge] UNKNOWN block type=%r in assistant event — keys=%s preview=%s",
                                block_type, list(block.keys())[:10],
                                json.dumps(block, ensure_ascii=False, default=str)[:500])

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
            result.stop_reason = event.get("stop_reason")
            result.usage = cls._extract_usage_summary(event)
            log.info(
                "[BridgeResult] result_len=%d preview=%r session=%s",
                len(result.result_text),
                _preview_text(result.result_text),
                event.get("session_id") or "-",
            )
            if result.usage:
                log.info(
                    "[BridgeUsage] input=%s output=%s cache_write=%s cache_read=%s total=%s cost_usd=%s",
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    result.usage.cache_creation_input_tokens,
                    result.usage.cache_read_input_tokens,
                    result.usage.total_tokens,
                    result.usage.cost_usd,
                )

            # DIAG: dump full result event when result text is empty
            if not result.result_text:
                log.warning(
                    "[BridgeResult] EMPTY result text — full event JSON: %s",
                    json.dumps(event, ensure_ascii=False, default=str)[:3000],
                )

            log.info(f"[Bridge] result event: stop_reason={event.get('stop_reason')}, "
                     f"num_turns={event.get('num_turns')}, has_result={bool(event.get('result'))}")

            if event.get("stop_reason") == "tool_use":
                log.warning(f"[Bridge] CLI stopped mid-tool-use! num_turns={event.get('num_turns')} "
                            f"session={event.get('session_id')} — model wanted to continue but CLI exited")

            if result.result_text:
                cls._emit_stream_event(emitted, "result", text=result.result_text)
            session_id = event.get("session_id")
            if session_id:
                result.session_id = session_id
                emitted.append({"type": "session", "session_id": session_id})

        # DIAG: log any event type not handled above
        elif event_type is not None:
            log.warning(
                "[BridgeEvent] UNRECOGNIZED event type=%r keys=%s preview=%s",
                event_type,
                list(event.keys())[:15],
                json.dumps(event, ensure_ascii=False, default=str)[:500],
            )

        return emitted

    @classmethod
    def _consume_event_codex(cls, state: StreamState, event: dict[str, Any]) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        result = state.result
        event_type = event.get("type")

        if event_type == "thread.started":
            session_id = event.get("thread_id")
            if session_id:
                result.session_id = session_id
                emitted.append({"type": "session", "session_id": session_id})

        elif event_type == "item.started":
            item = event.get("item", {})
            emitted.extend(cls._consume_codex_item(state, item, completed=False))

        elif event_type == "item.completed":
            item = event.get("item", {})
            emitted.extend(cls._consume_codex_item(state, item, completed=True))

        elif event_type == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                result.usage = UsageSummary(
                    input_tokens=cls._coerce_int(usage.get("input_tokens")),
                    output_tokens=cls._coerce_int(usage.get("output_tokens")),
                    cache_read_input_tokens=cls._coerce_int(usage.get("cached_input_tokens")),
                    total_tokens=cls._coerce_int(usage.get("total_tokens")),
                )
                if result.usage and not result.usage.total_tokens:
                    result.usage.total_tokens = sum(
                        value or 0 for value in (
                            result.usage.input_tokens,
                            result.usage.output_tokens,
                            result.usage.cache_read_input_tokens,
                        )
                    )
            result.stop_reason = result.stop_reason or "end_turn"

        elif event_type == "turn.failed":
            result.stop_reason = "error"
            message = event.get("message") or event.get("error") or "Codex turn failed."
            result.result_text = str(message).strip()
            if result.result_text:
                cls._emit_stream_event(emitted, "result", text=result.result_text)

        elif event_type is not None:
            log.warning(
                "[BridgeEvent] UNRECOGNIZED codex event type=%r keys=%s preview=%s",
                event_type,
                list(event.keys())[:15],
                json.dumps(event, ensure_ascii=False, default=str)[:500],
            )

        return emitted

    @classmethod
    def _consume_codex_item(
        cls,
        state: StreamState,
        item: dict[str, Any],
        completed: bool,
    ) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        result = state.result
        item_type = item.get("type")
        item_id = item.get("id", "")

        if item_type == "agent_message":
            text = str(item.get("text") or "").strip()
            if text:
                result.assistant_texts.append(text)
                cls._emit_stream_event(emitted, "text", text=text)
            return emitted

        if item_type == "command_execution":
            command = str(item.get("command") or "").strip()
            if completed:
                output = cls._stringify_tool_result_content(item.get("aggregated_output", ""))
                tool_name = ""
                if item_id in state.pending_tools:
                    state.pending_tools[item_id].output_text = output
                    tool_name = state.pending_tools[item_id].name
                elif command:
                    tc = ToolCall(name="command_execution", input_text=command, output_text=output)
                    result.tool_calls.append(tc)
                    if item_id:
                        state.pending_tools[item_id] = tc
                    tool_name = tc.name
                cls._emit_stream_event(emitted, "tool_result", text=output, tool_name=tool_name, tool_use_id=item_id)
                return emitted

            tc = ToolCall(name="command_execution", input_text=command)
            result.tool_calls.append(tc)
            if item_id:
                state.pending_tools[item_id] = tc
            cls._emit_stream_event(emitted, "tool_use", text=command, tool_name=tc.name, tool_use_id=item_id)
            return emitted

        return emitted

    @staticmethod
    def _parse_stream(raw: str, provider: str = "claude") -> StreamResult:
        """Parse stream-json output from Claude Code CLI."""
        state = StreamState()
        for line in raw.splitlines():
            ClaudeCodeBridge._consume_stream_line(state, line, provider=provider)
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

    def get_session(self, topic_id: str) -> str | None:
        return self._sessions.get(topic_id)

    async def reset_session(self, topic_id: str) -> str | None:
        if not topic_id:
            return None
        async with self._session_lock:
            session_id = self._sessions.pop(topic_id, None)
            await asyncio.to_thread(self._save_sessions)
        if session_id:
            log.info("[Bridge] session reset: topic=%s old_session=%s", topic_id, session_id)
        return session_id

    def _save_sessions(self) -> None:
        directory = os.path.dirname(self.session_store_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = f"{self.session_store_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._sessions, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, self.session_store_path)

    def _build_cmd(self, prompt: str, session_id: str | None) -> list[str]:
        if self.provider == "codex":
            return self._build_cmd_codex(prompt, session_id)
        return self._build_cmd_claude(prompt, session_id)

    def _build_cmd_claude(self, prompt: str, session_id: str | None) -> list[str]:
        safe = lambda s: "'" + s.replace("\n", "\\n").replace("\r", "").replace("'", "'\"'\"'") + "'"
        parts = [
            f"-p {safe(prompt)}",
            "--dangerously-skip-permissions",
            "--output-format stream-json",
            "--verbose",
        ]
        if session_id:
            parts.insert(0, f"--resume {session_id}")
        if self.allowed_tools:
            parts.append(f"--allowedTools {self.allowed_tools}")
        return [self.ttadk_cmd, "code", "-t", self.target, "-m", self.model, "-a", " ".join(parts)]

    def _build_cmd_codex(self, prompt: str, session_id: str | None) -> list[str]:
        safe = lambda s: "'" + s.replace("\n", "\\n").replace("\r", "").replace("'", "'\"'\"'") + "'"
        if session_id:
            parts = ["exec", "--yolo", "--json", "resume", session_id, safe(prompt)]
        else:
            parts = ["exec", "--yolo", "--json", safe(prompt)]
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
