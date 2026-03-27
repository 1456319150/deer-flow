"""Claude Code CLI provider — wraps ttadk code as a LangChain BaseChatModel.

Architecture:
  DeerFlow (UI + session + memory) → ClaudeCodeCliModel → ttadk CLI → Claude Code agent loop

Key design:
  - Claude Code runs its OWN internal agent loop with built-in tools
    (Bash, Read, Write, Edit, WebSearch, etc.)
  - DeerFlow's tool-calling loop effectively runs 1 iteration: send task → get final answer
  - bind_tools() is a no-op — DeerFlow's tools are NOT forwarded to Claude Code
  - --allowedTools controls which Claude Code built-in tools are available
  - Session continuity via --resume (optional, keyed by DeerFlow thread_id)
  - DeerFlow system prompt is SKIPPED — Claude Code gets a minimal system prompt
    to ensure it always produces a text response after tool use.
  - Multi-turn: when session_id exists, only the latest user message is sent.
  - Uses --output-format stream-json to work around Claude Code v2.1.83 bug
    where --output-format json returns empty result field.

Config example (config.yaml):
    - name: claude-code
      display_name: Claude Code (via ttadk)
      use: deerflow.models.claude_code_cli_provider:ClaudeCodeCliModel
      model: gpt-5.4
      target: claude
      timeout: 600
      allowed_tools: "Bash,Read,Write,Edit"
      system_prompt: "Always reply in the user's language. After using tools, summarize findings in text."
"""

import asyncio
import json
import logging
import re
import subprocess
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)

# Minimal system prompt for Claude Code.
# DeerFlow's 18KB system prompt is skipped (wrong tools/paths/instructions).
# This just ensures Claude Code always produces a text response.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI coding assistant. "
    "Always respond with a clear text answer in the user's language. "
    "When you use tools to gather information, summarize your findings in your final response. "
    "Never end with just a tool call — always provide a text conclusion."
)


class ClaudeCodeCliModel(BaseChatModel):
    """LangChain chat model that delegates to Claude Code via ttadk CLI.

    Claude Code handles its own agent loop + tool execution internally.
    DeerFlow acts as orchestration/UI layer only.
    """

    model: str = "gpt-5.4"
    target: str = "claude"
    timeout: int = 600
    allowed_tools: str | None = None
    system_prompt: str | None = None
    ttadk_cmd: str = "ttadk"
    retry_max_attempts: int = 3
    _session_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "claude-code-cli"

    @property
    def _effective_system_prompt(self) -> str:
        """Return the effective system prompt (config override or default)."""
        return self.system_prompt or DEFAULT_SYSTEM_PROMPT

    # ─── Message conversion ────────────────────────────────────────────

    @classmethod
    def _normalize_content(cls, content: Any) -> str:
        """Flatten LangChain content blocks to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [cls._normalize_content(item) for item in content]
            return "\n".join(p for p in parts if p)
        if isinstance(content, dict):
            for key in ("text", "output"):
                val = content.get(key)
                if isinstance(val, str):
                    return val
            nested = content.get("content")
            if nested is not None:
                return cls._normalize_content(nested)
            try:
                return json.dumps(content, ensure_ascii=False)
            except TypeError:
                return str(content)
        return str(content) if content else ""

    def _flatten_messages(self, messages: list[BaseMessage]) -> str:
        """Extract only the latest user message.

        DeerFlow's system prompt is always skipped — we use our own minimal
        system prompt via --system-prompt flag instead.

        Always extracts only the last HumanMessage because:
        1. Each ttadk call may create a new model instance (session_id lost)
        2. AI response content in history causes CLI arg escaping issues
        3. If --resume works, Claude Code already has context
        """
        # Log incoming messages for debugging
        msg_summary = []
        for msg in messages:
            content_preview = self._normalize_content(msg.content)[:80]
            msg_summary.append(f"{type(msg).__name__}({len(self._normalize_content(msg.content))}chars): {content_preview!r}")
        logger.warning(
            "[CCM] _flatten_messages called with %d messages (session_id=%s):\n  %s",
            len(messages), self._session_id, "\n  ".join(msg_summary)
        )

        # Always extract only the last HumanMessage
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = self._normalize_content(msg.content)
                if content:
                    logger.warning("[CCM] Extracted last HumanMessage (%d chars): %r", len(content), content[:200])
                    return content

        logger.warning("[CCM] No HumanMessage found, returning empty")
        return ""

    # ─── Stream-JSON parsing ──────────────────────────────────────────

    @staticmethod
    def _parse_stream_json(raw_output: str) -> dict:
        """Parse stream-json output from Claude Code CLI.

        stream-json format emits one JSON object per line. We look for:
        1. "assistant" messages with content[].text → aggregate as result text
        2. "result" event → extract metadata (session_id, usage, cost)

        This works around the v2.1.83 bug where --output-format json
        returns empty result field despite generating output tokens.

        Returns a dict compatible with the old json format:
            {"result": "...", "session_id": "...", "usage": {...}, ...}
        """
        assistant_texts = []
        result_event = {}

        for line in raw_output.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            if event_type == "assistant":
                # Extract text from assistant message content blocks
                message = event.get("message", {})
                content_blocks = message.get("content", [])
                for block in content_blocks:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            assistant_texts.append(text)
                logger.warning(
                    "[CCM] stream-json: assistant event, extracted %d text blocks",
                    len(content_blocks)
                )

            elif event_type == "result":
                result_event = event
                # Also check if result has non-empty text (fixed versions)
                result_text = event.get("result", "")
                if result_text:
                    # If the result field is non-empty, prefer it
                    # (means we're running a fixed Claude Code version)
                    assistant_texts = [result_text]
                logger.warning(
                    "[CCM] stream-json: result event, result=%d chars, session_id=%s",
                    len(result_text), event.get("session_id", "N/A")
                )

        # Build the combined response
        combined_text = "\n\n".join(assistant_texts) if assistant_texts else ""

        return {
            "result": combined_text,
            "session_id": result_event.get("session_id"),
            "conversation_id": result_event.get("conversation_id"),
            "uuid": result_event.get("uuid"),
            "usage": result_event.get("usage", {}),
            "total_cost_usd": result_event.get("total_cost_usd"),
            "stop_reason": result_event.get("stop_reason"),
            "output_tokens": result_event.get("output_tokens", 0),
        }

    # ─── JSON extraction (legacy fallback) ────────────────────────────

    @staticmethod
    def _extract_json(output: str) -> dict | None:
        """Extract the last valid JSON object from mixed CLI output.

        Used as fallback when stream-json parsing yields no result.
        """
        # Strategy 1: scan lines from bottom
        for line in reversed(output.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        # Strategy 2: regex for nested JSON objects
        matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", output)
        for m in reversed(matches):
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue

        return None

    # ─── CLI execution ────────────────────────────────────────────────

    def _build_cli_args(self, prompt: str) -> list[str]:
        """Build the ttadk CLI command as a list of arguments."""
        clean_prompt = prompt.replace("\r\n", "\\n").replace("\n", "\\n")
        safe_prompt = "'" + clean_prompt.replace("'", "'\"'\"'") + "'"

        # System prompt — short enough to safely pass via CLI
        sys_prompt = self._effective_system_prompt.replace("\r\n", "\\n").replace("\n", "\\n")
        safe_sys = "'" + sys_prompt.replace("'", "'\"'\"'") + "'"

        # Use stream-json instead of json to work around Claude Code v2.1.83 bug
        # where --output-format json returns empty result field.
        # stream-json emits per-event JSON lines, including assistant messages
        # with the actual text content.
        a_parts = [f"-p {safe_prompt}", f"--system-prompt {safe_sys}", "--output-format stream-json"]

        if self._session_id:
            a_parts.insert(0, f"--resume {self._session_id}")

        if self.allowed_tools:
            a_parts.append(f"--allowedTools {self.allowed_tools}")

        a_arg = " ".join(a_parts)
        cmd = [self.ttadk_cmd, "code", "-t", self.target, "-m", self.model, "-a", a_arg]

        # Log the full command for debugging
        prompt_display = prompt[:300] + "..." if len(prompt) > 300 else prompt
        logger.warning(
            "[CCM] CLI command built:\n"
            "  ttadk_cmd: %s\n"
            "  target: %s, model: %s\n"
            "  session_id: %s\n"
            "  output_format: stream-json\n"
            "  system_prompt (%d chars): %r\n"
            "  prompt (%d chars): %r\n"
            "  -a arg (%d chars): %r",
            self.ttadk_cmd, self.target, self.model,
            self._session_id,
            len(self._effective_system_prompt), self._effective_system_prompt[:200],
            len(prompt), prompt_display,
            len(a_arg), a_arg[:500] + ("..." if len(a_arg) > 500 else "")
        )

        return cmd

    def _process_raw_output(self, raw_output: str, exit_code: int) -> dict:
        """Process raw CLI output: try stream-json first, fall back to legacy JSON.

        Returns parsed response dict with at minimum: result, session_id, usage.
        """
        # Check for auth redirects first
        auth_patterns = [
            r"https?://\S*auth\S*",
            r"https?://\S*login\S*",
            r"https?://\S*oauth\S*",
        ]
        for pattern in auth_patterns:
            urls = re.findall(pattern, raw_output)
            if urls:
                logger.error("[CCM] AUTH REQUIRED: %s", urls[0])
                raise RuntimeError(
                    f"Claude Code requires authentication. "
                    f"Please open in browser: {urls[0]}"
                )

        # Primary: parse as stream-json
        parsed = self._parse_stream_json(raw_output)
        if parsed["result"]:
            logger.warning(
                "[CCM] stream-json parsed OK: result=%d chars, session_id=%s",
                len(parsed["result"]), parsed.get("session_id")
            )
            return parsed

        # Fallback 1: try legacy single-JSON extraction
        logger.warning("[CCM] stream-json yielded empty result, trying legacy JSON extraction")
        legacy = self._extract_json(raw_output)
        if legacy and legacy.get("result"):
            logger.warning(
                "[CCM] Legacy JSON OK: result=%d chars, session_id=%s",
                len(legacy["result"]), legacy.get("session_id")
            )
            return legacy

        # Fallback 2: use raw tail text as last resort
        logger.warning("[CCM] All JSON parsing failed, using raw tail text")
        tail_lines = raw_output.strip().splitlines()[-20:]
        # Filter out JSON lines (they're events, not user-facing text)
        text_lines = [l for l in tail_lines if not l.strip().startswith("{")]
        tail = "\n".join(text_lines).strip()
        if not tail:
            # If everything was JSON, just use whatever we have
            tail = "\n".join(tail_lines).strip()

        return {"result": tail or "(Claude Code returned no text)", "session_id": None, "usage": {}}

    def _call_cli_sync(self, prompt: str) -> dict:
        """Execute ttadk code synchronously and return parsed response."""
        cmd = self._build_cli_args(prompt)

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            stdout, _ = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            logger.error("[CCM] TIMEOUT after %ds", self.timeout)
            raise RuntimeError(f"Claude Code CLI timed out after {self.timeout}s")
        except FileNotFoundError:
            logger.error("[CCM] COMMAND NOT FOUND: %s", self.ttadk_cmd)
            raise RuntimeError(
                f"Command not found: {self.ttadk_cmd}. "
                "Ensure ttadk is installed and in PATH."
            )

        raw_output = stdout or ""
        exit_code = proc.returncode

        logger.warning(
            "[CCM] CLI finished (exit_code=%d, output=%d chars):\n"
            "--- RAW OUTPUT START ---\n%s\n--- RAW OUTPUT END ---",
            exit_code, len(raw_output),
            raw_output[:3000] + ("...[truncated]" if len(raw_output) > 3000 else "")
        )

        return self._process_raw_output(raw_output, exit_code)

    async def _call_cli_async(self, prompt: str) -> dict:
        """Execute ttadk code asynchronously and return parsed response."""
        cmd = self._build_cli_args(prompt)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.error("[CCM] ASYNC TIMEOUT after %ds", self.timeout)
            raise RuntimeError(f"Claude Code CLI timed out after {self.timeout}s")
        except FileNotFoundError:
            logger.error("[CCM] COMMAND NOT FOUND: %s", self.ttadk_cmd)
            raise RuntimeError(
                f"Command not found: {self.ttadk_cmd}. "
                "Ensure ttadk is installed and in PATH."
            )

        raw_output = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        exit_code = proc.returncode

        logger.warning(
            "[CCM] async CLI finished (exit_code=%d, output=%d chars):\n"
            "--- RAW OUTPUT START ---\n%s\n--- RAW OUTPUT END ---",
            exit_code, len(raw_output),
            raw_output[:3000] + ("...[truncated]" if len(raw_output) > 3000 else "")
        )

        return self._process_raw_output(raw_output, exit_code)

    # ─── Response parsing ─────────────────────────────────────────────

    def _parse_response(self, response: dict) -> ChatResult:
        """Convert parsed response to LangChain ChatResult."""
        old_session_id = self._session_id

        for key in ("session_id", "conversation_id", "uuid"):
            sid = response.get(key)
            if sid:
                self._session_id = sid
                break

        result_text = response.get("result", "")
        usage = response.get("usage", {})

        logger.warning(
            "[CCM] _parse_response: session_id %s -> %s, result_text=%d chars, empty=%s",
            old_session_id, self._session_id,
            len(result_text), result_text == ""
        )

        message = AIMessage(
            content=result_text,
            tool_calls=[],
            additional_kwargs={
                "session_id": self._session_id,
            },
            response_metadata={
                "model": self.model,
                "usage": usage,
                "cost_usd": response.get("total_cost_usd"),
            },
        )

        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "token_usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                "model_name": self.model,
            },
        )

    # ─── BaseChatModel interface ──────────────────────────────────────

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation via Claude Code CLI."""
        prompt = self._flatten_messages(messages)
        response = self._call_cli_sync(prompt)
        return self._parse_response(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation via Claude Code CLI."""
        prompt = self._flatten_messages(messages)
        response = await self._call_cli_async(prompt)
        return self._parse_response(response)

    def bind_tools(self, tools: list, **kwargs: Any) -> Any:
        """No-op: Claude Code uses its own built-in tools.

        DeerFlow's tools are not forwarded. Claude Code's internal agent loop
        handles tool execution via --allowedTools configuration.
        """
        from langchain_core.runnables import RunnableBinding

        return RunnableBinding(bound=self, kwargs={}, **kwargs)
