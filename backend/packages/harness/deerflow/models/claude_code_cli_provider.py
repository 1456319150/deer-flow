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

Config example (config.yaml):
    - name: claude-code
      display_name: Claude Code (via ttadk)
      use: deerflow.models.claude_code_cli_provider:ClaudeCodeCliModel
      model: gpt-5.4
      target: claude
      timeout: 600
      allowed_tools: "Bash,Read,Write,Edit"
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


class ClaudeCodeCliModel(BaseChatModel):
    """LangChain chat model that delegates to Claude Code via ttadk CLI.

    Claude Code handles its own agent loop + tool execution internally.
    DeerFlow acts as orchestration/UI layer only.
    """

    model: str = "gpt-5.4"
    target: str = "claude"
    timeout: int = 600
    allowed_tools: str | None = None
    ttadk_cmd: str = "ttadk"
    retry_max_attempts: int = 3
    _session_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "claude-code-cli"

    # ─── Message conversion ───────────────────────────────────────────

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
        """Convert LangChain message list into a single prompt string.

        Simple approach: Claude Code's -p flag takes a plain text prompt.
        For single messages, pass content directly. For multi-turn, use
        minimal role prefixes separated by newlines.
        """
        system_parts: list[str] = []
        conversation_parts: list[str] = []

        for msg in messages:
            content = self._normalize_content(msg.content)
            if not content:
                continue

            if isinstance(msg, SystemMessage):
                system_parts.append(content)
            elif isinstance(msg, HumanMessage):
                conversation_parts.append(content)
            elif isinstance(msg, AIMessage):
                if content:
                    conversation_parts.append(f"Assistant: {content}")
            elif isinstance(msg, ToolMessage):
                conversation_parts.append(f"Tool result: {content}")

        parts = []
        if system_parts:
            parts.extend(system_parts)
        if conversation_parts:
            # If only one user message and no system, return it directly
            if len(parts) == 0 and len(conversation_parts) == 1 and not isinstance(messages[-1], (AIMessage, ToolMessage)):
                return conversation_parts[0]
            parts.extend(conversation_parts)

        return "\n\n".join(parts)


    # ─── JSON extraction (from claude_chain.py) ──────────────────────

    @staticmethod
    def _extract_json(output: str) -> dict | None:
        """Extract the last valid JSON object from mixed CLI output."""
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

    # ─── CLI execution ───────────────────────────────────────────────

    def _build_cli_args(self, prompt: str) -> list[str]:
        """Build the ttadk CLI command as a list of arguments."""
        # Escape single quotes for shell-like arg parsing (ttadk parses -a value)
        # Collapse newlines to literal \n to avoid ttadk -a arg parsing breakage
        clean_prompt = prompt.replace("\r\n", "\\n").replace("\n", "\\n")
        # Escape single quotes for shell-like arg parsing in ttadk
        safe_prompt = "'" + clean_prompt.replace("'", "'\"'\"'") + "'"
        a_parts = [f"-p {safe_prompt}", "--output-format json"]

        if self._session_id:
            a_parts.insert(0, f"--resume {self._session_id}")

        if self.allowed_tools:
            a_parts.append(f"--allowedTools {self.allowed_tools}")

        a_arg = " ".join(a_parts)
        return [self.ttadk_cmd, "code", "-t", self.target, "-m", self.model, "-a", a_arg]

    def _call_cli_sync(self, prompt: str) -> dict:
        """Execute ttadk code synchronously and return parsed JSON."""
        cmd = self._build_cli_args(prompt)
        logger.info(f"ClaudeCodeCli: calling ttadk (model={self.model}, target={self.target})")

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
            raise RuntimeError(f"Claude Code CLI timed out after {self.timeout}s")
        except FileNotFoundError:
            raise RuntimeError(
                f"Command not found: {self.ttadk_cmd}. "
                "Ensure ttadk is installed and in PATH."
            )

        raw_output = stdout or ""

        # Detect auth URLs
        auth_patterns = [
            r"https?://\S*auth\S*",
            r"https?://\S*login\S*",
            r"https?://\S*oauth\S*",
        ]
        for pattern in auth_patterns:
            urls = re.findall(pattern, raw_output)
            if urls:
                raise RuntimeError(
                    f"Claude Code requires authentication. "
                    f"Please open in browser: {urls[0]}"
                )

        parsed = self._extract_json(raw_output)
        if parsed is None:
            logger.warning("Failed to parse JSON from CLI output, using raw text")
            tail = "\n".join(raw_output.strip().splitlines()[-20:])
            return {"result": tail, "session_id": None, "usage": {}}

        return parsed

    async def _call_cli_async(self, prompt: str) -> dict:
        """Execute ttadk code asynchronously."""
        cmd = self._build_cli_args(prompt)
        logger.info(f"ClaudeCodeCli: async calling ttadk (model={self.model}, target={self.target})")

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
            raise RuntimeError(f"Claude Code CLI timed out after {self.timeout}s")
        except FileNotFoundError:
            raise RuntimeError(
                f"Command not found: {self.ttadk_cmd}. "
                "Ensure ttadk is installed and in PATH."
            )

        raw_output = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""

        auth_patterns = [
            r"https?://\S*auth\S*",
            r"https?://\S*login\S*",
            r"https?://\S*oauth\S*",
        ]
        for pattern in auth_patterns:
            urls = re.findall(pattern, raw_output)
            if urls:
                raise RuntimeError(
                    f"Claude Code requires authentication. "
                    f"Please open in browser: {urls[0]}"
                )

        parsed = self._extract_json(raw_output)
        if parsed is None:
            logger.warning("Failed to parse JSON from CLI output, using raw text")
            tail = "\n".join(raw_output.strip().splitlines()[-20:])
            return {"result": tail, "session_id": None, "usage": {}}

        return parsed

    # ─── Response parsing ────────────────────────────────────────────

    def _parse_response(self, response: dict) -> ChatResult:
        """Convert ttadk JSON response to LangChain ChatResult."""
        # Extract session_id for continuity
        for key in ("session_id", "conversation_id", "uuid"):
            sid = response.get(key)
            if sid:
                self._session_id = sid
                break

        result_text = response.get("result", "")
        usage = response.get("usage", {})

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

    # ─── BaseChatModel interface ────────────────────────────────────

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
