"""Feishu → Claude Code Gateway

Minimal relay: Feishu messages in, ttadk Claude Code out.
No LangGraph, no multi-agent, no frontend — just a message bridge.

Architecture:
    Feishu WebSocket (lark-oapi)
        → FeishuBot._on_message()
            → ClaudeCodeBridge.ask(topic_id, text)
                → ttadk code -t claude (subprocess)
            ← result text
        ← update Feishu card with result

Session continuity:
    topic_id (root_id or msg_id) → ttadk session_id via --resume
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from typing import Any

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
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")  # Strip surrounding quotes
                os.environ.setdefault(key, val)  # Existing env vars take priority
    except FileNotFoundError:
        pass


def load_config(path: str = "config.yaml") -> dict:
    """Load .env then YAML config with $ENV_VAR substitution."""
    load_dotenv()
    with open(path) as f:
        text = f.read()
    for match in set(re.findall(r"\$([A-Z_][A-Z0-9_]*)", text)):
        text = text.replace(f"${match}", os.environ.get(match, ""))
    return yaml.safe_load(text)


# ===========================================================================
# Claude Code Bridge — ttadk CLI wrapper
# ===========================================================================

class ClaudeCodeBridge:
    """Calls Claude Code via ttadk subprocess. Tracks sessions per topic."""

    def __init__(self, cfg: dict):
        self.ttadk_cmd: str = cfg.get("ttadk_cmd", "ttadk")
        self.model: str = cfg.get("model", "gpt-5.4")
        self.target: str = cfg.get("target", "claude")
        self.timeout: int = cfg.get("timeout", 600)
        self.allowed_tools: str = cfg.get("allowed_tools", "Bash,Read,Write,Edit")
        self.system_prompt: str = cfg.get(
            "system_prompt",
            "You are a helpful AI assistant. Always respond in the user's language. "
            "After using tools, summarize findings in text.",
        )
        self._sessions: dict[str, str] = {}  # topic_id → session_id

    async def ask(self, topic_id: str, prompt: str) -> str:
        """Send prompt to Claude Code, return text response."""
        session_id = self._sessions.get(topic_id)
        cmd = self._build_cmd(prompt, session_id)

        log.info("[Bridge] topic=%s session=%s prompt=%d chars", topic_id, session_id, len(prompt))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,  # Critical: prevent stdin corruption
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"⏰ Claude Code timed out after {self.timeout}s"
        except FileNotFoundError:
            return f"❌ Command not found: {self.ttadk_cmd}. Is ttadk installed and in PATH?"

        raw = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        log.info("[Bridge] exit=%d output=%d chars", proc.returncode, len(raw))

        # Auth redirect detection
        for pattern in (r"https?://\S*auth\S*", r"https?://\S*login\S*", r"https?://\S*oauth\S*"):
            urls = re.findall(pattern, raw)
            if urls:
                return f"🔐 需要认证，请在浏览器打开: {urls[0]}"

        # Parse JSON response
        parsed = self._extract_json(raw)
        if parsed is None:
            tail = "\n".join(raw.strip().splitlines()[-20:])
            return tail or "(empty response)"

        # Update session for multi-turn continuity
        for key in ("session_id", "conversation_id", "uuid"):
            sid = parsed.get(key)
            if sid:
                self._sessions[topic_id] = sid
                log.info("[Bridge] session: topic=%s → %s", topic_id, sid)
                break

        return parsed.get("result", "") or "(empty result)"

    def _build_cmd(self, prompt: str, session_id: str | None) -> list[str]:
        """Build ttadk CLI command list."""
        # Escape for shell single-quote context
        safe = lambda s: "'" + s.replace("\n", "\\n").replace("\r", "").replace("'", "'\"'\"'") + "'"

        parts = [
            f"-p {safe(prompt)}",
            f"--system-prompt {safe(self.system_prompt)}",
            "--output-format json",
        ]
        if session_id:
            parts.insert(0, f"--resume {session_id}")
        if self.allowed_tools:
            parts.append(f"--allowedTools {self.allowed_tools}")

        return [self.ttadk_cmd, "code", "-t", self.target, "-m", self.model, "-a", " ".join(parts)]

    @staticmethod
    def _extract_json(output: str) -> dict | None:
        """Extract last valid JSON object from mixed CLI output."""
        for line in reversed(output.splitlines()):
            stripped = line.strip()
            if stripped.startswith("{"):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    continue
        # Fallback: regex scan
        for m in reversed(re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", output)):
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue
        return None


# ===========================================================================
# Feishu Bot — lark-oapi WebSocket (no public IP needed)
# ===========================================================================

class FeishuBot:
    """Receives Feishu messages via WebSocket, forwards to ClaudeCodeBridge."""

    def __init__(self, cfg: dict, bridge: ClaudeCodeBridge):
        self.app_id: str = cfg["app_id"]
        self.app_secret: str = cfg["app_secret"]
        self.bridge = bridge
        self._api_client: Any = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._ws_thread: threading.Thread | None = None
        # Feishu SDK message classes (lazy-loaded)
        self._sdk: dict[str, Any] = {}

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
        """Run lark WS client in a thread with its own event loop.

        lark-oapi caches a module-level event loop and calls run_until_complete(),
        which conflicts with the main thread's running loop. We patch it.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            import lark_oapi as lark
            import lark_oapi.ws.client as ws_mod

            ws_mod.loop = loop  # Patch SDK's module-level loop

            handler = (
                lark.EventDispatcherHandler.builder("", "")
                .register_p2_im_message_receive_v1(self._on_message)
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

    # --- Message handler ---

    def _on_message(self, event: Any) -> None:
        """Handle incoming message (runs in lark thread)."""
        try:
            msg = event.event.message
            chat_id = msg.chat_id
            msg_id = msg.message_id
            root_id = getattr(msg, "root_id", None) or None
            topic_id = root_id or msg_id  # Thread replies share the same topic

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
        """Full message lifecycle: react → thinking card → Claude Code → update card → done."""
        # 1. OK reaction (fire-and-forget)
        asyncio.create_task(self._react(msg_id, "OK"))

        # 2. Reply with "thinking..." card in thread
        card_id = await self._reply_card(msg_id, "🤔 Thinking...")

        # 3. Call Claude Code
        try:
            result = await self.bridge.ask(topic_id, text)
        except Exception as e:
            log.exception("Bridge error")
            result = f"❌ Error: {e}"

        # 4. Truncate if too long for Feishu card (100KB limit, ~50K chars safe)
        if len(result) > 50000:
            result = result[:50000] + "\n\n... (truncated, response too long)"

        # 5. Update card with result
        if card_id:
            await self._update_card(card_id, result)
        else:
            await self._reply_card(msg_id, result)

        # 6. DONE reaction
        await self._react(msg_id, "DONE")

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
        """Reply with interactive card in thread, return card message_id."""
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
        """Build Feishu interactive card JSON (renders markdown natively)."""
        return json.dumps({
            "config": {"wide_screen_mode": True, "update_multi": True},
            "elements": [{"tag": "markdown", "content": text}],
        })

    @staticmethod
    def _extract_text(content: dict) -> str:
        """Extract plain text from Feishu message content (text + rich text)."""
        # Plain text message
        if "text" in content:
            return content["text"]
        # Rich text (post) message
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    cfg = load_config()
    bridge = ClaudeCodeBridge(cfg.get("claude", {}))
    bot = FeishuBot(cfg["feishu"], bridge)

    await bot.start()

    log.info("🚀 Gateway running. Ctrl+C to stop.")
    try:
        await asyncio.Event().wait()  # Block forever
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
