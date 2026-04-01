"""MiraBridge — Mira Agent API adapter for DeerFlow gateway.

Same interface as ClaudeCodeBridge: stream_ask() yields the same
event format, so FeishuBot/WeixinBot work without any changes.

Key design decisions:
    - Mira manages conversation context server-side, so we just reuse
      sessionId per topic_id (no need to replay message history).
    - Reason events are classified by data.type:
        * stream_event — actual Claude streaming deltas (process these)
        * assistant — full accumulated snapshot (skip to avoid duplicates)
        * user — tool result injected back (internal, not surfaced)
        * system — metadata (skip)
    - Content event maps to a single "result" StreamEvent (fallback).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, AsyncIterator

from gateway import StreamResult, StreamEvent, UsageSummary
from mira_client import MiraClient, MiraAuthError

log = logging.getLogger("mira_bridge")


class MiraBridge:
    """Wraps MiraClient to match ClaudeCodeBridge's stream_ask interface.

    Interface contract (duck-typed, same as ClaudeCodeBridge):
        stream_ask(topic_id, prompt) → AsyncIterator[dict]
        ask(topic_id, prompt) → StreamResult
        get_session(topic_id) → str | None
        reset_session(topic_id) → str | None
    """

    def __init__(self, cfg: dict):
        token = cfg.get("session_token") or os.environ.get("MIRA_SESSION", "")
        if not token:
            raise ValueError(
                "Mira session token required. Set MIRA_SESSION env var "
                "or mira.session_token in config.yaml"
            )
        base_url = cfg.get("base_url", "https://mira.byteintl.net")
        self.model: str = cfg.get("model", "re-o-46")
        self.mode: str = cfg.get("mode", "quick")
        self.timeout: int = cfg.get("timeout", 300)
        self.session_store_path: str = cfg.get(
            "session_store_path", ".mira-sessions.json"
        )
        self.client = MiraClient(
            token, base_url=base_url, timeout=float(self.timeout)
        )
        self._session_lock = asyncio.Lock()
        self._sessions: dict[str, str] = self._load_sessions()

    # ── Primary interface (matches ClaudeCodeBridge) ───────────────

    async def ask(self, topic_id: str, prompt: str) -> StreamResult:
        """Send prompt to Mira, return structured StreamResult."""
        result: StreamResult | None = None
        async for event in self.stream_ask(topic_id, prompt):
            if event["type"] == "final":
                result = event["result"]
        return result or StreamResult()

    async def stream_ask(
        self, topic_id: str, prompt: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream Mira output, yielding gateway-compatible events.

        Event format (same as ClaudeCodeBridge):
            {"type": "stream_event", "event": StreamEvent(...)}
            {"type": "final", "result": StreamResult(...)}
        """
        # Get or create Mira session for this topic
        session_id = self._sessions.get(topic_id)
        if not session_id:
            try:
                session_id = await self.client.create_session(model=self.model)
                await self._remember_session(topic_id, session_id)
                log.info(
                    "[MiraBridge] new session %s for topic %s",
                    session_id, topic_id,
                )
            except MiraAuthError as e:
                yield {
                    "type": "final",
                    "result": StreamResult(
                        assistant_texts=[
                            f"\U0001f511 Mira token expired: {e}\n"
                            f"Please update MIRA_SESSION in .env"
                        ]
                    ),
                }
                return
            except Exception as e:
                yield {
                    "type": "final",
                    "result": StreamResult(
                        assistant_texts=[f"\u274c Mira session creation failed: {e}"]
                    ),
                }
                return

        log.info(
            "[MiraBridge] topic=%s session=%s prompt=%d chars",
            topic_id, session_id, len(prompt),
        )

        result = StreamResult()
        result.session_id = session_id

        # ── Streaming state ────────────────────────────────────────
        # Track current content block type from content_block_start
        current_block: str = ""           # "thinking" | "text" | "tool_use" | ""
        thinking_chunks: list[str] = []   # accumulated thinking text
        result_chunks: list[str] = []     # accumulated answer text

        try:
            async for evt in self.client.chat(
                session_id, prompt, model=self.model, mode=self.mode
            ):
                if evt.event != "reason":
                    # ── content event: extract usage (text already streamed) ─
                    if evt.event == "content":
                        inner = evt.data.get("content", {}) if isinstance(evt.data, dict) else {}
                        if isinstance(inner, dict):
                            result.usage = self._extract_usage(inner)
                            # If we got no stream deltas, fall back to content.result
                            if not result_chunks and inner.get("result"):
                                result.result_text = inner["result"]
                                yield {
                                    "type": "stream_event",
                                    "event": StreamEvent(kind="result", text=result.result_text),
                                }
                    elif evt.event == "title" and evt.text:
                        log.info("[MiraBridge] session title: %s", evt.text)
                    continue

                # ── reason event: classify by data_type ────────────

                # Skip assistant summaries (full accumulated text, NOT a delta)
                # and system metadata — these cause duplicate output if processed
                if evt.data_type in ("assistant", "system", "safety_audit"):
                    log.debug(
                        "[MiraBridge] skip %s reason (not a delta)",
                        evt.data_type,
                    )
                    continue

                # Skip user events (internal tool results — not for display)
                if evt.data_type == "user":
                    log.debug("[MiraBridge] tool result (internal): %s",
                              evt.text[:100] if evt.text else "")
                    continue

                # ── stream_event or untyped reason: streaming deltas ─

                # Block type transitions (content_block_start)
                if evt.inner_type == "content_block_start" and evt.block_type:
                    current_block = evt.block_type
                    log.debug("[MiraBridge] block start: %s", current_block)

                    # Log tool_use blocks silently (internal Mira tool calls)
                    if evt.block_type == "tool_use":
                        log.info("[MiraBridge] tool call: %s",
                                 evt.tool_name or "unknown")
                    continue

                if evt.inner_type == "content_block_stop":
                    current_block = ""
                    continue

                # Skip input_json_delta (tool input JSON streaming, not display text)
                if evt.delta_type == "input_json_delta":
                    continue

                # Only process events with text deltas
                if not evt.text:
                    continue

                # Determine kind based on current block or delta type
                if current_block == "thinking" or evt.delta_type == "thinking_delta":
                    thinking_chunks.append(evt.text)
                    yield {
                        "type": "stream_event",
                        "event": StreamEvent(kind="thinking", text=evt.text),
                    }
                else:
                    # text block or unclassified → treat as answer
                    result_chunks.append(evt.text)
                    yield {
                        "type": "stream_event",
                        "event": StreamEvent(kind="result", text=evt.text),
                    }

        except MiraAuthError as e:
            result.assistant_texts.append(
                f"\U0001f511 Mira token expired: {e}"
            )
        except Exception as e:
            log.exception("[MiraBridge] stream error")
            result.assistant_texts.append(f"\u274c Mira error: {e}")

        # Build final result from accumulated chunks
        if thinking_chunks:
            result.thinking = ["".join(thinking_chunks)]
        if result_chunks:
            result.result_text = "".join(result_chunks)

        result.stop_reason = "end_turn"
        yield {"type": "final", "result": result}

    # ── Session management (matches ClaudeCodeBridge) ──────────────

    def get_session(self, topic_id: str) -> str | None:
        return self._sessions.get(topic_id)

    async def reset_session(self, topic_id: str) -> str | None:
        if not topic_id:
            return None
        async with self._session_lock:
            session_id = self._sessions.pop(topic_id, None)
            await asyncio.to_thread(self._save_sessions)
        if session_id:
            log.info(
                "[MiraBridge] session reset: topic=%s old=%s",
                topic_id, session_id,
            )
            try:
                await self.client.delete_session(session_id)
            except Exception:
                log.debug(
                    "Failed to delete Mira session %s",
                    session_id, exc_info=True,
                )
        return session_id

    # ── Usage extraction ───────────────────────────────────────────

    @staticmethod
    def _extract_usage(data: dict) -> UsageSummary | None:
        """Extract token usage from Mira content event data."""
        input_tokens = data.get("input_tokens")
        output_tokens = data.get("output_tokens")
        cost = data.get("cost") or data.get("total_cost_usd")

        if input_tokens is None and output_tokens is None:
            return None

        total = None
        if input_tokens is not None or output_tokens is not None:
            total = (input_tokens or 0) + (output_tokens or 0)

        cost_usd = None
        if cost is not None:
            try:
                cost_str = str(cost).replace("$", "").replace("USD", "").strip()
                cost_usd = float(cost_str)
            except (ValueError, TypeError):
                pass

        return UsageSummary(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            cost_usd=cost_usd,
        )

    # ── Session persistence ────────────────────────────────────────

    def _load_sessions(self) -> dict[str, str]:
        try:
            with open(self.session_store_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError):
            log.warning(
                "[MiraBridge] failed to load sessions: %s",
                self.session_store_path,
            )
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if k and v}

    async def _remember_session(
        self, topic_id: str, session_id: str
    ) -> None:
        if (
            not topic_id
            or not session_id
            or self._sessions.get(topic_id) == session_id
        ):
            return
        async with self._session_lock:
            self._sessions[topic_id] = session_id
            await asyncio.to_thread(self._save_sessions)

    def _save_sessions(self) -> None:
        directory = os.path.dirname(self.session_store_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = f"{self.session_store_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                self._sessions, f, ensure_ascii=False, indent=2, sort_keys=True
            )
        os.replace(tmp, self.session_store_path)
