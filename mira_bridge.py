"""MiraBridge — Mira Agent API adapter for DeerFlow gateway.

Same interface as ClaudeCodeBridge: stream_ask() yields the same
event format, so FeishuBot/WeixinBot work without any changes.

Key design decisions:
    - Mira manages conversation context server-side, so we just reuse
      sessionId per topic_id (no need to replay message history).
    - Reason events are classified by data.type:
        * stream_event — actual Claude streaming deltas (process these)
        * assistant — full accumulated snapshot; extract tool_use_id → name
          mapping (fallback) and emit any tool_use blocks not seen via deltas
        * user — tool results → surfaced as tool_result events
        * system — metadata (skip)
    - Tool call lifecycle:
        1. content_block_start (block_type=tool_use) → record name/id
        2. input_json_delta events → accumulate partial JSON
        3. content_block_stop → yield StreamEvent(kind="tool_use")
        4. assistant message snapshot → fallback: extract tool names and
           emit tool_use events for any tools not captured via deltas
        5. user event with tool_result blocks → yield StreamEvent(kind="tool_result")
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

    async def ask(self, topic_id: str, prompt: str, *, image_paths: list[str] | None = None, file_paths: list[str] | None = None) -> StreamResult:
        """Send prompt to Mira, return structured StreamResult."""
        result: StreamResult | None = None
        async for event in self.stream_ask(topic_id, prompt, image_paths=image_paths, file_paths=file_paths):
            if event["type"] == "final":
                result = event["result"]
        return result or StreamResult()

    async def stream_ask(
        self, topic_id: str, prompt: str, *, image_paths: list[str] | None = None, file_paths: list[str] | None = None
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

        # --- Upload inbound attachments ---
        attachments: list[dict] = []
        if image_paths or file_paths:
            all_paths = list(image_paths or []) + list(file_paths or [])
            for fpath in all_paths:
                try:
                    file_info = await self.client.upload_file(fpath)
                    attachments.append(file_info.to_attachment())
                    log.info("[MiraBridge] uploaded attachment %s -> %s", fpath, file_info.url[:80])
                except Exception:
                    log.warning("[MiraBridge] failed to upload %s", fpath, exc_info=True)

        result = StreamResult()
        result.session_id = session_id

        # ── Streaming state ────────────────────────────────────────
        current_block: str = ""           # "thinking" | "text" | "tool_use" | ""
        thinking_chunks: list[str] = []   # accumulated thinking text
        result_chunks: list[str] = []     # accumulated answer text

        # Tool call tracking: accumulate input_json_delta for current tool_use
        _cur_tool_name: str = ""
        _cur_tool_use_id: str = ""
        _cur_tool_input_chunks: list[str] = []
        # Map tool_use_id → tool_name for correlating tool_result events
        _tool_names: dict[str, str] = {}
        # Track which tool_use_ids we've already emitted via content_block_stop
        _emitted_tool_ids: set[str] = set()

        try:
            async for evt in self.client.chat(
                session_id, prompt, model=self.model, mode=self.mode,
                attachments=attachments or None,
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

                # Skip system metadata and safety audit events
                if evt.data_type in ("system", "safety_audit"):
                    log.debug(
                        "[MiraBridge] skip %s reason (metadata)",
                        evt.data_type,
                    )
                    continue

                # ── assistant snapshots: extract tool_use_id→name mapping ──
                # The frontend uses assistant messages as a fallback source
                # for tool names (recordToolName in parseStreamingEvents).
                # We do the same: extract tool names, and emit tool_use events
                # for any tools not already captured via content_block_stop.
                if evt.data_type == "assistant":
                    for tool_evt in self._extract_assistant_tool_uses(
                        evt, _tool_names, _emitted_tool_ids
                    ):
                        yield tool_evt
                    continue

                # ── user events: tool results ──────────────────────
                if evt.data_type == "user":
                    # Parse tool_result blocks from user message content
                    for tr_evt in self._parse_user_tool_results(evt, _tool_names):
                        yield tr_evt
                    continue

                # ── stream_event or untyped reason: streaming deltas ─

                # Block type transitions (content_block_start)
                if evt.inner_type == "content_block_start" and evt.block_type:
                    current_block = evt.block_type
                    log.debug("[MiraBridge] block start: %s", current_block)

                    if evt.block_type == "tool_use":
                        # Start tracking a new tool call
                        _cur_tool_name = evt.tool_name or ""
                        _cur_tool_use_id = evt.tool_use_id or ""
                        _cur_tool_input_chunks = []
                        if _cur_tool_use_id and _cur_tool_name:
                            _tool_names[_cur_tool_use_id] = _cur_tool_name
                        log.info(
                            "[MiraBridge] tool_use start: name=%s id=%s",
                            _cur_tool_name, _cur_tool_use_id,
                        )
                    continue

                if evt.inner_type == "content_block_stop":
                    # When a tool_use block ends, emit the complete tool_use event
                    if current_block == "tool_use" and _cur_tool_use_id:
                        tool_input = "".join(_cur_tool_input_chunks)
                        # Try to pretty-format the JSON input for readability
                        tool_input_display = self._format_tool_input(
                            tool_input, _cur_tool_name
                        )
                        log.info(
                            "[MiraBridge] tool_use complete: name=%s id=%s input_len=%d",
                            _cur_tool_name, _cur_tool_use_id, len(tool_input),
                        )
                        _emitted_tool_ids.add(_cur_tool_use_id)
                        yield {
                            "type": "stream_event",
                            "event": StreamEvent(
                                kind="tool_use",
                                text=tool_input_display,
                                tool_name=_cur_tool_name,
                                tool_use_id=_cur_tool_use_id,
                            ),
                        }
                        # Reset tool tracking
                        _cur_tool_name = ""
                        _cur_tool_use_id = ""
                        _cur_tool_input_chunks = []
                    current_block = ""
                    continue

                # Accumulate input_json_delta for tool_use blocks
                if evt.delta_type == "input_json_delta":
                    delta = evt.data.get("event", {}).get("delta", {})
                    partial = delta.get("partial_json", "")
                    if partial:
                        _cur_tool_input_chunks.append(partial)
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

    # ── Tool event helpers ─────────────────────────────────────────

    @staticmethod
    def _extract_assistant_tool_uses(
        evt: Any,
        tool_names: dict[str, str],
        emitted_tool_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Extract tool_use_id→name mappings from assistant message snapshots.

        The frontend (parseStreamingEvents) uses assistant messages as a
        fallback source for tool names via recordToolName(). We replicate
        this: for every tool_use block in the assistant snapshot, record
        the name mapping. If a tool_use was NOT already emitted via
        content_block_stop, emit it now as a tool_use event.

        This handles cases where:
        - content_block_start didn't carry the tool name
        - The entire tool_use lifecycle was delivered only as an assistant
          snapshot (no individual content_block_start/delta/stop events)
        """
        events: list[dict[str, Any]] = []
        msg = evt.data.get("message", {}) if isinstance(evt.data, dict) else {}
        if not isinstance(msg, dict):
            return events

        blocks = msg.get("content", [])
        if not isinstance(blocks, list):
            return events

        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") != "tool_use":
                continue

            tool_id = blk.get("id", "")
            tool_name = blk.get("name", "")

            # Always record the name mapping (primary purpose)
            if tool_id and tool_name:
                tool_names[tool_id] = tool_name

            # If this tool_use wasn't emitted via content_block_stop,
            # emit it now from the assistant snapshot
            if tool_id and tool_id not in emitted_tool_ids:
                emitted_tool_ids.add(tool_id)
                # Format the input for display
                tool_input = blk.get("input", {})
                if isinstance(tool_input, dict):
                    try:
                        raw_json = json.dumps(tool_input, ensure_ascii=False)
                        display = MiraBridge._format_tool_input(raw_json, tool_name)
                    except (TypeError, ValueError):
                        display = str(tool_input)[:500]
                elif isinstance(tool_input, str):
                    display = tool_input[:500]
                else:
                    display = ""

                log.info(
                    "[MiraBridge] tool_use from assistant snapshot: name=%s id=%s",
                    tool_name, tool_id,
                )
                events.append({
                    "type": "stream_event",
                    "event": StreamEvent(
                        kind="tool_use",
                        text=display,
                        tool_name=tool_name,
                        tool_use_id=tool_id,
                    ),
                })

        if events:
            log.info(
                "[MiraBridge] assistant snapshot: recorded %d tool names, emitted %d new tool_use events",
                len([b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]),
                len(events),
            )
        return events

    @staticmethod
    def _parse_user_tool_results(
        evt: Any, tool_names: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Extract tool_result events from a 'user' reason event.

        Mira wraps Claude's tool results in user-type reason events:
        data.message.content = [
            {"type": "tool_result", "tool_use_id": "...", "content": "..."},
            ...
        ]

        Returns a list of stream_event dicts to yield.
        """
        events: list[dict[str, Any]] = []
        msg = evt.data.get("message", {}) if isinstance(evt.data, dict) else {}
        if not isinstance(msg, dict):
            return events

        blocks = msg.get("content", [])
        if not isinstance(blocks, list):
            return events

        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") != "tool_result":
                continue

            tool_use_id = blk.get("tool_use_id", "")
            tool_name = tool_names.get(tool_use_id, "")

            # Extract text from tool_result content
            content = blk.get("content", "")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # content can be [{type: "text", text: "..."}, ...]
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        text_parts.append(item["text"])
                text = "\n".join(text_parts)

            log.info(
                "[MiraBridge] tool_result: name=%s id=%s text_len=%d",
                tool_name or "unknown", tool_use_id, len(text),
            )
            events.append({
                "type": "stream_event",
                "event": StreamEvent(
                    kind="tool_result",
                    text=text,
                    tool_name=tool_name,
                    tool_use_id=tool_use_id,
                ),
            })

        return events

    @staticmethod
    def _format_tool_input(raw_json: str, tool_name: str) -> str:
        """Format tool input JSON for display.

        For readability, extract key fields from common tool inputs.
        Falls back to raw JSON if parsing fails.
        """
        if not raw_json:
            return ""
        try:
            parsed = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            return raw_json[:500]

        # For search tools, show the query
        if isinstance(parsed, dict):
            query = parsed.get("query") or parsed.get("prompt") or parsed.get("content")
            if query and isinstance(query, str):
                return query

            # For web fetch, show the URL
            url = parsed.get("url")
            if not url and isinstance(parsed.get("data"), dict):
                url = parsed["data"].get("url")
            if url and isinstance(url, str):
                return url

        # Fallback: compact JSON
        compact = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        if len(compact) > 500:
            return compact[:500] + "..."
        return compact

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
