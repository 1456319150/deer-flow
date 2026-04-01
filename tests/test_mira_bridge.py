"""Unit tests for mira_bridge.py — MiraBridge adapter.

All Mira network calls are mocked. Tests validate:
- Event mapping (reason deltas → immediate thinking/result stream events)
- data_type filtering (assistant/system/user skipped — internal tool calls hidden)
- Tool use events silently logged (not surfaced to user)
- input_json_delta skipped (not display text)
- Session persistence (load/save/reset)
- Error handling (auth, generic)
- Interface parity with ClaudeCodeBridge
"""
from __future__ import annotations

import asyncio
import json
import os
import pytest
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mira_client import MiraClient, MiraEvent, MiraAuthError, MiraAPIError
from gateway import StreamResult, StreamEvent, UsageSummary


# ── Helpers ────────────────────────────────────────────────────────

def _make_event(event: str, text: str = "", data: dict = None,
                message_id: str = "", session_id: str = "",
                block_type: str = "", delta_type: str = "",
                inner_type: str = "", data_type: str = "",
                tool_name: str = "", tool_use_id: str = "") -> MiraEvent:
    return MiraEvent(
        event=event,
        data=data or {},
        text=text,
        message_id=message_id,
        session_id=session_id,
        block_type=block_type,
        delta_type=delta_type,
        inner_type=inner_type,
        data_type=data_type,
        tool_name=tool_name,
        tool_use_id=tool_use_id,
    )


async def _collect_events(stream) -> list[dict]:
    """Collect all events from an async iterator."""
    events = []
    async for e in stream:
        events.append(e)
    return events


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def tmp_session_file(tmp_path):
    return str(tmp_path / "test-sessions.json")


@pytest.fixture
def bridge_cfg(tmp_session_file):
    return {
        "session_token": "test_jwt_token",
        "base_url": "https://mira.test.net",
        "model": "test-model",
        "mode": "deep",
        "timeout": 60,
        "session_store_path": tmp_session_file,
    }


@pytest.fixture
def bridge(bridge_cfg):
    from mira_bridge import MiraBridge
    return MiraBridge(bridge_cfg)


# ── Init Tests ─────────────────────────────────────────────────────

class TestBridgeInit:
    def test_init_with_config(self, bridge):
        assert bridge.model == "test-model"
        assert bridge.mode == "deep"
        assert bridge.timeout == 60

    def test_init_missing_token_raises(self, tmp_session_file):
        from mira_bridge import MiraBridge
        with patch.dict(os.environ, {}, clear=True):
            # Remove MIRA_SESSION from env if present
            os.environ.pop("MIRA_SESSION", None)
            with pytest.raises(ValueError, match="session token"):
                MiraBridge({"session_store_path": tmp_session_file})

    def test_init_env_fallback(self, tmp_session_file):
        from mira_bridge import MiraBridge
        with patch.dict(os.environ, {"MIRA_SESSION": "env_token"}):
            b = MiraBridge({"session_store_path": tmp_session_file})
            assert b.client._token == "env_token"

    def test_init_mode_default(self, tmp_session_file):
        from mira_bridge import MiraBridge
        b = MiraBridge({
            "session_token": "tok",
            "session_store_path": tmp_session_file,
        })
        assert b.mode == "quick"  # default when not specified


# ── Session Persistence Tests ──────────────────────────────────────

class TestSessionPersistence:
    def test_load_empty(self, bridge):
        assert bridge._sessions == {}

    def test_load_existing(self, tmp_session_file, bridge_cfg):
        from mira_bridge import MiraBridge
        with open(tmp_session_file, "w") as f:
            json.dump({"topic_1": "sess_a", "topic_2": "sess_b"}, f)
        b = MiraBridge(bridge_cfg)
        assert b._sessions == {"topic_1": "sess_a", "topic_2": "sess_b"}

    def test_load_corrupt_json(self, tmp_session_file, bridge_cfg):
        from mira_bridge import MiraBridge
        with open(tmp_session_file, "w") as f:
            f.write("{bad json")
        b = MiraBridge(bridge_cfg)
        assert b._sessions == {}

    @pytest.mark.asyncio
    async def test_remember_and_save(self, bridge, tmp_session_file):
        await bridge._remember_session("t1", "s1")
        assert bridge._sessions["t1"] == "s1"
        # Check file was written
        with open(tmp_session_file) as f:
            data = json.load(f)
        assert data["t1"] == "s1"

    @pytest.mark.asyncio
    async def test_remember_noop_same_value(self, bridge):
        bridge._sessions["t1"] = "s1"
        # Should not trigger save since value is same
        with patch.object(bridge, "_save_sessions") as mock_save:
            await bridge._remember_session("t1", "s1")
            mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_remember_empty_topic_noop(self, bridge):
        with patch.object(bridge, "_save_sessions") as mock_save:
            await bridge._remember_session("", "s1")
            mock_save.assert_not_called()

    def test_get_session(self, bridge):
        bridge._sessions["t1"] = "s1"
        assert bridge.get_session("t1") == "s1"
        assert bridge.get_session("nonexistent") is None

    @pytest.mark.asyncio
    async def test_reset_session(self, bridge, tmp_session_file):
        bridge._sessions["t1"] = "s1"
        bridge._save_sessions()

        with patch.object(bridge.client, "delete_session", new_callable=AsyncMock):
            old = await bridge.reset_session("t1")
        assert old == "s1"
        assert "t1" not in bridge._sessions

    @pytest.mark.asyncio
    async def test_reset_nonexistent(self, bridge):
        result = await bridge.reset_session("nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_reset_empty_topic(self, bridge):
        result = await bridge.reset_session("")
        assert result is None


# ── stream_ask Event Mapping Tests ─────────────────────────────────

class TestStreamAskEventMapping:
    @pytest.mark.asyncio
    async def test_normal_flow_streaming_deltas(self, bridge):
        """Thinking deltas + text deltas → individual stream events."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("start", text="")
            # thinking block
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="thinking", data_type="stream_event")
            yield _make_event("reason", text="Step 1. ",
                              inner_type="content_block_delta",
                              delta_type="thinking_delta",
                              data_type="stream_event")
            yield _make_event("reason", text="Step 2.",
                              inner_type="content_block_delta",
                              delta_type="thinking_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # text block
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="Final ",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", text="answer",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # content event with usage
            yield _make_event("content", text="Final answer",
                              data={"content": {"result": "Final answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        final_events = [e for e in events if e["type"] == "final"]

        thinking_events = [e for e in stream_events if e["event"].kind == "thinking"]
        result_events = [e for e in stream_events if e["event"].kind == "result"]

        assert len(thinking_events) == 2
        assert thinking_events[0]["event"].text == "Step 1. "
        assert thinking_events[1]["event"].text == "Step 2."

        assert len(result_events) == 2
        assert result_events[0]["event"].text == "Final "
        assert result_events[1]["event"].text == "answer"

        assert len(final_events) == 1
        assert final_events[0]["result"].result_text == "Final answer"
        assert final_events[0]["result"].stop_reason == "end_turn"
        assert final_events[0]["result"].thinking == ["Step 1. Step 2."]

    @pytest.mark.asyncio
    async def test_no_thinking(self, bridge):
        """Direct text deltas without thinking block."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text")
            yield _make_event("reason", text="Direct answer",
                              inner_type="content_block_delta",
                              delta_type="text_delta")
            yield _make_event("reason", inner_type="content_block_stop")
            yield _make_event("content", text="Direct answer",
                              data={"content": {"result": "Direct answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        assert len(stream_events) == 1
        assert stream_events[0]["event"].kind == "result"
        assert stream_events[0]["event"].text == "Direct answer"

    @pytest.mark.asyncio
    async def test_thinking_only_no_text_block(self, bridge):
        """Thinking deltas without a text block — flushed at end."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="thinking")
            yield _make_event("reason", text="Orphan thinking",
                              inner_type="content_block_delta",
                              delta_type="thinking_delta")
            yield _make_event("reason", inner_type="content_block_stop")

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        assert len(stream_events) == 1
        assert stream_events[0]["event"].kind == "thinking"
        assert stream_events[0]["event"].text == "Orphan thinking"

    @pytest.mark.asyncio
    async def test_title_event_ignored(self, bridge):
        """Title events should not produce stream events."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("title", text="My Session Title")
            yield _make_event("content", text="answer",
                             data={"content": {"result": "answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        # Only content → result event, no title event
        assert all(e["event"].kind in ("result", "thinking") for e in stream_events)

    @pytest.mark.asyncio
    async def test_content_fallback_when_no_stream_deltas(self, bridge):
        """Content event result used as fallback when no text deltas streamed."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("content", text="Fallback result",
                              data={"content": {"result": "Fallback result"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        assert len(stream_events) == 1
        assert stream_events[0]["event"].kind == "result"
        assert stream_events[0]["event"].text == "Fallback result"

    @pytest.mark.asyncio
    async def test_multiple_thinking_chunks_streamed_individually(self, bridge):
        """Each thinking delta yields its own stream event (not buffered)."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="thinking")
            yield _make_event("reason", text="A",
                              inner_type="content_block_delta",
                              delta_type="thinking_delta")
            yield _make_event("reason", text="B",
                              inner_type="content_block_delta",
                              delta_type="thinking_delta")
            yield _make_event("reason", text="C",
                              inner_type="content_block_delta",
                              delta_type="thinking_delta")
            yield _make_event("reason", inner_type="content_block_stop")
            yield _make_event("content", data={"content": {"result": "done"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        thinking_events = [
            e for e in events
            if e["type"] == "stream_event" and e["event"].kind == "thinking"
        ]
        assert len(thinking_events) == 3
        assert thinking_events[0]["event"].text == "A"
        assert thinking_events[1]["event"].text == "B"
        assert thinking_events[2]["event"].text == "C"

        final = [e for e in events if e["type"] == "final"][0]
        assert final["result"].thinking == ["ABC"]

    @pytest.mark.asyncio
    async def test_legacy_reason_without_inner_type(self, bridge):
        """Reason events without inner_type (legacy format) still work."""
        async def mock_chat(session_id, content, model=None, mode=None):
            # Legacy-style reason event: just has text, no block structure
            yield _make_event("reason", text="legacy thinking")
            yield _make_event("content", data={"content": {"result": "answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        # Legacy reason with no block_type/delta_type → treated as answer (text block)
        # because current_block is "" and delta_type is ""
        assert len(stream_events) >= 1

    # ── NEW: data_type filtering tests ─────────────────────────────

    @pytest.mark.asyncio
    async def test_assistant_type_skipped(self, bridge):
        """Assistant-type reason events (full snapshots) are skipped to avoid duplicates."""
        async def mock_chat(session_id, content, model=None, mode=None):
            # Stream deltas first
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="Hello world",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Assistant summary (duplicate — should be skipped)
            yield _make_event("reason", text="Hello world",
                              data_type="assistant")
            # Content event
            yield _make_event("content", text="Hello world",
                              data={"content": {"result": "Hello world"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        result_events = [e for e in stream_events if e["event"].kind == "result"]
        # Should only get 1 result event (the stream delta), not 2
        assert len(result_events) == 1
        assert result_events[0]["event"].text == "Hello world"

    @pytest.mark.asyncio
    async def test_system_type_skipped(self, bridge):
        """System-type reason events are skipped."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", data_type="system")
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="answer",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            yield _make_event("content", data={"content": {"result": "answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        # No system events should appear
        assert all(e["event"].kind in ("result", "thinking") for e in stream_events)
        assert len(stream_events) == 1


    @pytest.mark.asyncio
    async def test_safety_audit_type_skipped(self, bridge):
        """Safety audit metadata events are skipped and don't leak to output."""
        async def mock_chat(session_id, content, model=None, mode=None):
            # Normal text delta
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="Hello world",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Safety audit metadata (should be skipped)
            yield _make_event("reason", text="", data_type="safety_audit",
                              data={"recognizer_results": [], "dangerous": False})
            # Content event
            yield _make_event("content", text="Hello world",
                              data={"content": {"result": "Hello world"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        result_events = [e for e in stream_events if e["event"].kind == "result"]
        # Should only get 1 result event (the text delta), safety_audit skipped
        assert len(result_events) == 1
        assert result_events[0]["event"].text == "Hello world"
        # No event should contain recognizer_results text
        for e in stream_events:
            assert "recognizer_results" not in e["event"].text
    @pytest.mark.asyncio
    async def test_user_type_silently_skipped(self, bridge):
        """User-type reason events (internal tool results) are silently skipped."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", text="Search results: 3 items found",
                              data_type="user")
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="answer",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            yield _make_event("content", data={"content": {"result": "answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        # No tool_result events should appear — internal tool results are hidden
        tool_result_events = [e for e in stream_events if e["event"].kind == "tool_result"]
        assert len(tool_result_events) == 0
        # Only the text delta should appear
        result_events = [e for e in stream_events if e["event"].kind == "result"]
        assert len(result_events) == 1

    @pytest.mark.asyncio
    async def test_user_type_empty_text_silently_skipped(self, bridge):
        """User-type reason events with no text are silently skipped."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", text="", data_type="user")
            yield _make_event("content", data={"content": {"result": "answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        tool_result_events = [e for e in stream_events if e["event"].kind == "tool_result"]
        assert len(tool_result_events) == 0

    # ── NEW: tool_use event tests ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_tool_use_block_silently_logged(self, bridge):
        """content_block_start with tool_use is silently logged, not surfaced."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="tool_use", data_type="stream_event",
                              tool_name="web_search", tool_use_id="toolu_123")
            # input_json_delta should be skipped
            yield _make_event("reason", text="",
                              inner_type="content_block_delta",
                              delta_type="input_json_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            yield _make_event("content", data={"content": {"result": "answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        tool_use_events = [e for e in stream_events if e["event"].kind == "tool_use"]
        # Tool use events should NOT be surfaced to user
        assert len(tool_use_events) == 0

    @pytest.mark.asyncio
    async def test_input_json_delta_skipped(self, bridge):
        """input_json_delta events are not treated as text."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="tool_use", data_type="stream_event",
                              tool_name="calculator", tool_use_id="toolu_456")
            yield _make_event("reason", text="",
                              inner_type="content_block_delta",
                              delta_type="input_json_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Then a text block with actual answer
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="The result is 42",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            yield _make_event("content", data={"content": {"result": "The result is 42"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        result_events = [e for e in stream_events if e["event"].kind == "result"]
        # Only the text delta, not input_json_delta
        assert len(result_events) == 1
        assert result_events[0]["event"].text == "The result is 42"

    @pytest.mark.asyncio
    async def test_full_tool_use_flow(self, bridge):
        """Full flow: thinking → tool_use (hidden) → tool_result (hidden) → text answer."""
        async def mock_chat(session_id, content, model=None, mode=None):
            # Thinking
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="thinking", data_type="stream_event")
            yield _make_event("reason", text="I need to search",
                              inner_type="content_block_delta",
                              delta_type="thinking_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Tool use
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="tool_use", data_type="stream_event",
                              tool_name="web_search", tool_use_id="toolu_789")
            yield _make_event("reason", inner_type="content_block_delta",
                              delta_type="input_json_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Assistant summary (should be skipped)
            yield _make_event("reason", text="I need to search",
                              data_type="assistant")
            # Tool result from user
            yield _make_event("reason", text="Found: Python docs",
                              data_type="user")
            # Text answer
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="Based on the search, ",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", text="here is the answer.",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Another assistant summary (should be skipped)
            yield _make_event("reason", text="Based on the search, here is the answer.",
                              data_type="assistant")
            # Content
            yield _make_event("content", data={"content": {"result": "Based on the search, here is the answer."}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        kinds = [e["event"].kind for e in stream_events]

        assert "thinking" in kinds
        assert "result" in kinds
        # Internal tool calls should NOT be surfaced
        assert "tool_use" not in kinds
        assert "tool_result" not in kinds

        # Verify no duplicate text from assistant summaries
        result_events = [e for e in stream_events if e["event"].kind == "result"]
        assert len(result_events) == 2  # "Based on the search, " + "here is the answer."
        assert result_events[0]["event"].text == "Based on the search, "
        assert result_events[1]["event"].text == "here is the answer."

        final = [e for e in events if e["type"] == "final"][0]
        assert final["result"].result_text == "Based on the search, here is the answer."
        assert final["result"].thinking == ["I need to search"]


# ── Session Creation on stream_ask ─────────────────────────────────

class TestStreamAskSessionCreation:
    @pytest.mark.asyncio
    async def test_auto_creates_session(self, bridge):
        """When no session exists for topic, should auto-create one."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("content", text="response",
                             data={"content": {"result": "response"}})

        with patch.object(bridge.client, "create_session",
                         new_callable=AsyncMock, return_value="new_sess"):
            with patch.object(bridge.client, "chat", side_effect=mock_chat):
                events = await _collect_events(
                    bridge.stream_ask("new_topic", "hello")
                )

        final = [e for e in events if e["type"] == "final"][0]
        assert final["result"].session_id == "new_sess"

    @pytest.mark.asyncio
    async def test_auth_error_on_create(self, bridge):
        """Auth error during session creation returns error message."""
        with patch.object(bridge.client, "create_session",
                         new_callable=AsyncMock,
                         side_effect=MiraAuthError("expired")):
            events = await _collect_events(
                bridge.stream_ask("topic_x", "hello")
            )

        final = [e for e in events if e["type"] == "final"][0]
        assert "expired" in final["result"].assistant_texts[0]

    @pytest.mark.asyncio
    async def test_generic_error_on_create(self, bridge):
        """Generic error during session creation."""
        with patch.object(bridge.client, "create_session",
                         new_callable=AsyncMock,
                         side_effect=RuntimeError("network down")):
            events = await _collect_events(
                bridge.stream_ask("topic_x", "hello")
            )

        final = [e for e in events if e["type"] == "final"][0]
        assert "network down" in final["result"].assistant_texts[0]


# ── Error During Streaming ─────────────────────────────────────────

class TestStreamAskErrors:
    @pytest.mark.asyncio
    async def test_auth_error_during_chat(self, bridge):
        """Auth error mid-stream yields error in final result."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", text="thinking",
                              inner_type="content_block_delta",
                              delta_type="thinking_delta")
            raise MiraAuthError("token expired mid-stream")

        bridge._sessions["t1"] = "s1"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "q"))

        final = [e for e in events if e["type"] == "final"][0]
        assert "expired" in final["result"].assistant_texts[0]

    @pytest.mark.asyncio
    async def test_generic_error_during_chat(self, bridge):
        """Generic exception mid-stream."""
        async def mock_chat(session_id, content, model=None, mode=None):
            raise RuntimeError("something broke")
            yield  # make it an async generator

        bridge._sessions["t1"] = "s1"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "q"))

        final = [e for e in events if e["type"] == "final"][0]
        assert "something broke" in final["result"].assistant_texts[0]


# ── Usage Extraction Tests ─────────────────────────────────────────

class TestUsageExtraction:
    def test_extract_usage_with_tokens(self):
        from mira_bridge import MiraBridge
        data = {"input_tokens": 100, "output_tokens": 50}
        usage = MiraBridge._extract_usage(data)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_extract_usage_with_cost(self):
        from mira_bridge import MiraBridge
        data = {"input_tokens": 100, "output_tokens": 50, "cost": "$0.05"}
        usage = MiraBridge._extract_usage(data)
        assert usage.cost_usd == 0.05

    def test_extract_usage_none(self):
        from mira_bridge import MiraBridge
        data = {"some_other": "field"}
        usage = MiraBridge._extract_usage(data)
        assert usage is None

    def test_extract_usage_total_cost_usd(self):
        from mira_bridge import MiraBridge
        data = {"input_tokens": 10, "total_cost_usd": 0.001}
        usage = MiraBridge._extract_usage(data)
        assert usage.cost_usd == 0.001


# ── Interface Parity Tests ─────────────────────────────────────────

class TestInterfaceParity:
    """Verify MiraBridge has the same methods as expected by bots."""

    def test_has_stream_ask(self, bridge):
        assert callable(getattr(bridge, "stream_ask", None))

    def test_has_ask(self, bridge):
        assert callable(getattr(bridge, "ask", None))

    def test_has_get_session(self, bridge):
        assert callable(getattr(bridge, "get_session", None))

    def test_has_reset_session(self, bridge):
        assert callable(getattr(bridge, "reset_session", None))

    def test_stream_ask_is_async(self, bridge):
        import inspect
        assert inspect.isasyncgenfunction(bridge.stream_ask)

    def test_ask_is_coroutine(self, bridge):
        import inspect
        assert inspect.iscoroutinefunction(bridge.ask)

    def test_reset_session_is_coroutine(self, bridge):
        import inspect
        assert inspect.iscoroutinefunction(bridge.reset_session)


# ── ask() Convenience Tests ────────────────────────────────────────

class TestAsk:
    @pytest.mark.asyncio
    async def test_ask_returns_result_text(self, bridge):
        """ask() should return the final result text."""
        async def mock_stream_ask(topic_id, prompt):
            yield {
                "type": "stream_event",
                "event": StreamEvent(kind="result", text="answer"),
            }
            yield {
                "type": "final",
                "result": StreamResult(result_text="answer", stop_reason="end_turn"),
            }

        with patch.object(bridge, "stream_ask", side_effect=mock_stream_ask):
            result = await bridge.ask("t1", "question")

        assert isinstance(result, StreamResult)
        assert result.result_text == "answer"
