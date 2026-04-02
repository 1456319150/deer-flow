"""Unit tests for mira_bridge.py — MiraBridge adapter.

All Mira network calls are mocked. Tests validate:
- Event mapping (reason deltas → immediate thinking/result stream events)
- data_type filtering (assistant/system/safety_audit skipped)
- Tool use lifecycle: content_block_start → input_json_delta → content_block_stop → tool_use event
- Tool results from user events: parsed and yielded as tool_result events
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


def _make_user_event_with_tool_result(
    tool_use_id: str = "", content: str = "", is_error: bool = False
) -> MiraEvent:
    """Create a user-type reason event with proper message.content structure."""
    data = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                    "is_error": is_error,
                }
            ]
        },
    }
    return MiraEvent(
        event="reason",
        data=data,
        text=content,
        data_type="user",
        tool_use_id="",
    )


async def _collect_events(stream) -> list[dict]:
    """Drain an async iterator into a list."""
    events = []
    async for event in stream:
        events.append(event)
    return events


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def tmp_session_file(tmp_path):
    return str(tmp_path / "test-sessions.json")


@pytest.fixture
def bridge_cfg(tmp_session_file):
    return {
        "session_token": "test-token",
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

    def test_init_missing_token_raises(self, tmp_session_file):
        from mira_bridge import MiraBridge
        with pytest.raises(ValueError, match="session token required"):
            MiraBridge({"session_store_path": tmp_session_file})

    def test_init_env_fallback(self, tmp_session_file):
        from mira_bridge import MiraBridge
        with patch.dict(os.environ, {"MIRA_SESSION": "env-token"}):
            b = MiraBridge({"session_store_path": tmp_session_file})
            assert b.client._token == "env-token"

    def test_init_mode_default(self, tmp_session_file):
        from mira_bridge import MiraBridge
        b = MiraBridge({
            "session_token": "t",
            "session_store_path": tmp_session_file,
        })
        assert b.mode == "quick"


# ── Session Persistence Tests ──────────────────────────────────────

class TestSessionPersistence:
    def test_load_empty(self, bridge):
        assert bridge._sessions == {}

    def test_load_existing(self, tmp_session_file, bridge_cfg):
        with open(tmp_session_file, "w") as f:
            json.dump({"topic1": "sess1"}, f)
        from mira_bridge import MiraBridge
        b = MiraBridge(bridge_cfg)
        assert b._sessions == {"topic1": "sess1"}

    def test_load_corrupt_json(self, tmp_session_file, bridge_cfg):
        with open(tmp_session_file, "w") as f:
            f.write("not json")
        from mira_bridge import MiraBridge
        b = MiraBridge(bridge_cfg)
        assert b._sessions == {}

    @pytest.mark.asyncio
    async def test_remember_and_save(self, bridge, tmp_session_file):
        await bridge._remember_session("t1", "s1")
        assert bridge._sessions["t1"] == "s1"
        with open(tmp_session_file) as f:
            saved = json.load(f)
        assert saved["t1"] == "s1"

    @pytest.mark.asyncio
    async def test_remember_noop_same_value(self, bridge):
        bridge._sessions["t1"] = "s1"
        # Should not raise even though file may not exist
        await bridge._remember_session("t1", "s1")
        assert bridge._sessions["t1"] == "s1"

    @pytest.mark.asyncio
    async def test_remember_empty_topic_noop(self, bridge):
        await bridge._remember_session("", "s1")
        assert "" not in bridge._sessions

    def test_get_session(self, bridge):
        bridge._sessions["t1"] = "s1"
        assert bridge.get_session("t1") == "s1"
        assert bridge.get_session("unknown") is None

    @pytest.mark.asyncio
    async def test_reset_session(self, bridge, tmp_session_file):
        bridge._sessions["t1"] = "s1"
        with open(tmp_session_file, "w") as f:
            json.dump({"t1": "s1"}, f)
        with patch.object(bridge.client, "delete_session", new_callable=AsyncMock):
            old = await bridge.reset_session("t1")
        assert old == "s1"
        assert "t1" not in bridge._sessions

    @pytest.mark.asyncio
    async def test_reset_nonexistent(self, bridge):
        old = await bridge.reset_session("nonexistent")
        assert old is None

    @pytest.mark.asyncio
    async def test_reset_empty_topic(self, bridge):
        old = await bridge.reset_session("")
        assert old is None


# ── Stream Event Mapping Tests ─────────────────────────────────────

class TestStreamAskEventMapping:

    @pytest.mark.asyncio
    async def test_normal_flow_streaming_deltas(self, bridge):
        """Verify thinking + text deltas → individual stream events."""
        async def mock_chat(session_id, content, model=None, mode=None):
            # Thinking block
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="thinking", data_type="stream_event")
            yield _make_event("reason", text="Let me think",
                              inner_type="content_block_delta",
                              delta_type="thinking_delta",
                              data_type="stream_event")
            yield _make_event("reason", text=" about this",
                              inner_type="content_block_delta",
                              delta_type="thinking_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Text block
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="Hello ",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", text="world",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Content event
            yield _make_event("content", text="Hello world",
                              data={"content": {"result": "Hello world",
                                                "input_tokens": 100,
                                                "output_tokens": 50}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        assert len(stream_events) == 4  # 2 thinking + 2 result

        # Thinking events
        assert stream_events[0]["event"].kind == "thinking"
        assert stream_events[0]["event"].text == "Let me think"
        assert stream_events[1]["event"].kind == "thinking"
        assert stream_events[1]["event"].text == " about this"

        # Result events
        assert stream_events[2]["event"].kind == "result"
        assert stream_events[2]["event"].text == "Hello "
        assert stream_events[3]["event"].kind == "result"
        assert stream_events[3]["event"].text == "world"

        # Final event
        final = [e for e in events if e["type"] == "final"]
        assert len(final) == 1
        result = final[0]["result"]
        assert result.thinking == ["Let me think about this"]
        assert result.result_text == "Hello world"
        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50

    @pytest.mark.asyncio
    async def test_no_thinking(self, bridge):
        """When no thinking block, only result events emitted."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="Direct answer",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            yield _make_event("content", data={"content": {"result": "Direct answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        assert len(stream_events) == 1
        assert stream_events[0]["event"].kind == "result"
        assert stream_events[0]["event"].text == "Direct answer"

    @pytest.mark.asyncio
    async def test_thinking_only_no_text_block(self, bridge):
        """Thinking-only response (no text block) falls back to content."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="thinking", data_type="stream_event")
            yield _make_event("reason", text="Deep thought",
                              inner_type="content_block_delta",
                              delta_type="thinking_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            yield _make_event("content", data={"content": {"result": "Final answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        thinking = [e for e in stream_events if e["event"].kind == "thinking"]
        result = [e for e in stream_events if e["event"].kind == "result"]
        assert len(thinking) == 1
        assert len(result) == 1  # fallback from content event

    @pytest.mark.asyncio
    async def test_title_event_ignored(self, bridge):
        """Title events should be logged but not yielded."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("title", text="Conversation Title")
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
        assert len(stream_events) == 1
        assert stream_events[0]["event"].kind == "result"

    @pytest.mark.asyncio
    async def test_content_fallback_when_no_stream_deltas(self, bridge):
        """When no reason stream deltas arrive, content event is used as fallback."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("content", data={"content": {"result": "fallback answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        assert len(stream_events) == 1
        assert stream_events[0]["event"].kind == "result"
        assert stream_events[0]["event"].text == "fallback answer"

    @pytest.mark.asyncio
    async def test_multiple_thinking_chunks_streamed_individually(self, bridge):
        """Each thinking delta is yielded as a separate stream event."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="thinking", data_type="stream_event")
            for i in range(5):
                yield _make_event("reason", text=f"chunk{i} ",
                                  inner_type="content_block_delta",
                                  delta_type="thinking_delta",
                                  data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
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
        thinking = [e for e in stream_events if e["event"].kind == "thinking"]
        assert len(thinking) == 5
        for i, e in enumerate(thinking):
            assert e["event"].text == f"chunk{i} "

    @pytest.mark.asyncio
    async def test_legacy_reason_without_inner_type(self, bridge):
        """Reason events without inner_type (legacy format) use direct text."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", text="legacy text", data_type="stream_event")
            yield _make_event("content", data={"content": {"result": "legacy text"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        result_events = [e for e in stream_events if e["event"].kind == "result"]
        assert len(result_events) == 1
        assert result_events[0]["event"].text == "legacy text"

    # ── data_type filtering tests ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_assistant_type_skipped(self, bridge):
        """Assistant-type reason events (accumulated snapshots) are skipped."""
        async def mock_chat(session_id, content, model=None, mode=None):
            # Stream delta (should be kept)
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="real delta",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Assistant snapshot (should be skipped — duplicate)
            yield _make_event("reason", text="real delta", data_type="assistant")
            yield _make_event("content", data={"content": {"result": "real delta"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        result_events = [e for e in stream_events if e["event"].kind == "result"]
        # Only 1 result from the delta, not a duplicate from assistant
        assert len(result_events) == 1
        assert result_events[0]["event"].text == "real delta"

    @pytest.mark.asyncio
    async def test_system_type_skipped(self, bridge):
        """System-type reason events are skipped."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", text="system init", data_type="system")
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
        assert len(stream_events) == 1
        assert stream_events[0]["event"].kind == "result"

    @pytest.mark.asyncio
    async def test_safety_audit_type_skipped(self, bridge):
        """Safety audit events are skipped (recognizer_results, <cis-ctrl>)."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", text="audit data",
                              data_type="safety_audit")
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
        assert len(stream_events) == 1
        assert stream_events[0]["event"].kind == "result"

    # ── user event → tool_result tests ─────────────────────────────

    @pytest.mark.asyncio
    async def test_user_event_yields_tool_result(self, bridge):
        """User-type reason events with tool_result content yield tool_result events."""
        async def mock_chat(session_id, content, model=None, mode=None):
            # First, a tool_use block to register the tool name
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="tool_use", data_type="stream_event",
                              tool_name="web_search", tool_use_id="toolu_abc")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # User event with tool_result
            yield _make_user_event_with_tool_result(
                tool_use_id="toolu_abc",
                content="Search results: 3 items found",
            )
            yield _make_event("content", data={"content": {"result": "answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        tool_result_events = [e for e in stream_events if e["event"].kind == "tool_result"]
        assert len(tool_result_events) == 1
        assert tool_result_events[0]["event"].text == "Search results: 3 items found"
        assert tool_result_events[0]["event"].tool_name == "web_search"
        assert tool_result_events[0]["event"].tool_use_id == "toolu_abc"

    @pytest.mark.asyncio
    async def test_user_event_no_message_structure_yields_nothing(self, bridge):
        """User-type reason events without message.content structure yield nothing."""
        async def mock_chat(session_id, content, model=None, mode=None):
            # User event with no proper structure (just text, no message.content)
            yield _make_event("reason", text="", data_type="user")
            yield _make_event("content", data={"content": {"result": "answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        tool_result_events = [e for e in stream_events if e["event"].kind == "tool_result"]
        assert len(tool_result_events) == 0

    # ── tool_use lifecycle tests ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_tool_use_block_yields_event_on_stop(self, bridge):
        """tool_use block: start → input_json_delta → stop → yields tool_use event."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="tool_use", data_type="stream_event",
                              tool_name="web_search", tool_use_id="toolu_123")
            # input_json_delta with partial JSON in data.event.delta.partial_json
            yield _make_event("reason", text="",
                              inner_type="content_block_delta",
                              delta_type="input_json_delta",
                              data_type="stream_event",
                              data={"event": {"delta": {"partial_json": '{"query": "AI news"}'}}})
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            yield _make_event("content", data={"content": {"result": "answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        tool_use_events = [e for e in stream_events if e["event"].kind == "tool_use"]
        assert len(tool_use_events) == 1
        assert tool_use_events[0]["event"].tool_name == "web_search"
        assert tool_use_events[0]["event"].tool_use_id == "toolu_123"
        # Input should be formatted (query extracted)
        assert "AI news" in tool_use_events[0]["event"].text

    @pytest.mark.asyncio
    async def test_input_json_delta_not_treated_as_text(self, bridge):
        """input_json_delta events are accumulated for tool input, not treated as text."""
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
        """Full flow: thinking → tool_use → tool_result → text answer."""
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
                              data_type="stream_event",
                              data={"event": {"delta": {"partial_json": '{"query":"python"}'}}})
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Assistant summary (should be skipped — duplicate)
            yield _make_event("reason", text="I need to search",
                              data_type="assistant")
            # Tool result from user
            yield _make_user_event_with_tool_result(
                tool_use_id="toolu_789",
                content="Found: Python docs",
            )
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
        assert "tool_use" in kinds
        assert "tool_result" in kinds
        assert "result" in kinds

        # Tool use event has correct metadata
        tool_use = [e for e in stream_events if e["event"].kind == "tool_use"]
        assert len(tool_use) == 1
        assert tool_use[0]["event"].tool_name == "web_search"
        assert tool_use[0]["event"].tool_use_id == "toolu_789"

        # Tool result has correct content and correlates by tool_use_id
        tool_result = [e for e in stream_events if e["event"].kind == "tool_result"]
        assert len(tool_result) == 1
        assert tool_result[0]["event"].text == "Found: Python docs"
        assert tool_result[0]["event"].tool_name == "web_search"
        assert tool_result[0]["event"].tool_use_id == "toolu_789"

        # No duplicate text from assistant summaries
        result_events = [e for e in stream_events if e["event"].kind == "result"]
        assert len(result_events) == 2
        assert result_events[0]["event"].text == "Based on the search, "
        assert result_events[1]["event"].text == "here is the answer."

        final = [e for e in events if e["type"] == "final"][0]
        assert final["result"].result_text == "Based on the search, here is the answer."
        assert final["result"].thinking == ["I need to search"]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, bridge):
        """Multiple tool_use blocks each produce their own tool_use event."""
        async def mock_chat(session_id, content, model=None, mode=None):
            # First tool
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="tool_use", data_type="stream_event",
                              tool_name="web_search", tool_use_id="toolu_001")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Second tool
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="tool_use", data_type="stream_event",
                              tool_name="knowledge_search", tool_use_id="toolu_002")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            # Results
            yield _make_user_event_with_tool_result("toolu_001", "web result")
            yield _make_user_event_with_tool_result("toolu_002", "kb result")
            # Final answer
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="Combined answer",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            yield _make_event("content", data={"content": {"result": "Combined answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        tool_use = [e for e in stream_events if e["event"].kind == "tool_use"]
        tool_result = [e for e in stream_events if e["event"].kind == "tool_result"]

        assert len(tool_use) == 2
        assert tool_use[0]["event"].tool_name == "web_search"
        assert tool_use[1]["event"].tool_name == "knowledge_search"

        assert len(tool_result) == 2
        # tool_name resolved via _tool_names map
        assert tool_result[0]["event"].tool_name == "web_search"
        assert tool_result[0]["event"].text == "web result"
        assert tool_result[1]["event"].tool_name == "knowledge_search"
        assert tool_result[1]["event"].text == "kb result"

    @pytest.mark.asyncio
    async def test_format_tool_input_extracts_query(self, bridge):
        """_format_tool_input extracts query field for search tools."""
        from mira_bridge import MiraBridge
        assert MiraBridge._format_tool_input('{"query": "test search"}', "web_search") == "test search"
        assert MiraBridge._format_tool_input('{"url": "https://example.com"}', "web_fetch") == "https://example.com"
        assert MiraBridge._format_tool_input('{"a": 1}', "unknown") == '{"a":1}'
        assert MiraBridge._format_tool_input("", "test") == ""
        assert MiraBridge._format_tool_input("invalid json{", "test") == "invalid json{"

    @pytest.mark.asyncio
    async def test_no_thinking_status_for_tool_phase(self, bridge):
        """Tool_use blocks produce tool_use events, not a thinking 'researching' status."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="tool_use", data_type="stream_event",
                              tool_name="web_search", tool_use_id="toolu_001")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
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
        thinking_events = [e for e in stream_events if e["event"].kind == "thinking"]
        # No fake "调研中" thinking status — tool_use events replace that
        assert len(thinking_events) == 0

        tool_use_events = [e for e in stream_events if e["event"].kind == "tool_use"]
        assert len(tool_use_events) == 1

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
        """Auth error during session creation → error result."""
        with patch.object(bridge.client, "create_session",
                         new_callable=AsyncMock,
                         side_effect=MiraAuthError("expired")):
            events = await _collect_events(
                bridge.stream_ask("new_topic", "hello")
            )

        final = [e for e in events if e["type"] == "final"][0]
        assert "expired" in final["result"].reply_text

    @pytest.mark.asyncio
    async def test_generic_error_on_create(self, bridge):
        """Generic error during session creation → error result."""
        with patch.object(bridge.client, "create_session",
                         new_callable=AsyncMock,
                         side_effect=Exception("network error")):
            events = await _collect_events(
                bridge.stream_ask("new_topic", "hello")
            )

        final = [e for e in events if e["type"] == "final"][0]
        assert "network error" in final["result"].reply_text


# ── Error Handling ─────────────────────────────────────────────────

class TestStreamAskErrors:
    @pytest.mark.asyncio
    async def test_auth_error_during_chat(self, bridge):
        """Auth error during chat stream → error in final result."""
        async def mock_chat(session_id, content, model=None, mode=None):
            raise MiraAuthError("token expired")
            yield  # make it a generator

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        final = [e for e in events if e["type"] == "final"][0]
        assert "expired" in final["result"].reply_text

    @pytest.mark.asyncio
    async def test_generic_error_during_chat(self, bridge):
        """Generic error during chat → error in final result."""
        async def mock_chat(session_id, content, model=None, mode=None):
            raise RuntimeError("stream broken")
            yield

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        final = [e for e in events if e["type"] == "final"][0]
        assert "stream broken" in final["result"].reply_text


# ── Usage Extraction Tests ─────────────────────────────────────────

class TestUsageExtraction:
    def test_extract_usage_with_tokens(self):
        from mira_bridge import MiraBridge
        usage = MiraBridge._extract_usage({
            "input_tokens": 100, "output_tokens": 200
        })
        assert usage.input_tokens == 100
        assert usage.output_tokens == 200
        assert usage.total_tokens == 300

    def test_extract_usage_with_cost(self):
        from mira_bridge import MiraBridge
        usage = MiraBridge._extract_usage({
            "input_tokens": 10, "output_tokens": 20, "cost": "$0.0015"
        })
        assert usage.cost_usd == 0.0015

    def test_extract_usage_none(self):
        from mira_bridge import MiraBridge
        usage = MiraBridge._extract_usage({})
        assert usage is None

    def test_extract_usage_total_cost_usd(self):
        from mira_bridge import MiraBridge
        usage = MiraBridge._extract_usage({
            "input_tokens": 50, "output_tokens": 100, "total_cost_usd": "0.005"
        })
        assert usage.cost_usd == 0.005


# ── Interface Parity Tests ─────────────────────────────────────────

class TestInterfaceParity:
    """Verify MiraBridge has the same methods as ClaudeCodeBridge."""

    def test_has_stream_ask(self, bridge):
        assert hasattr(bridge, "stream_ask")

    def test_has_ask(self, bridge):
        assert hasattr(bridge, "ask")

    def test_has_get_session(self, bridge):
        assert hasattr(bridge, "get_session")

    def test_has_reset_session(self, bridge):
        assert hasattr(bridge, "reset_session")

    def test_stream_ask_is_async(self, bridge):
        import inspect
        assert inspect.isasyncgenfunction(bridge.stream_ask)

    def test_ask_is_coroutine(self, bridge):
        import inspect
        assert inspect.iscoroutinefunction(bridge.ask)

    def test_reset_session_is_coroutine(self, bridge):
        import inspect
        assert inspect.iscoroutinefunction(bridge.reset_session)


# ── Ask (non-streaming) Tests ──────────────────────────────────────

class TestAsk:
    @pytest.mark.asyncio
    async def test_ask_returns_result_text(self, bridge):
        """ask() consumes stream and returns final StreamResult."""
        async def mock_chat(session_id, content, model=None, mode=None):
            yield _make_event("reason", inner_type="content_block_start",
                              block_type="text", data_type="stream_event")
            yield _make_event("reason", text="the answer",
                              inner_type="content_block_delta",
                              delta_type="text_delta",
                              data_type="stream_event")
            yield _make_event("reason", inner_type="content_block_stop",
                              data_type="stream_event")
            yield _make_event("content", data={"content": {"result": "the answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            result = await bridge.ask("t1", "question")

        assert result.result_text == "the answer"
        assert result.stop_reason == "end_turn"
