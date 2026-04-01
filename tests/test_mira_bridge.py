"""Unit tests for mira_bridge.py — MiraBridge adapter.

All Mira network calls are mocked. Tests validate:
- Event mapping (reason → thinking buffer, content → result)
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
                message_id: str = "", session_id: str = "") -> MiraEvent:
    return MiraEvent(
        event=event,
        data=data or {},
        text=text,
        message_id=message_id,
        session_id=session_id,
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
    async def test_normal_flow(self, bridge):
        """reason → start_content → content → finish = thinking + result"""
        async def mock_chat(session_id, content, model=None):
            yield _make_event("start", text="")
            yield _make_event("reason", text="Step 1. ")
            yield _make_event("reason", text="Step 2.")
            yield _make_event("start_content", text="")
            yield _make_event("content", text="Final answer",
                             data={"content": {"result": "Final answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        # Should get: thinking event, result event, final event
        stream_events = [e for e in events if e["type"] == "stream_event"]
        final_events = [e for e in events if e["type"] == "final"]

        assert len(stream_events) == 2
        assert stream_events[0]["event"].kind == "thinking"
        assert stream_events[0]["event"].text == "Step 1. Step 2."
        assert stream_events[1]["event"].kind == "result"
        assert stream_events[1]["event"].text == "Final answer"

        assert len(final_events) == 1
        assert final_events[0]["result"].result_text == "Final answer"
        assert final_events[0]["result"].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_no_thinking(self, bridge):
        """Direct content without reason events."""
        async def mock_chat(session_id, content, model=None):
            yield _make_event("start_content", text="")
            yield _make_event("content", text="Direct answer",
                             data={"content": {"result": "Direct answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        assert len(stream_events) == 1
        assert stream_events[0]["event"].kind == "result"

    @pytest.mark.asyncio
    async def test_thinking_without_start_content(self, bridge):
        """Reason events without start_content (edge case) — flush at end."""
        async def mock_chat(session_id, content, model=None):
            yield _make_event("reason", text="Orphan thinking")

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
        async def mock_chat(session_id, content, model=None):
            yield _make_event("title", text="My Session Title")
            yield _make_event("content", text="answer",
                             data={"content": {"result": "answer"}})

        bridge._sessions["t1"] = "existing_session"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "question"))

        stream_events = [e for e in events if e["type"] == "stream_event"]
        # Only content → result event, no title event
        assert all(e["event"].kind in ("result", "thinking") for e in stream_events)


# ── Session Creation on stream_ask ─────────────────────────────────

class TestStreamAskSessionCreation:
    @pytest.mark.asyncio
    async def test_auto_creates_session(self, bridge):
        """When no session exists for topic, should auto-create one."""
        async def mock_chat(session_id, content, model=None):
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
        async def mock_chat(session_id, content, model=None):
            yield _make_event("reason", text="thinking")
            raise MiraAuthError("token expired mid-stream")

        bridge._sessions["t1"] = "s1"
        with patch.object(bridge.client, "chat", side_effect=mock_chat):
            events = await _collect_events(bridge.stream_ask("t1", "q"))

        final = [e for e in events if e["type"] == "final"][0]
        assert "expired" in final["result"].assistant_texts[0]

    @pytest.mark.asyncio
    async def test_generic_error_during_chat(self, bridge):
        """Generic exception mid-stream."""
        async def mock_chat(session_id, content, model=None):
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
