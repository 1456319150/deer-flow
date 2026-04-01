"""Integration smoke test — hits real Mira API.

Requires MIRA_SESSION env var or explicit token.
Run with:  pytest tests/test_mira_smoke.py -v -s
Skip in CI with: pytest -m "not smoke"
"""
from __future__ import annotations

import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mira_client import MiraClient, MiraEvent, MiraAuthError

TOKEN = os.environ.get("MIRA_SESSION", "")

pytestmark = pytest.mark.smoke


@pytest.fixture
def client():
    if not TOKEN:
        pytest.skip("MIRA_SESSION not set")
    return MiraClient(TOKEN)


class TestSmokeCreateListDelete:
    """Create a session → list → delete. Validates basic CRUD."""

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, client):
        # 1) Create
        sid = await client.create_session(model="re-o-46")
        assert sid, "create_session returned empty session_id"
        print(f"\n  ✅ created session: {sid}")

        # 2) List — should include the new session
        data = await client.list_sessions(page_size=20)
        ids = [
            str(s.get("sessionId", ""))
            for s in data.get("sessions", data.get("sessionItems", []))
        ]
        assert sid in ids, f"new session {sid} not found in list"
        print(f"  ✅ session visible in list ({len(ids)} total)")

        # 3) Delete
        await client.delete_session(sid)
        print(f"  ✅ deleted session: {sid}")


class TestSmokeChat:
    """Send a simple prompt and collect SSE events."""

    @pytest.mark.asyncio
    async def test_chat_stream(self, client):
        sid = await client.create_session(model="re-o-46")
        print(f"\n  session: {sid}")

        events: list[MiraEvent] = []
        async for evt in client.chat(sid, "Reply with exactly: PONG"):
            events.append(evt)
            if evt.event in ("content", "reason"):
                print(f"  [{evt.event}] {evt.text[:80]}")

        event_types = {e.event for e in events}
        # Must have at least start + content
        assert "content" in event_types, f"no content event, got: {event_types}"
        print(f"  ✅ got {len(events)} events, types: {event_types}")

        # Verify answer contains PONG
        answer_texts = [e.text for e in events if e.event == "content"]
        full_answer = "".join(answer_texts)
        assert "PONG" in full_answer.upper(), f"unexpected answer: {full_answer[:200]}"
        print(f"  ✅ answer: {full_answer[:100]}")

        # Cleanup
        await client.delete_session(sid)


class TestSmokeBridge:
    """Test MiraBridge end-to-end via stream_ask."""

    @pytest.mark.asyncio
    async def test_bridge_stream_ask(self):
        if not TOKEN:
            pytest.skip("MIRA_SESSION not set")

        from mira_bridge import MiraBridge
        bridge = MiraBridge({
            "session_token": TOKEN,
            "model": "re-o-46",
            "session_store_path": "/tmp/test-mira-sessions.json",
        })

        events = []
        async for evt in bridge.stream_ask("smoke_test_topic", "Reply with exactly: HELLO"):
            events.append(evt)
            if evt["type"] == "stream_event":
                se = evt["event"]
                print(f"  [{se.kind}] {se.text[:80]}")

        final = [e for e in events if e["type"] == "final"]
        assert len(final) == 1, "missing final event"

        result = final[0]["result"]
        assert result.result_text, "empty result_text"
        assert result.stop_reason == "end_turn"
        print(f"  ✅ result: {result.result_text[:100]}")
        print(f"  ✅ session_id: {result.session_id}")

        # Cleanup
        if result.session_id:
            await bridge.client.delete_session(result.session_id)
            # Also remove from bridge sessions
            bridge._sessions.pop("smoke_test_topic", None)
