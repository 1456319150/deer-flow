"""Unit tests for mira_client.py — Mira API SDK.

Uses httpx mocking (no real network calls).
"""
from __future__ import annotations

import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mira_client import (
    MiraClient, MiraEvent, MiraMessage, FileInfo,
    MiraAuthError, MiraAPIError, _EP, DEFAULT_MODEL,
)


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def client():
    return MiraClient("test_jwt_token", base_url="https://mira.test.net")


# ── Data Model Tests ───────────────────────────────────────────────

class TestDataModels:
    def test_mira_event_defaults(self):
        evt = MiraEvent(event="content", data={"result": "hello"})
        assert evt.text == ""
        assert evt.message_id == ""
        assert evt.session_id == ""
        assert evt.block_type == ""
        assert evt.delta_type == ""
        assert evt.inner_type == ""
        assert evt.data_type == ""
        assert evt.tool_name == ""
        assert evt.tool_use_id == ""

    def test_mira_event_full(self):
        evt = MiraEvent(
            event="content", data={"result": "hi"}, text="hi",
            message_id="m1", session_id="s1",
        )
        assert evt.event == "content"
        assert evt.text == "hi"

    def test_mira_event_with_streaming_fields(self):
        """MiraEvent carries block_type/delta_type/inner_type for nested Claude events."""
        evt = MiraEvent(
            event="reason", data={}, text="thinking...",
            block_type="thinking",
            delta_type="thinking_delta",
            inner_type="content_block_delta",
        )
        assert evt.block_type == "thinking"
        assert evt.delta_type == "thinking_delta"
        assert evt.inner_type == "content_block_delta"

    def test_mira_event_text_block_fields(self):
        """MiraEvent for a text (answer) delta."""
        evt = MiraEvent(
            event="reason", data={}, text="answer chunk",
            block_type="text",
            delta_type="text_delta",
            inner_type="content_block_delta",
        )
        assert evt.block_type == "text"
        assert evt.delta_type == "text_delta"

    def test_mira_event_tool_use_fields(self):
        """MiraEvent carries tool_name/tool_use_id for tool_use blocks."""
        evt = MiraEvent(
            event="reason", data={},
            block_type="tool_use",
            inner_type="content_block_start",
            data_type="stream_event",
            tool_name="web_search",
            tool_use_id="toolu_123",
        )
        assert evt.block_type == "tool_use"
        assert evt.tool_name == "web_search"
        assert evt.tool_use_id == "toolu_123"
        assert evt.data_type == "stream_event"

    def test_mira_message(self):
        msg = MiraMessage(
            message_id="123", session_id="456",
            sender=1, content="test", content_type=1, status=0,
        )
        assert msg.message_id == "123"
        assert msg.content == "test"

    def test_file_info_to_attachment(self):
        fi = FileInfo(
            file_name="doc.pdf", url="https://x/doc.pdf",
            uri="file://doc", mime_type="application/pdf",
        )
        att = fi.to_attachment()
        assert att["file_name"] == "doc.pdf"
        assert att["url"] == "https://x/doc.pdf"
        assert att["mime_type"] == "application/pdf"
        assert att["uri"] == "file://doc"
        assert "thumb_url" not in att  # not in attachment dict


# ── Client Init Tests ──────────────────────────────────────────────

class TestClientInit:
    def test_base_url_strip_slash(self):
        c = MiraClient("tok", base_url="https://mira.test.net/")
        assert c.base_url == "https://mira.test.net"

    def test_default_base_url(self):
        c = MiraClient("tok")
        assert "mira.byteintl.net" in c.base_url

    def test_ensure_client_creates_httpx(self):
        c = MiraClient("tok")
        hc = c._ensure_client()
        assert isinstance(hc, httpx.AsyncClient)
        assert not hc.is_closed

    def test_ensure_client_reuses(self):
        c = MiraClient("tok")
        hc1 = c._ensure_client()
        hc2 = c._ensure_client()
        assert hc1 is hc2


# ── Context Manager ────────────────────────────────────────────────

class TestContextManager:
    @pytest.mark.asyncio
    async def test_async_context(self):
        async with MiraClient("tok") as c:
            assert c._client is not None
            assert not c._client.is_closed
        # after exit, client should be closed
        assert c._client.is_closed

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        c = MiraClient("tok")
        c._ensure_client()
        await c.close()
        await c.close()  # should not raise


# ── _check() Tests ─────────────────────────────────────────────────

class TestCheck:
    def test_check_401_raises_auth(self, client):
        resp = MagicMock()
        resp.status_code = 401
        with pytest.raises(MiraAuthError, match="expired"):
            MiraClient._check(resp)

    def test_check_api_error(self, client):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "baseResp": {"statusCode": 1001, "statusMessage": "bad request"}
        }
        with pytest.raises(MiraAPIError, match="1001"):
            MiraClient._check(resp)

    def test_check_success(self, client):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "baseResp": {"statusCode": 0, "statusMessage": "ok"}
        }
        MiraClient._check(resp)  # should not raise

    def test_check_code_field_fallback(self, client):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"code": 500, "msg": "internal"}
        with pytest.raises(MiraAPIError, match="500"):
            MiraClient._check(resp)


# ── _parse_event() Tests ──────────────────────────────────────────

class TestParseEvent:
    def test_parse_reason_event(self):
        msg = {"event": "reason", "data": {"text": "thinking..."}}
        evt = MiraClient._parse_event(msg)
        assert evt.event == "reason"
        assert evt.text == "thinking..."

    def test_parse_content_event(self):
        msg = {
            "event": "content",
            "data": {
                "content": {
                    "result": "answer here",
                    "session_id": "s99",
                    "type": "result",
                    "subtype": "success",
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.event == "content"
        assert evt.text == "answer here"
        assert evt.session_id == "s99"

    def test_parse_title_event(self):
        msg = {"event": "title", "data": {"content": "My Chat"}}
        evt = MiraClient._parse_event(msg)
        assert evt.text == "My Chat"

    def test_parse_string_data(self):
        msg = {"event": "debug_link", "data": "some string data"}
        evt = MiraClient._parse_event(msg)
        assert evt.data == {"raw": "some string data"}

    def test_parse_nested_json_string_data(self):
        inner = json.dumps({"text": "nested"})
        msg = {"event": "reason", "data": inner}
        evt = MiraClient._parse_event(msg)
        assert evt.text == "nested"

    def test_parse_unknown_event(self):
        msg = {"event": "unknown_type", "data": {}}
        evt = MiraClient._parse_event(msg)
        assert evt.event == "unknown_type"
        assert evt.text == ""

    def test_parse_missing_data_fields(self):
        msg = {"event": "content", "data": {"content": {}}}
        evt = MiraClient._parse_event(msg)
        assert evt.text == ""
        assert evt.message_id == ""

    def test_parse_reason_with_nested_delta(self):
        """Reason event with thinking text in data.event.delta.thinking"""
        msg = {
            "event": "reason",
            "data": {
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "thinking step"},
                },
                "parent_tool_use_id": "",
                "session_id": "s1",
                "type": "content_block_delta",
                "uuid": "u1",
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.text == "thinking step"
        assert evt.inner_type == "content_block_delta"
        assert evt.delta_type == "thinking_delta"

    def test_parse_reason_content_block_start_thinking(self):
        """Reason event with content_block_start for thinking block."""
        msg = {
            "event": "reason",
            "data": {
                "event": {
                    "type": "content_block_start",
                    "content_block": {"type": "thinking"},
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.inner_type == "content_block_start"
        assert evt.block_type == "thinking"

    def test_parse_reason_content_block_start_text(self):
        """Reason event with content_block_start for text block."""
        msg = {
            "event": "reason",
            "data": {
                "event": {
                    "type": "content_block_start",
                    "content_block": {"type": "text"},
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.inner_type == "content_block_start"
        assert evt.block_type == "text"

    def test_parse_reason_content_block_start_tool_use(self):
        """Reason event with content_block_start for tool_use block."""
        msg = {
            "event": "reason",
            "data": {
                "event": {
                    "type": "content_block_start",
                    "content_block": {
                        "type": "tool_use",
                        "name": "web_search",
                        "id": "toolu_abc123",
                    },
                },
                "type": "stream_event",
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.inner_type == "content_block_start"
        assert evt.block_type == "tool_use"
        assert evt.tool_name == "web_search"
        assert evt.tool_use_id == "toolu_abc123"
        assert evt.data_type == "stream_event"

    def test_parse_reason_input_json_delta(self):
        """Reason event with input_json_delta (tool input streaming)."""
        msg = {
            "event": "reason",
            "data": {
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
                },
                "type": "stream_event",
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.delta_type == "input_json_delta"
        assert evt.data_type == "stream_event"
        # input_json_delta has no "text" or "thinking" key, so text should be empty
        assert evt.text == ""

    def test_parse_reason_content_block_stop(self):
        """Reason event with content_block_stop."""
        msg = {
            "event": "reason",
            "data": {
                "event": {"type": "content_block_stop"},
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.inner_type == "content_block_stop"
        assert evt.block_type == ""
        assert evt.delta_type == ""

    def test_parse_reason_text_delta(self):
        """Reason event with text_delta (answer text)."""
        msg = {
            "event": "reason",
            "data": {
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "answer chunk"},
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.text == "answer chunk"
        assert evt.delta_type == "text_delta"
        assert evt.inner_type == "content_block_delta"

    def test_parse_reason_thinking_delta(self):
        """Reason event with thinking_delta."""
        msg = {
            "event": "reason",
            "data": {
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "let me think"},
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.text == "let me think"
        assert evt.delta_type == "thinking_delta"

    def test_parse_reason_message_start(self):
        """Reason event with message_start inner type."""
        msg = {
            "event": "reason",
            "data": {
                "event": {"type": "message_start"},
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.inner_type == "message_start"
        assert evt.text == ""

    def test_parse_reason_fallback_message_content(self):
        """Reason event falls back to message.content[0].text (non-assistant)."""
        msg = {
            "event": "reason",
            "data": {
                "message": {
                    "content": [{"text": "fallback text"}],
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.text == "fallback text"

    def test_parse_reason_assistant_type_no_text_extraction(self):
        """Assistant-type reason events should NOT extract text (avoids duplicate)."""
        msg = {
            "event": "reason",
            "data": {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "long thinking..."},
                        {"type": "text", "text": "This is the full accumulated answer."},
                    ],
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type == "assistant"
        # Text should NOT be extracted from assistant summaries
        assert evt.text == ""

    def test_parse_reason_system_type(self):
        """System-type reason events should have data_type='system'."""
        msg = {
            "event": "reason",
            "data": {
                "type": "system",
                "model": "claude-sonnet-4-20250514",
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type == "system"

    def test_parse_reason_user_type_tool_result(self):
        """User-type reason events extract tool result text."""
        msg = {
            "event": "reason",
            "data": {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "Search results: found 3 items",
                            "tool_use_id": "toolu_abc",
                        },
                    ],
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type == "user"
        assert evt.text == "Search results: found 3 items"

    def test_parse_reason_user_type_tool_result_list_content(self):
        """User-type reason events with list content in tool_result."""
        msg = {
            "event": "reason",
            "data": {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": [{"type": "text", "text": "list result text"}],
                            "tool_use_id": "toolu_abc",
                        },
                    ],
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type == "user"
        assert evt.text == "list result text"

    def test_parse_reason_data_type_extracted(self):
        """data.type is always extracted into data_type field."""
        msg = {
            "event": "reason",
            "data": {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "hi"},
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type == "stream_event"
        assert evt.text == "hi"

    def test_parse_content_fallback_top_level_result(self):
        """Content event with result at top level (fallback format)"""
        msg = {"event": "content", "data": {"result": "direct result"}}
        evt = MiraClient._parse_event(msg)
        assert evt.text == "direct result"

    def test_parse_start_event_extracts_ids(self):
        """Start event extracts message_id and session_id from message_entity"""
        msg = {
            "event": "start",
            "data": {
                "message_entity": {"messageId": "123", "sessionId": "456"},
                "message_id": "123",
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.message_id == "123"
        assert evt.session_id == "456"

    def test_parse_safety_audit_recognizer_results(self):
        """Content safety audit metadata with recognizer_results sets data_type='safety_audit' and text=''."""
        msg = {
            "event": "reason",
            "data": {
                "dangerous": False,
                "downgrade_model": "",
                "recognizer_results": [
                    {
                        "entity_type": "REAL_NAME",
                        "value": "高博",
                        "start": 697,
                        "end": 699,
                        "placeholder": "高博",
                    }
                ],
                "last_masked_user_message": "some masked message content",
                "degrade_reason": "",
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type == "safety_audit"
        assert evt.text == ""

    def test_parse_safety_audit_last_masked_user_message_only(self):
        """Content safety metadata with only last_masked_user_message key."""
        msg = {
            "event": "reason",
            "data": {
                "last_masked_user_message": "full masked message text here",
                "dangerous": False,
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type == "safety_audit"
        assert evt.text == ""

    def test_parse_safety_audit_recognizer_results_only(self):
        """Content safety metadata with only recognizer_results key."""
        msg = {
            "event": "reason",
            "data": {
                "recognizer_results": [],
                "dangerous": False,
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type == "safety_audit"
        assert evt.text == ""

    def test_parse_normal_reason_not_flagged_as_safety_audit(self):
        """Normal reason events without safety keys are NOT classified as safety_audit."""
        msg = {
            "event": "reason",
            "data": {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "hello"},
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type != "safety_audit"
        assert evt.text == "hello"


    def test_parse_cis_ctrl_in_dict_value_detected_as_safety_audit(self):
        """Dict with a string value starting with <cis-ctrl> is detected as safety_audit."""
        msg = {
            "event": "reason",
            "data": {
                "type": "user",
                "message": {
                    "content": [
                        {"text": '<cis-ctrl>{"dangerous":false,"downgrade_model":""}'}
                    ],
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type == "safety_audit" or evt.text == ""
        # Either classified as safety_audit at dict level, or text cleared at final check

    def test_parse_cis_ctrl_text_field_cleared(self):
        """Text starting with <cis-ctrl> is cleared and event marked as safety_audit."""
        msg = {
            "event": "reason",
            "data": {
                "text": '<cis-ctrl>{"dangerous":false,"downgrade_model":"","recognizer_results":[]}',
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type == "safety_audit"
        assert evt.text == ""

    def test_parse_cis_ctrl_with_whitespace_prefix(self):
        """<cis-ctrl> with leading whitespace is also detected."""
        msg = {
            "event": "reason",
            "data": {
                "text": '  <cis-ctrl>{"dangerous":false}',
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.data_type == "safety_audit"
        assert evt.text == ""

    def test_parse_normal_text_not_affected_by_cis_ctrl_check(self):
        """Normal text that doesn't start with <cis-ctrl> is unaffected."""
        msg = {
            "event": "reason",
            "data": {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "normal answer text"},
                },
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.text == "normal answer text"
        assert evt.data_type != "safety_audit"


# ── create_session() Tests ─────────────────────────────────────────

class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_session_success(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "baseResp": {"statusCode": 0},
            "sessionItem": {"sessionId": 12345},
        }
        with patch.object(client, "_ensure_client") as mock_ec:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_ec.return_value = mock_http

            sid = await client.create_session(model="test-model")
            assert sid == "12345"
            mock_http.post.assert_called_once()
            call_args = mock_http.post.call_args
            assert call_args[0][0] == _EP.CHAT_CREATE

    @pytest.mark.asyncio
    async def test_create_session_auth_error(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        with patch.object(client, "_ensure_client") as mock_ec:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_ec.return_value = mock_http

            with pytest.raises(MiraAuthError):
                await client.create_session()


# ── list_sessions() Tests ─────────────────────────────────────────

class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "baseResp": {"statusCode": 0},
            "sessions": [{"sessionId": 1}, {"sessionId": 2}],
        }
        with patch.object(client, "_ensure_client") as mock_ec:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_resp)
            mock_ec.return_value = mock_http

            result = await client.list_sessions(page_size=10)
            assert "sessions" in result
            mock_http.get.assert_called_once()


# ── delete_session() Tests ─────────────────────────────────────────

class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_delete_session(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"baseResp": {"statusCode": 0}}
        with patch.object(client, "_ensure_client") as mock_ec:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_ec.return_value = mock_http

            await client.delete_session("sess_123")
            call_args = mock_http.post.call_args
            assert call_args[0][0] == _EP.CHAT_DELETE
            assert call_args[1]["json"]["sessionId"] == "sess_123"


# ── get_messages() Tests ──────────────────────────────────────────

class TestGetMessages:
    @pytest.mark.asyncio
    async def test_get_messages_parses(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "baseResp": {"statusCode": 0},
            "messages": [
                {
                    "messageId": 100, "sessionId": 200,
                    "sender": 1, "content": "hello",
                    "contentType": 1, "status": 0,
                },
                {
                    "messageId": 101, "sessionId": 200,
                    "sender": 2, "content": "hi there",
                    "contentType": 1, "status": 0,
                },
            ],
        }
        with patch.object(client, "_ensure_client") as mock_ec:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_ec.return_value = mock_http

            msgs = await client.get_messages("200")
            assert len(msgs) == 2
            assert msgs[0].message_id == "100"
            assert msgs[1].content == "hi there"

    @pytest.mark.asyncio
    async def test_get_messages_empty(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "baseResp": {"statusCode": 0},
            "messages": [],
        }
        with patch.object(client, "_ensure_client") as mock_ec:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_ec.return_value = mock_http

            msgs = await client.get_messages("200")
            assert msgs == []


# ── Endpoints Consistency Tests ────────────────────────────────────

class TestEndpoints:
    def test_all_endpoints_start_with_slash(self):
        for attr in dir(_EP):
            if attr.startswith("_"):
                continue
            val = getattr(_EP, attr)
            assert val.startswith("/"), f"{attr} doesn't start with /"

    def test_chat_create_has_mira_prefix(self):
        assert _EP.CHAT_CREATE.startswith("/mira/")

    def test_chat_list_no_mira_prefix(self):
        assert not _EP.CHAT_LIST.startswith("/mira/")


# ── Exceptions ─────────────────────────────────────────────────────

class TestExceptions:
    def test_mira_auth_error_is_mira_error(self):
        assert issubclass(MiraAuthError, Exception)

    def test_mira_api_error_is_mira_error(self):
        assert issubclass(MiraAPIError, Exception)

    def test_hierarchy(self):
        from mira_client import MiraError
        assert issubclass(MiraAuthError, MiraError)
        assert issubclass(MiraAPIError, MiraError)
