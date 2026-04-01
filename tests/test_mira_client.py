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

    def test_mira_event_full(self):
        evt = MiraEvent(
            event="content", data={"result": "hi"}, text="hi",
            message_id="m1", session_id="s1",
        )
        assert evt.event == "content"
        assert evt.text == "hi"

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
        """Reason event with thinking text in data.event.delta.text"""
        msg = {
            "event": "reason",
            "data": {
                "event": {"delta": {"text": "thinking step"}},
                "parent_tool_use_id": "",
                "session_id": "s1",
                "type": "content_block_delta",
                "uuid": "u1",
            },
        }
        evt = MiraClient._parse_event(msg)
        assert evt.text == "thinking step"

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
