"""mira_client.py — Mira Agent API SDK for deer-flow integration.

Wraps Mira's HTTP + SSE API into an async Python client.
Auth: mira_session JWT cookie (copy from browser DevTools).

Quickstart:
    from mira_client import MiraClient

    async with MiraClient("your_jwt_here") as client:
        reply = await client.ask("帮我分析这段代码")
        print(reply)

deer-flow integration:
    Used by MiraBridge (mira_bridge.py) — you don't call this directly.
    MiraBridge wraps this client to match ClaudeCodeBridge's interface.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import httpx

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────

class _EP:
    """Mira API endpoints. Note the inconsistent /mira prefix."""
    CHAT_CREATE     = "/mira/api/v1/chat/create"
    CHAT_COMPLETION = "/mira/api/v1/chat/completion"
    CHAT_MESSAGES   = "/mira/api/v1/chat/messages"
    FILE_UPLOAD     = "/mira/api/v1/file/upload"
    CHAT_LIST       = "/api/v1/chat/list"
    CHAT_RESUME     = "/api/v1/chat/completion/resume"
    CHAT_DELETE     = "/api/v1/chat/delete"


DEFAULT_MODEL = "re-o-46"
DEFAULT_BASE  = "https://mira.byteintl.net"

DEFAULT_TOOLS: list[dict[str, Any]] = [
    {"name": "KnowledgeQASearch", "id": 54604816403, "scope": "GLOBAL"},
    {"name": "Web",              "id": 54604802835, "scope": "GLOBAL"},
    {"name": "Lingo",            "id": 54604817683, "scope": "GLOBAL"},
    {"name": "ImageRich",        "id": 54604820243, "scope": "GLOBAL"},
]


# ── Data Models ────────────────────────────────────────────────────

@dataclass
class MiraEvent:
    """A single SSE event from Mira's streaming response.

    For reason events carrying stream deltas:
        block_type: "thinking" | "text" | "tool_use" | "" (from content_block_start)
        delta_type: "thinking_delta" | "text_delta" | "input_json_delta" | "" (from content_block_delta)
        inner_type: the raw event.type string (content_block_start/delta/stop, etc.)
        data_type:  "stream_event" | "assistant" | "user" | "system" | "" (from data.type)
        tool_name:  tool name from content_block_start with tool_use block
        tool_use_id: tool use ID from content_block_start with tool_use block
    """
    event: str
    data: dict
    text: str = ""
    message_id: str = ""
    session_id: str = ""
    block_type: str = ""      # "thinking" | "text" | "tool_use" from content_block_start
    delta_type: str = ""      # "thinking_delta" | "text_delta" | "input_json_delta" from delta
    inner_type: str = ""      # raw event.type (content_block_start/delta/stop/message_start...)
    data_type: str = ""       # "stream_event" | "assistant" | "user" | "system" | ""
    tool_name: str = ""       # tool name from content_block_start with tool_use
    tool_use_id: str = ""     # tool use ID from content_block_start with tool_use


@dataclass
class MiraMessage:
    """A message in a Mira chat session."""
    message_id: str
    session_id: str
    sender: int
    content: str
    content_type: int
    status: int


@dataclass
class FileInfo:
    """Uploaded file metadata, passable as attachment."""
    file_name: str
    url: str
    uri: str
    mime_type: str
    thumb_url: str = ""
    md5: str = ""

    def to_attachment(self) -> dict:
        return {
            "file_name": self.file_name,
            "url": self.url,
            "mime_type": self.mime_type,
            "uri": self.uri,
        }


# ── Exceptions ─────────────────────────────────────────────────────

class MiraError(Exception):
    pass

class MiraAuthError(MiraError):
    pass

class MiraAPIError(MiraError):
    pass


# ── Client ─────────────────────────────────────────────────────────

class MiraClient:
    def __init__(
        self,
        session_token: str,
        base_url: str = DEFAULT_BASE,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._token = session_token
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                cookies={"mira_session": self._token},
                timeout=httpx.Timeout(self._timeout, read=300.0),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "MiraClient":
        self._ensure_client()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ── Session Management ─────────────────────────────────────────

    async def create_session(
        self,
        model: str = DEFAULT_MODEL,
        data_sources: Optional[list[str]] = None,
    ) -> str:
        payload = {
            "sessionProperties": {
                "topic": "",
                "dataSource": "360_performance",
                "dataSources": data_sources or ["manus"],
                "model": model,
            }
        }
        resp = await self._ensure_client().post(_EP.CHAT_CREATE, json=payload)
        self._check(resp)
        sid = str(resp.json()["sessionItem"]["sessionId"])
        logger.info("Created Mira session %s", sid)
        return sid

    async def list_sessions(self, page_size: int = 50, page_number: int = 1) -> dict:
        resp = await self._ensure_client().get(
            _EP.CHAT_LIST,
            params={"pageSize": page_size, "pageNumber": page_number},
        )
        self._check(resp)
        return resp.json()

    async def delete_session(self, session_id: str) -> None:
        resp = await self._ensure_client().post(
            _EP.CHAT_DELETE, json={"sessionId": session_id}
        )
        self._check(resp)

    async def get_messages(self, session_id: str) -> list[MiraMessage]:
        resp = await self._ensure_client().post(
            _EP.CHAT_MESSAGES, json={"sessionId": session_id}
        )
        self._check(resp)
        return [
            MiraMessage(
                message_id=str(m.get("messageId", "")),
                session_id=str(m.get("sessionId", "")),
                sender=m.get("sender", 0),
                content=m.get("content", ""),
                content_type=m.get("contentType", 0),
                status=m.get("status", 0),
            )
            for m in resp.json().get("messages", [])
        ]

    # ── Chat Completion (SSE Streaming) ────────────────────────────

    async def chat(
        self,
        session_id: str,
        content: str,
        attachments: Optional[list[dict]] = None,
        model: str = DEFAULT_MODEL,
        tool_list: Optional[list[dict]] = None,
        mode: str = "quick",
    ) -> AsyncGenerator[MiraEvent, None]:
        payload: dict[str, Any] = {
            "sessionId": session_id,
            "content": content,
            "messageType": 1,
            "summaryAgent": model,
            "dataSources": ["manus"],
            "comprehensive": 1,
            "config": {
                "online": True,
                "mode": mode,
                "tool_list": tool_list or DEFAULT_TOOLS,
            },
        }
        if attachments:
            payload["attachments"] = attachments

        async for event in self._sse_stream(_EP.CHAT_COMPLETION, payload):
            yield event

    async def resume(self, message_id: str) -> AsyncGenerator[MiraEvent, None]:
        async for event in self._sse_stream(
            _EP.CHAT_RESUME, {"messageId": message_id}
        ):
            yield event

    # ── Convenience: ask → str ─────────────────────────────────────

    async def ask(
        self,
        content: str,
        session_id: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        **kwargs: Any,
    ) -> str:
        if session_id is None:
            session_id = await self.create_session(model=model)
        async for evt in self.chat(session_id, content, model=model, **kwargs):
            if evt.event == "content" and evt.text:
                return evt.text
        return ""

    # ── File Upload ────────────────────────────────────────────────

    async def upload_file(
        self,
        file_path: str | Path,
        sensitive_detection: bool = False,
    ) -> FileInfo:
        path = Path(file_path)
        async with httpx.AsyncClient(
            base_url=self.base_url,
            cookies={"mira_session": self._token},
            timeout=httpx.Timeout(60.0),
        ) as uc:
            with open(path, "rb") as f:
                resp = await uc.post(
                    _EP.FILE_UPLOAD,
                    params={"sensitive_detection": str(sensitive_detection).lower()},
                    files={"files": (path.name, f)},
                )
        self._check(resp)
        info = resp.json()["data"]["file_infos"][0]
        return FileInfo(
            file_name=info["file_name"],
            url=info["url"],
            uri=info["uri"],
            mime_type=info["mime_type"],
            thumb_url=info.get("thumb_url", ""),
            md5=info.get("md5", ""),
        )

    # ── SSE Parser (internal) ──────────────────────────────────────

    async def _sse_stream(
        self, endpoint: str, payload: dict
    ) -> AsyncGenerator[MiraEvent, None]:
        client = self._ensure_client()
        async with client.stream("POST", endpoint, json=payload) as resp:
            if resp.status_code == 401:
                raise MiraAuthError(
                    "mira_session expired. Copy fresh JWT from browser: "
                    "DevTools -> Application -> Cookies -> mira_session"
                )
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("data: "):
                    raw = line[6:]
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if parsed.get("done") is True:
                        return

                    msg_str = parsed.get("Message", "")
                    if not msg_str:
                        continue
                    try:
                        msg = json.loads(msg_str)
                    except json.JSONDecodeError:
                        continue

                    yield self._parse_event(msg)
                else:
                    try:
                        final = json.loads(line)
                        if final.get("code") == 0:
                            return
                    except json.JSONDecodeError:
                        continue

    @staticmethod
    def _parse_event(msg: dict) -> MiraEvent:
        event_type = msg.get("event", "unknown")
        raw_data = msg.get("data", {})

        if isinstance(raw_data, str):
            try:
                raw_data = json.loads(raw_data)
            except (json.JSONDecodeError, TypeError):
                raw_data = {"raw": raw_data}

        text = ""
        message_id = ""
        session_id = ""
        block_type = ""
        delta_type = ""
        inner_type = ""
        data_type = ""
        tool_name = ""
        tool_use_id = ""

        if isinstance(raw_data, dict):
            # Extract data.type for reason event subtype classification
            data_type = raw_data.get("type", "")

            # Skip content safety audit metadata (PII recognizer results)
            if "recognizer_results" in raw_data or "last_masked_user_message" in raw_data:
                data_type = "safety_audit"
                # Don't extract text from audit metadata

            # Also detect <cis-ctrl> wrapped safety audit events
            if not data_type:
                for v in raw_data.values():
                    if isinstance(v, str) and v.strip().startswith('<cis-ctrl>'):
                        data_type = "safety_audit"
                        break

            if event_type == "reason" and data_type != "safety_audit":
                # Extract inner event structure (Claude-style wrapped in Mira reason)
                evt_inner = raw_data.get("event", {})
                if isinstance(evt_inner, dict):
                    inner_type = evt_inner.get("type", "")

                    # content_block_start → extract block type (thinking/text/tool_use)
                    cb = evt_inner.get("content_block", {})
                    if isinstance(cb, dict) and cb:
                        block_type = cb.get("type", "")  # "thinking", "text", or "tool_use"
                        # For tool_use blocks, extract tool name and ID
                        if block_type == "tool_use":
                            tool_name = cb.get("name", "")
                            tool_use_id = cb.get("id", "")

                    # content_block_delta → extract delta text and type
                    delta = evt_inner.get("delta", {})
                    if isinstance(delta, dict) and delta:
                        delta_type = delta.get("type", "")  # "thinking_delta", "text_delta", "input_json_delta"
                        text = delta.get("text", delta.get("thinking", ""))

                # For "user" type events (tool results), extract text from message content
                if data_type == "user":
                    msg_inner = raw_data.get("message", {})
                    if isinstance(msg_inner, dict):
                        blocks = msg_inner.get("content", [])
                        if isinstance(blocks, list):
                            for blk in blocks:
                                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                                    # Extract text from tool_result content
                                    tr_content = blk.get("content", "")
                                    if isinstance(tr_content, str) and tr_content:
                                        text = tr_content
                                        break
                                    elif isinstance(tr_content, list):
                                        for item in tr_content:
                                            if isinstance(item, dict) and item.get("text"):
                                                text = item["text"]
                                                break
                                        if text:
                                            break
                                elif isinstance(blk, dict) and blk.get("text"):
                                    if not text:
                                        text = blk["text"]

                # Fallback: direct text field (only for stream_event types, not assistant summaries)
                if not text and data_type != "assistant":
                    text = raw_data.get("text", "")
                # Fallback: message.content[0].text (only for stream_event types)
                if not text and data_type != "assistant":
                    msg_inner = raw_data.get("message", {})
                    if isinstance(msg_inner, dict):
                        blocks = msg_inner.get("content", [])
                        if isinstance(blocks, list) and blocks:
                            text = blocks[0].get("text", "") if isinstance(blocks[0], dict) else ""

            elif event_type == "content":
                # Response text at: data.content.result (actual API)
                inner = raw_data.get("content", {})
                if isinstance(inner, dict) and inner:
                    text = inner.get("result", "")
                    session_id = str(inner.get("session_id", ""))
                if not text:
                    text = raw_data.get("result", "")

            elif event_type == "title":
                text = raw_data.get("content", raw_data.get("title", raw_data.get("text", "")))

            elif event_type == "start":
                me = raw_data.get("message_entity", {})
                if isinstance(me, dict):
                    message_id = str(me.get("messageId", ""))
                    session_id = str(me.get("sessionId", ""))
                message_id = message_id or str(raw_data.get("message_id", ""))

            # Fallback ID extraction
            if not message_id:
                message_id = str(raw_data.get("messageId", raw_data.get("message_id", "")))
            if not session_id:
                session_id = str(raw_data.get("sessionId", raw_data.get("session_id", "")))

        # Final safety: clear text if it contains <cis-ctrl> tags
        if text and text.strip().startswith('<cis-ctrl>'):
            data_type = "safety_audit"
            text = ""

        return MiraEvent(
            event=event_type,
            data=raw_data if isinstance(raw_data, dict) else {"raw": raw_data},
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

    @staticmethod
    def _check(resp: httpx.Response) -> None:
        if resp.status_code == 401:
            raise MiraAuthError("mira_session expired. Refresh from browser.")
        resp.raise_for_status()
        data = resp.json()
        base = data.get("baseResp", {})
        code = base.get("statusCode", data.get("code", 0))
        if code != 0:
            msg = base.get("statusMessage", data.get("msg", "unknown"))
            raise MiraAPIError(f"Mira API error [{code}]: {msg}")
