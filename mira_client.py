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
    """A single SSE event from Mira's streaming response."""
    event: str
    data: dict
    text: str = ""
    message_id: str = ""
    session_id: str = ""


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
        if event_type == "reason" and isinstance(raw_data, dict):
            text = raw_data.get("text", "")
        elif event_type == "content" and isinstance(raw_data, dict):
            text = raw_data.get("result", "")
        elif event_type == "title" and isinstance(raw_data, dict):
            text = raw_data.get("title", raw_data.get("text", ""))

        return MiraEvent(
            event=event_type,
            data=raw_data if isinstance(raw_data, dict) else {"raw": raw_data},
            text=text,
            message_id=str(raw_data.get("messageId", "")) if isinstance(raw_data, dict) else "",
            session_id=str(raw_data.get("sessionId", "")) if isinstance(raw_data, dict) else "",
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
