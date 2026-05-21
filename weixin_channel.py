"""WeChat iLink Channel — standalone protocol layer.

Pure HTTP/JSON implementation of Tencent's iLink bot protocol.
No dependency on OpenClaw. Can be used independently or as a channel
adapter for any Agent backend.

Protocol reference:
  - Base URL: https://ilinkai.weixin.qq.com
  - Auth: Bearer {bot_token} per request
  - Inbound: long-poll via /ilink/bot/getupdates (35s hold)
  - Outbound: POST /ilink/bot/sendmessage

Usage:
  channel = WeixinChannel(token="...", baseurl="...")
  async for msg in channel.poll():
      reply = await my_agent(msg.text)
      await channel.send_text(msg.from_user, msg.context_token, reply)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional
from urllib.parse import quote

import aiohttp

log = logging.getLogger("weixin")

ILINK_DEFAULT_BASE = "https://ilinkai.weixin.qq.com"
CHANNEL_VERSION = "2.0.1"
POLL_TIMEOUT = 40       # aiohttp timeout (> server's 35s hold)
RETRY_SHORT = 2         # seconds between retries
RETRY_LONG = 30         # seconds after consecutive failures
MAX_CONSECUTIVE_FAIL = 3
SESSION_EXPIRED_CODE = -14


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class WeixinMedia:
    """Parsed inbound media reference from WeChat."""
    url: str
    aes_key: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class WeixinMessage:
    """Parsed inbound message from WeChat."""
    from_user: str
    text: str
    context_token: str
    msg_id: str = ""
    timestamp: int = 0
    image_urls: list[str] = field(default_factory=list)
    image_media: list[WeixinMedia] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and not self.image_urls and not self.image_media


@dataclass
class WeixinAccount:
    """Persisted account credentials from QR login."""
    bot_token: str
    baseurl: str = ILINK_DEFAULT_BASE
    saved_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "bot_token": self.bot_token,
            "baseurl": self.baseurl,
            "saved_at": self.saved_at or time.time(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> WeixinAccount:
        return cls(
            bot_token=d["bot_token"],
            baseurl=d.get("baseurl", ILINK_DEFAULT_BASE),
            saved_at=d.get("saved_at", 0.0),
        )


# ===========================================================================
# Protocol helpers
# ===========================================================================

def _random_uin() -> str:
    """X-WECHAT-UIN: random uint32 → decimal string → base64."""
    uint32 = struct.unpack(">I", os.urandom(4))[0]
    return base64.b64encode(str(uint32).encode()).decode()


def _base_info() -> dict:
    return {"channel_version": CHANNEL_VERSION}


def _client_id() -> str:
    return f"bot-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


def _headers(token: Optional[str] = None) -> dict[str, str]:
    h: dict[str, str] = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "X-WECHAT-UIN": _random_uin(),
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _extract_text(msg: dict) -> str:
    """Extract plain text from iLink message item_list."""
    parts: list[str] = []
    for item in msg.get("item_list", []):
        if item.get("type") == 1 and item.get("text_item"):
            parts.append(item["text_item"].get("text", ""))
    return "".join(parts)


def _has_non_text_items(msg: dict) -> bool:
    """Return True for image/file/etc. messages mixed into item_list."""
    for item in msg.get("item_list", []):
        if not (item.get("type") == 1 and item.get("text_item")):
            return True
    return False


def _extract_image_media(msg: dict) -> list[WeixinMedia]:
    """Extract image media references from non-text iLink item payloads."""
    media_refs: list[WeixinMedia] = []
    seen_urls: set[str] = set()

    def collect(value: object) -> tuple[list[str], str]:
        urls: list[str] = []
        aes_key = ""
        if isinstance(value, dict):
            for k, v in value.items():
                key = str(k).lower()
                if isinstance(v, str) and v.startswith(("http://", "https://")):
                    if "url" in key or "uri" in key:
                        urls.append(v)
                elif isinstance(v, str) and key in {"encrypt_query_param", "encrypted_query_param"}:
                    urls.append(
                        "https://novac2c.cdn.weixin.qq.com/c2c/download"
                        f"?encrypted_query_param={quote(v, safe='')}"
                    )
                elif isinstance(v, str) and "aes" in key and "key" in key:
                    aes_key = aes_key or v
                else:
                    child_urls, child_key = collect(v)
                    urls.extend(child_urls)
                    aes_key = aes_key or child_key
        elif isinstance(value, list):
            for item in value:
                child_urls, child_key = collect(item)
                urls.extend(child_urls)
                aes_key = aes_key or child_key
        return urls, aes_key

    for item in msg.get("item_list", []):
        if item.get("type") == 1 and item.get("text_item"):
            continue
        urls, aes_key = collect(item)
        raw = item if isinstance(item, dict) else {}
        for url in urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            media_refs.append(WeixinMedia(url=url, aes_key=aes_key, raw=raw))
    return media_refs


def _extract_image_urls(msg: dict) -> list[str]:
    """Extract image URLs from non-text iLink item payloads."""
    return [m.url for m in _extract_image_media(msg)]


def strip_markdown(text: str) -> str:
    """Convert Markdown to WeChat-friendly plain text.

    WeChat doesn't render Markdown, so we strip formatting while preserving
    readability. Keeps code blocks, converts headers to plain text with
    newlines, removes bold/italic markers.
    """
    # Fenced code blocks → keep content, remove fence markers
    text = re.sub(r"```\w*\n", "```\n", text)
    # Headers → plain text with separator
    text = re.sub(r"^#{1,6}\s+(.+)$", r"【\1】", text, flags=re.MULTILINE)
    # Bold/italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # Inline code — keep backticks for readability
    # Links → text (url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    # Horizontal rules
    text = re.sub(r"^---+$", "————————", text, flags=re.MULTILINE)
    # Blockquotes
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    return text.strip()


# ===========================================================================
# QR Login
# ===========================================================================

async def qr_login(baseurl: str = ILINK_DEFAULT_BASE) -> WeixinAccount:
    """Interactive QR code login. Returns WeixinAccount with bot_token.

    Prints QR code URL to terminal. In production, render it as an image
    or forward to a web UI.
    """
    async with aiohttp.ClientSession() as session:
        # 1. Get QR code
        async with session.get(
            f"{baseurl}/ilink/bot/get_bot_qrcode?bot_type=3",
            headers=_headers(),
        ) as resp:
            data = await resp.json(content_type=None)

        qr_url = data.get("qrcode_img_content", "")
        qr_key = data.get("qrcode", "")

        if not qr_url or not qr_key:
            raise RuntimeError(f"Failed to get QR code: {data}")

        log.info("=== 微信扫码登录 ===")
        log.info("请用微信扫描以下链接中的二维码:")
        log.info(qr_url)

        # Try to render QR in terminal (optional dependency)
        try:
            import qrcode as qr_lib
            qr = qr_lib.QRCode(box_size=1, border=1)
            qr.add_data(qr_url)
            qr.print_ascii(invert=True)
        except ImportError:
            log.info("(安装 qrcode 库可在终端显示二维码: pip install qrcode)")

        # 2. Poll for scan confirmation
        max_refreshes = 3
        refreshes = 0
        while True:
            await asyncio.sleep(2)
            async with session.get(
                f"{baseurl}/ilink/bot/get_qrcode_status?qrcode={qr_key}",
                headers=_headers(),
            ) as resp:
                status_data = await resp.json(content_type=None)

            if status_data.get("bot_token"):
                log.info("✅ 微信登录成功!")
                return WeixinAccount(
                    bot_token=status_data["bot_token"],
                    baseurl=status_data.get("baseurl", baseurl),
                    saved_at=time.time(),
                )

            status = status_data.get("status", "")
            if status == "expired":
                refreshes += 1
                if refreshes >= max_refreshes:
                    raise RuntimeError("二维码已过期且刷新次数超限")
                log.info("二维码过期，自动刷新中...")
                async with session.get(
                    f"{baseurl}/ilink/bot/get_bot_qrcode?bot_type=3",
                    headers=_headers(),
                ) as resp:
                    data = await resp.json(content_type=None)
                qr_key = data.get("qrcode", "")
                qr_url = data.get("qrcode_img_content", "")
                log.info("新二维码: %s", qr_url)
            elif status == "scaned":
                log.info("已扫码，等待确认...")


# ===========================================================================
# Credential persistence
# ===========================================================================

def save_account(account: WeixinAccount, path: str = ".weixin-account.json") -> None:
    """Save account credentials to disk."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(account.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    log.info("Account saved to %s", path)


def load_account(path: str = ".weixin-account.json") -> Optional[WeixinAccount]:
    """Load account from disk, or return None."""
    try:
        with open(path, encoding="utf-8") as f:
            return WeixinAccount.from_dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


# ===========================================================================
# Context token store
# ===========================================================================

class ContextTokenStore:
    """Per-user context_token persistence.

    context_token is required for replies — without it, messages may not
    reach the correct conversation context on WeChat's side.
    """

    def __init__(self, path: str = ".weixin-context-tokens.json"):
        self._path = path
        self._tokens: dict[str, str] = self._load()

    def get(self, user_id: str) -> Optional[str]:
        return self._tokens.get(user_id)

    def set(self, user_id: str, token: str) -> None:
        if self._tokens.get(user_id) == token:
            return
        self._tokens[user_id] = token
        self._save()

    def _load(self) -> dict[str, str]:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            return {str(k): str(v) for k, v in data.items() if k and v}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self) -> None:
        tmp = f"{self._path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._tokens, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)


# ===========================================================================
# Sync cursor persistence
# ===========================================================================

class SyncCursor:
    """Persists get_updates_buf for crash recovery."""

    def __init__(self, path: str = ".weixin-sync-cursor.json"):
        self._path = path
        self._buf = self._load()

    @property
    def buf(self) -> str:
        return self._buf

    def update(self, new_buf: str) -> None:
        if new_buf and new_buf != self._buf:
            self._buf = new_buf
            self._save()

    def _load(self) -> str:
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("get_updates_buf", "")
        except (FileNotFoundError, json.JSONDecodeError):
            return ""

    def _save(self) -> None:
        tmp = f"{self._path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"get_updates_buf": self._buf, "updated_at": time.time()}, f)
        os.replace(tmp, self._path)


# ===========================================================================
# WeixinChannel — core protocol implementation
# ===========================================================================

class WeixinChannel:
    """iLink protocol client: poll, send, typing."""

    def __init__(
        self,
        account: WeixinAccount,
        ctx_store: Optional[ContextTokenStore] = None,
        sync_cursor: Optional[SyncCursor] = None,
    ):
        self.account = account
        self.ctx_store = ctx_store or ContextTokenStore()
        self.sync_cursor = sync_cursor or SyncCursor()
        self._session: Optional[aiohttp.ClientSession] = None
        self._consecutive_fails = 0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=POLL_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # --- Inbound: long-poll messages ---

    async def poll(self) -> AsyncIterator[WeixinMessage]:
        """Infinite generator: yields inbound messages from WeChat.

        Handles retries, backoff, session expiry. Caller should wrap in
        try/except for RuntimeError (session expired → need re-login).
        """
        while True:
            try:
                msgs = await self._get_updates()
                self._consecutive_fails = 0
                for msg in msgs:
                    yield msg
            except _SessionExpired:
                log.error("Session expired (errcode=%d), need re-login", SESSION_EXPIRED_CODE)
                raise RuntimeError("WeChat session expired, please re-login")
            except asyncio.TimeoutError:
                # Normal: long-poll timeout with no messages
                self._consecutive_fails = 0
                continue
            except Exception as e:
                self._consecutive_fails += 1
                if self._consecutive_fails >= MAX_CONSECUTIVE_FAIL:
                    log.warning("连续失败 %d 次, 退避 %ds: %s",
                                self._consecutive_fails, RETRY_LONG, e)
                    await asyncio.sleep(RETRY_LONG)
                else:
                    log.warning("轮询失败 (%d/%d): %s",
                                self._consecutive_fails, MAX_CONSECUTIVE_FAIL, e)
                    await asyncio.sleep(RETRY_SHORT)

    async def _get_updates(self) -> list[WeixinMessage]:
        """Single long-poll call to iLink."""
        session = await self._ensure_session()
        base = self.account.baseurl

        async with session.post(
            f"{base}/ilink/bot/getupdates",
            json={"get_updates_buf": self.sync_cursor.buf, "base_info": _base_info()},
            headers=_headers(self.account.bot_token),
        ) as resp:
            data = await resp.json(content_type=None)

        # Check for session expiry
        errcode = data.get("errcode", 0)
        if errcode == SESSION_EXPIRED_CODE:
            raise _SessionExpired()

        # Update sync cursor
        new_buf = data.get("get_updates_buf", "")
        self.sync_cursor.update(new_buf)

        # Parse messages
        messages: list[WeixinMessage] = []
        for msg in data.get("msgs", []):
            # Skip bot's own messages (message_type=2)
            if msg.get("message_type") == 2:
                continue

            from_user = msg.get("from_user_id", "")
            raw_context_token = msg.get("context_token", "")
            text = _extract_text(msg)
            has_non_text_items = _has_non_text_items(msg)
            image_media = _extract_image_media(msg)
            image_urls = [m.url for m in image_media]
            context_token = raw_context_token

            # Image/file messages may carry a one-off context_token that opens a
            # separate WeChat context. Prefer the user's existing token so replies
            # stay in the active conversation; use the inbound token only as fallback.
            if from_user and raw_context_token:
                if has_non_text_items:
                    stored = self.ctx_store.get(from_user)
                    if stored:
                        context_token = stored
                        log.info("Using stored context_token for non-text message from %s", from_user)
                    else:
                        self.ctx_store.set(from_user, raw_context_token)
                else:
                    self.ctx_store.set(from_user, raw_context_token)

            messages.append(WeixinMessage(
                from_user=from_user,
                text=text,
                context_token=context_token,
                msg_id=msg.get("msg_id", ""),
                timestamp=msg.get("timestamp", 0),
                image_urls=image_urls,
                image_media=image_media,
                raw=msg,
            ))

        return messages

    # --- Outbound: send messages ---

    async def send_text(self, to: str, context_token: str, text: str) -> dict:
        """Send a text message to a WeChat user.

        Args:
            to: WeChat user ID (from_user_id from inbound message)
            context_token: must match the conversation context
            text: plain text content (WeChat doesn't render Markdown)
        """
        if not context_token:
            # Try to recover from store
            stored = self.ctx_store.get(to)
            if stored:
                context_token = stored
                log.info("Recovered context_token from store for %s", to)
            else:
                log.warning("No context_token for %s, send may fail", to)

        session = await self._ensure_session()
        base = self.account.baseurl

        # WeChat has a message length limit; split if necessary
        chunks = self._split_text(text, max_len=4000)
        last_result = {}

        for chunk in chunks:
            async with session.post(
                f"{base}/ilink/bot/sendmessage",
                json={
                    "msg": {
                        "from_user_id": "",
                        "to_user_id": to,
                        "client_id": _client_id(),
                        "message_type": 2,      # BOT
                        "message_state": 2,      # FINISH
                        "context_token": context_token,
                        "item_list": [{"type": 1, "text_item": {"text": chunk}}],
                    },
                    "base_info": _base_info(),
                },
                headers=_headers(self.account.bot_token),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                last_result = await resp.json(content_type=None)

            if len(chunks) > 1:
                await asyncio.sleep(0.5)  # Rate limit between chunks

        return last_result

    async def download_url(self, url: str) -> tuple[bytes, str]:
        """Download an inbound media URL using the same iLink auth context."""
        session = await self._ensure_session()
        async with session.get(
            url,
            headers=_headers(self.account.bot_token),
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.read()
            return data, resp.headers.get("Content-Type", "")

    # --- Typing indicator ---

    async def send_typing(self, user_id: str, context_token: str, typing: bool = True) -> None:
        """Show/hide 'typing...' indicator in WeChat."""
        try:
            session = await self._ensure_session()
            base = self.account.baseurl

            # Get typing ticket
            async with session.post(
                f"{base}/ilink/bot/getconfig",
                json={
                    "ilink_user_id": user_id,
                    "context_token": context_token,
                    "base_info": _base_info(),
                },
                headers=_headers(self.account.bot_token),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                config = await resp.json(content_type=None)

            ticket = config.get("typing_ticket")
            if not ticket:
                return

            async with session.post(
                f"{base}/ilink/bot/sendtyping",
                json={
                    "ilink_user_id": user_id,
                    "typing_ticket": ticket,
                    "status": 1 if typing else 2,
                    "base_info": _base_info(),
                },
                headers=_headers(self.account.bot_token),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                await resp.json(content_type=None)
        except Exception:
            log.debug("Typing indicator failed", exc_info=True)

    # --- Helpers ---

    @staticmethod
    def _split_text(text: str, max_len: int = 4000) -> list[str]:
        """Split long text into chunks, preferring paragraph boundaries."""
        if len(text) <= max_len:
            return [text]

        chunks: list[str] = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break

            # Find a good split point
            split_at = max_len
            # Prefer paragraph break
            para_break = text.rfind("\n\n", 0, max_len)
            if para_break > max_len // 2:
                split_at = para_break + 2
            else:
                # Fallback to line break
                line_break = text.rfind("\n", 0, max_len)
                if line_break > max_len // 2:
                    split_at = line_break + 1

            chunks.append(text[:split_at])
            text = text[split_at:]

        return chunks


class _SessionExpired(Exception):
    pass
