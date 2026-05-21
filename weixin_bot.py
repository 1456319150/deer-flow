"""WeixinBot — WeChat channel adapter for DeerFlow gateway.

Mirrors FeishuBot's role: receives WeChat messages via iLink long-poll,
routes them to ClaudeCodeBridge, and sends replies back as plain text.

Integration:
    In gateway.py main():
        weixin_bot = WeixinBot(cfg["weixin"], bridge)
        await weixin_bot.start()
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import shutil
import subprocess
import time
from urllib.parse import urlparse

from weixin_channel import (
    WeixinChannel,
    WeixinMedia,
    WeixinMessage,
    ContextTokenStore,
    SyncCursor,
    load_account,
    save_account,
    qr_login,
    strip_markdown,
)

# Import from gateway — these are the shared types
from gateway import ClaudeCodeBridge, StreamResult, UsageSummary, _preview_text

log = logging.getLogger("weixin_bot")


class WeixinBot:
    """WeChat Bot: iLink long-poll → ClaudeCodeBridge → text reply.

    Design decisions:
    - Plain text replies (WeChat doesn't render Markdown)
    - Session keyed by from_user (1:1 conversation = 1 session)
    - Typing indicator while Claude is processing
    - Auto-reconnect on transient failures
    - Re-login prompt on session expiry
    """

    MSG_MAX_AGE = 120     # Skip messages older than 2 min (offline replay)
    _INBOUND_DIR = "/tmp/deerflow_weixin_inbound"
    _MIRA_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

    def __init__(self, cfg: dict, bridge: ClaudeCodeBridge):
        self.bridge = bridge
        self._enabled = cfg.get("enabled", False)
        self._account_path = cfg.get("account_path", ".weixin-account.json")
        self._ctx_store_path = cfg.get("context_tokens_path", ".weixin-context-tokens.json")
        self._cursor_path = cfg.get("sync_cursor_path", ".weixin-sync-cursor.json")
        self._auto_login = cfg.get("auto_login", False)
        self._channel: WeixinChannel | None = None
        self._seen_msgs: set[str] = set()

    async def start(self) -> None:
        """Initialize and start the WeChat polling loop."""
        if not self._enabled:
            log.info("[WeixinBot] disabled in config (weixin.enabled=false)")
            return

        # Load or create account
        account = load_account(self._account_path)
        if not account:
            if self._auto_login:
                log.info("[WeixinBot] No saved account, starting QR login...")
                account = await qr_login()
                save_account(account, self._account_path)
            else:
                log.warning(
                    "[WeixinBot] No account found at %s. "
                    "Set weixin.auto_login=true or run: python -m weixin_login",
                    self._account_path,
                )
                return

        ctx_store = ContextTokenStore(self._ctx_store_path)
        sync_cursor = SyncCursor(self._cursor_path)
        self._channel = WeixinChannel(account, ctx_store, sync_cursor)

        # Start polling in background task
        asyncio.create_task(self._poll_loop())
        log.info("✅ WeixinBot started (baseurl=%s)", account.baseurl)

    async def _poll_loop(self) -> None:
        """Main message processing loop."""
        assert self._channel is not None

        try:
            async for msg in self._channel.poll():
                try:
                    await self._handle_message(msg)
                except Exception:
                    log.exception("[WeixinBot] Error handling message from %s", msg.from_user)
        except RuntimeError as e:
            if "expired" in str(e).lower():
                log.error("[WeixinBot] Session expired! Delete %s and restart to re-login.", self._account_path)
            else:
                log.exception("[WeixinBot] Fatal error in poll loop")

    async def _handle_message(self, msg: WeixinMessage) -> None:
        """Process a single inbound message."""
        # Dedup
        if msg.msg_id:
            if msg.msg_id in self._seen_msgs:
                return
            self._seen_msgs.add(msg.msg_id)
            if len(self._seen_msgs) > 10000:
                self._seen_msgs.clear()

        # Skip stale messages
        if msg.timestamp:
            age = time.time() - msg.timestamp
            if age > self.MSG_MAX_AGE:
                log.info("[WeixinBot] skip stale msg=%s age=%.0fs", msg.msg_id, age)
                return

        # Skip empty
        if msg.is_empty:
            return

        text = msg.text.strip()
        from_user = msg.from_user
        ctx_token = msg.context_token
        image_urls = list(getattr(msg, "image_urls", []) or [])
        image_media = list(getattr(msg, "image_media", []) or [])
        if not image_media and image_urls:
            image_media = [WeixinMedia(url=u) for u in image_urls]

        if not text and image_media:
            text = "[用户发送了图片]"

        topic_id = f"wx_{from_user}"
        log.info(
            "[WeixinBot] 收到消息 from=%s topic=%s text=%r images=%d",
            from_user, topic_id, text[:100], len(image_media),
        )

        control_reply = await self._handle_control_command(topic_id, text)
        if control_reply is not None:
            plain = strip_markdown(control_reply)
            log.info("[WeixinBot] 发送控制回复 to=%s len=%d preview=%r",
                     from_user, len(plain), _preview_text(plain))
            await self._channel.send_text(from_user, ctx_token, plain)
            return

        if ctx_token:
            await self._channel.send_typing(from_user, ctx_token, typing=True)

        # --- Download inbound images and route to Claude Code ---
        image_paths: list[str] = []
        all_temp_paths: list[str] = []
        try:
            if image_media:
                image_paths = await self._download_images(image_media, msg.msg_id or "msg")
                all_temp_paths.extend(image_paths)
            result = await self._process_with_streaming(
                topic_id,
                text,
                from_user,
                ctx_token,
                image_paths=image_paths or None,
            )
        except Exception as e:
            log.exception("[WeixinBot] Bridge error")
            result = StreamResult(assistant_texts=[f"处理出错: {e}"])
        finally:
            if all_temp_paths:
                self._cleanup_temp_files(all_temp_paths)

        # --- Send reply ---
        reply = self._format_reply(result)
        if reply:
            plain = strip_markdown(reply)
            log.info("[WeixinBot] 发送回复 to=%s len=%d preview=%r",
                     from_user, len(plain), _preview_text(plain))
            await self._channel.send_text(from_user, ctx_token, plain)
        else:
            await self._channel.send_text(
                from_user, ctx_token, "(Claude Code 已执行操作但未生成文字回复)")

        usage_text = self._format_usage_summary(result.usage)
        if usage_text:
            try:
                log.info("[WeixinBot] 发送统计 to=%s len=%d preview=%r",
                         from_user, len(usage_text), _preview_text(usage_text))
                await self._channel.send_text(from_user, ctx_token, usage_text)
            except Exception:
                log.exception("[WeixinBot] Failed to send usage summary to %s", from_user)

        # Cancel typing
        if ctx_token:
            await self._channel.send_typing(from_user, ctx_token, typing=False)

    async def _handle_control_command(self, topic_id: str, text: str) -> str | None:
        normalized = text.strip()
        if normalized not in {"/new", "/reset", "/session"}:
            return None
        if normalized in {"/new", "/reset"}:
            old_session = await self.bridge.reset_session(topic_id)
            if old_session:
                return f"已重置当前会话。\n旧 session: {old_session}\n下一条消息会开启新会话。"
            return "当前还没有活动会话。\n下一条消息会开启新会话。"
        session_id = self.bridge.get_session(topic_id)
        if session_id:
            return f"当前 session: {session_id}"
        return "当前还没有活动会话。"

    async def _process_with_streaming(
        self,
        topic_id: str,
        text: str,
        from_user: str,
        ctx_token: str,
        *,
        image_paths: list[str] | None = None,
        file_paths: list[str] | None = None,
    ) -> StreamResult:
        """Call Bridge with streaming, send intermediate typing keepalive."""
        result: StreamResult | None = None
        streamed_reply_chunks: list[str] = []
        last_typing_time = time.time()

        kwargs: dict[str, list[str]] = {}
        if image_paths:
            kwargs["image_paths"] = image_paths
        if file_paths:
            kwargs["file_paths"] = file_paths

        async for event in self.bridge.stream_ask(topic_id, text, **kwargs):
            if event["type"] == "final":
                result = event["result"]
            elif event["type"] == "stream_event":
                stream_event = event["event"]
                if stream_event.kind in {"result", "text"} and stream_event.text:
                    streamed_reply_chunks.append(stream_event.text)
                # Keepalive typing every 5s during processing
                now = time.time()
                if now - last_typing_time > 5 and ctx_token:
                    await self._channel.send_typing(from_user, ctx_token, typing=True)
                    last_typing_time = now

        result = result or StreamResult()
        if not result.reply_text and streamed_reply_chunks:
            result.result_text = "".join(streamed_reply_chunks)
        return result

    async def _download_images(self, image_media: list[WeixinMedia | str], msg_id: str) -> list[str]:
        assert self._channel is not None
        os.makedirs(self._INBOUND_DIR, exist_ok=True)
        paths: list[str] = []
        for idx, media in enumerate(image_media):
            if isinstance(media, str):
                media = WeixinMedia(url=media)
            url = media.url
            try:
                data, content_type = await self._channel.download_url(url)
                decrypted = self._decrypt_wechat_media(data, media.aes_key)
                if decrypted:
                    data = decrypted
                    log.info("[WeixinBot] decrypted WeChat media url_path=%s", urlparse(url).path)

                decoded = self._decode_wechat_xor_image(data)
                if decoded:
                    data, ext = decoded
                    log.info("[WeixinBot] decoded WeChat XOR image url=%s ext=%s", url, ext)
                else:
                    ext = self._guess_image_ext(data, url, content_type)
                    if ext == ".bin":
                        log.warning(
                            "[WeixinBot] unknown image format url_path=%s has_aes_key=%s content_type=%r head=%s",
                            urlparse(url).path, bool(media.aes_key), content_type, data[:24].hex(),
                        )
                path = os.path.join(self._INBOUND_DIR, f"{msg_id}_{idx}{ext}")
                with open(path, "wb") as f:
                    f.write(data)
                path = await asyncio.to_thread(self._ensure_supported_image_file, path, ext)
                paths.append(path)
                saved_size = os.path.getsize(path) if os.path.exists(path) else len(data)
                log.info("[WeixinBot] saved image %s (%d bytes)", path, saved_size)
            except Exception:
                log.warning("[WeixinBot] failed to download image url=%s", url, exc_info=True)
        return paths

    @staticmethod
    def _guess_image_ext(data: bytes, url: str = "", content_type: str = "") -> str:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if data[:2] == b"\xff\xd8":
            return ".jpg"
        if data[:4] == b"GIF8":
            return ".gif"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        if data[:2] == b"BM":
            return ".bmp"
        if data[:4] in {b"II*\x00", b"MM\x00*"}:
            return ".tiff"
        if len(data) >= 12 and data[4:8] == b"ftyp":
            brand = data[8:12].lower()
            compatible = data[8:64].lower()
            if brand in {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
                return ".heic"
            if b"heic" in compatible or b"heif" in compatible:
                return ".heic"
            if brand in {b"avif", b"avis"} or b"avif" in compatible:
                return ".avif"

        ctype = content_type.lower().split(";", 1)[0].strip()
        ctype_exts = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
            "image/heic": ".heic",
            "image/heif": ".heic",
            "image/avif": ".avif",
        }
        if ctype in ctype_exts:
            return ctype_exts[ctype]

        suffix = os.path.splitext(urlparse(url).path)[1].lower()
        known_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif"}
        return suffix if suffix in known_exts else ".bin"

    @classmethod
    def _decrypt_wechat_media(cls, data: bytes, aes_key: str = "") -> bytes | None:
        key = cls._parse_wechat_aes_key(aes_key)
        if not key:
            return None

        openssl = shutil.which("openssl")
        if not openssl:
            log.warning("[WeixinBot] cannot decrypt WeChat media: openssl not found")
            return None

        try:
            proc = subprocess.run(
                [openssl, "enc", "-d", "-aes-128-ecb", "-K", key.hex(), "-nosalt"],
                input=data,
                capture_output=True,
                check=True,
                timeout=20,
            )
            return proc.stdout
        except Exception:
            log.warning("[WeixinBot] failed to decrypt WeChat media", exc_info=True)
            return None

    @staticmethod
    def _parse_wechat_aes_key(value: str = "") -> bytes | None:
        value = (value or "").strip()
        if not value:
            return None

        try:
            raw_hex = bytes.fromhex(value)
            if len(raw_hex) == 16:
                return raw_hex
        except ValueError:
            pass

        padded = value + ("=" * (-len(value) % 4))
        decoded_variants: list[bytes] = []
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                decoded = decoder(padded)
            except (binascii.Error, ValueError):
                continue
            if decoded not in decoded_variants:
                decoded_variants.append(decoded)

        for decoded in decoded_variants:
            if len(decoded) == 16:
                return decoded
            try:
                decoded_hex = bytes.fromhex(decoded.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                continue
            if len(decoded_hex) == 16:
                return decoded_hex
        return None

    @classmethod
    def _decode_wechat_xor_image(cls, data: bytes) -> tuple[bytes, str] | None:
        """Decode WeChat-style XOR-obfuscated image bytes when present."""
        signatures = [
            (b"\xff\xd8\xff", ".jpg"),
            (b"\x89PNG\r\n\x1a\n", ".png"),
            (b"GIF8", ".gif"),
            (b"RIFF", ".webp"),
        ]
        for signature, ext in signatures:
            if len(data) < len(signature):
                continue
            key = data[0] ^ signature[0]
            if key == 0:
                continue
            if all((data[i] ^ key) == signature[i] for i in range(len(signature))):
                decoded = bytes(b ^ key for b in data)
                if ext != ".webp" or decoded[8:12] == b"WEBP":
                    return decoded, ext
        return None

    @classmethod
    def _ensure_supported_image_file(cls, path: str, ext: str) -> str:
        ext = ext.lower()
        if ext in cls._MIRA_IMAGE_EXTS:
            return path

        converted = os.path.splitext(path)[0] + ".jpg"
        if cls._convert_image_to_jpeg(path, converted):
            try:
                os.remove(path)
            except OSError:
                pass
            log.info("[WeixinBot] converted image %s -> %s for upload", path, converted)
            return converted

        log.warning("[WeixinBot] image format %s may be rejected by Mira upload: %s", ext, path)
        return path

    @staticmethod
    def _convert_image_to_jpeg(src: str, dst: str) -> bool:
        sips = shutil.which("sips")
        if not sips:
            return False
        try:
            subprocess.run(
                [sips, "-s", "format", "jpeg", src, "--out", dst],
                check=True,
                capture_output=True,
                timeout=20,
            )
            return os.path.exists(dst) and os.path.getsize(dst) > 0
        except Exception:
            try:
                if os.path.exists(dst):
                    os.remove(dst)
            except OSError:
                pass
            log.warning("[WeixinBot] failed to convert image for upload: %s", src, exc_info=True)
            return False

    @staticmethod
    def _cleanup_temp_files(paths: list[str]) -> None:
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass

    @classmethod
    def _format_reply(cls, result: StreamResult) -> str:
        """Format StreamResult for WeChat (plain text, no cards).

        Unlike Feishu which uses rich cards, WeChat gets a simpler format:
        - Tool calls summarized briefly
        - Thinking omitted (too verbose for mobile)
        - Focus on the actual reply text
        """
        sections: list[str] = []

        # Brief tool summary (collapsed)
        if result.tool_calls:
            tool_names = [tc.name for tc in result.tool_calls[:5]]
            summary = ", ".join(tool_names)
            remaining = len(result.tool_calls) - 5
            if remaining > 0:
                summary += f" 等{remaining + 5}个工具"
            sections.append(f"[执行了: {summary}]")

        # Main reply
        reply = result.reply_text
        if reply:
            sections.append(reply)

        return "\n\n".join(sections)

    @staticmethod
    def _format_usage_summary(usage: UsageSummary | None) -> str:
        if not usage or not usage.has_values:
            return ""

        lines = ["用量统计"]
        if usage.input_tokens is not None:
            lines.append(f"输入 tokens: {usage.input_tokens:,}")
        if usage.output_tokens is not None:
            lines.append(f"输出 tokens: {usage.output_tokens:,}")
        if usage.cache_creation_input_tokens is not None:
            lines.append(f"缓存写入 tokens: {usage.cache_creation_input_tokens:,}")
        if usage.cache_read_input_tokens is not None:
            lines.append(f"缓存命中 tokens: {usage.cache_read_input_tokens:,}")
        if usage.total_tokens is not None:
            lines.append(f"总 tokens: {usage.total_tokens:,}")
        if usage.cost_usd is not None:
            lines.append(f"金额: ${usage.cost_usd:.4f}")
        return "\n".join(lines)
