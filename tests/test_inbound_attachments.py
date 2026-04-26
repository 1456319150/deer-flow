"""Tests for inbound attachment (image/file) handling across all three layers:

1. feishu_bot.py — message parsing, download helpers, temp file management
2. gateway.py   — CLI command building with --images/--image flags
3. mira_bridge.py — file upload and attachment passing to Mira API
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Ensure project root on sys.path (same as conftest.py)
# ---------------------------------------------------------------------------
import sys

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gateway import ClaudeCodeBridge, StreamResult, StreamEvent


# ===========================================================================
# 1. FeishuBot — _guess_image_ext
# ===========================================================================

class TestGuessImageExt:
    """Test FeishuBot._guess_image_ext static method."""

    @pytest.fixture(autouse=True)
    def _import_bot(self):
        from feishu_bot import FeishuBot
        self.FeishuBot = FeishuBot

    def test_png(self):
        data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        assert self.FeishuBot._guess_image_ext(data) == ".png"

    def test_jpeg(self):
        data = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        assert self.FeishuBot._guess_image_ext(data) == ".jpg"

    def test_gif(self):
        data = b'GIF89a' + b'\x00' * 100
        assert self.FeishuBot._guess_image_ext(data) == ".gif"

    def test_webp(self):
        data = b'RIFF\x00\x00\x00\x00WEBP' + b'\x00' * 100
        assert self.FeishuBot._guess_image_ext(data) == ".webp"

    def test_unknown_fallback(self):
        data = b'\x00\x01\x02\x03' + b'\x00' * 100
        assert self.FeishuBot._guess_image_ext(data) == ".bin"

    def test_empty_data(self):
        assert self.FeishuBot._guess_image_ext(b'') == ".bin"

    def test_short_data(self):
        assert self.FeishuBot._guess_image_ext(b'\xff\xd8') == ".jpg"


# ===========================================================================
# 2. FeishuBot — _cleanup_inbound_attachments
# ===========================================================================

class TestCleanupInboundAttachments:
    """Test temp file cleanup."""

    @pytest.fixture(autouse=True)
    def _import_bot(self):
        from feishu_bot import FeishuBot
        self.FeishuBot = FeishuBot

    def test_cleanup_removes_files(self, tmp_path):
        f1 = tmp_path / "img1.png"
        f2 = tmp_path / "doc.pdf"
        f1.write_bytes(b"fake png")
        f2.write_bytes(b"fake pdf")
        assert f1.exists() and f2.exists()

        self.FeishuBot._cleanup_inbound_attachments([str(f1), str(f2)])
        assert not f1.exists()
        assert not f2.exists()

    def test_cleanup_ignores_missing(self, tmp_path):
        """Should not raise on non-existent files."""
        self.FeishuBot._cleanup_inbound_attachments([
            str(tmp_path / "nonexistent1"),
            str(tmp_path / "nonexistent2"),
        ])

    def test_cleanup_empty_list(self):
        """Should not raise on empty list."""
        self.FeishuBot._cleanup_inbound_attachments([])


# ===========================================================================
# 3. FeishuBot — _on_message attachment_meta construction
# ===========================================================================

class TestOnMessageAttachmentMeta:
    """Test that _on_message correctly builds attachment_meta for different msg types."""

    @pytest.fixture
    def bot(self):
        from feishu_bot import FeishuBot
        bot = object.__new__(FeishuBot)
        bot.app_id = "test_id"
        bot.app_secret = "test_secret"
        bot.bridge = MagicMock()
        bot._api_client = MagicMock()
        bot._main_loop = MagicMock()
        bot._main_loop.is_running.return_value = True
        bot._seen_msgs = set()
        bot._image_cache = {}
        bot._sdk = {}
        bot._lark = MagicMock()
        return bot

    def _make_event(self, msg_type, content, msg_id="msg_001",
                    chat_id="chat_001", create_time=None):
        import time as _time
        msg = SimpleNamespace(
            chat_id=chat_id,
            message_id=msg_id,
            root_id=None,
            message_type=msg_type,
            content=json.dumps(content),
            create_time=create_time or str(int(_time.time() * 1000)),
        )
        return SimpleNamespace(event=SimpleNamespace(message=msg))

    def test_image_message_builds_attachment_meta(self, bot):
        captured_args = {}
        def mock_run_coroutine(coro, loop):
            captured_args['coro'] = coro
            fut = MagicMock()
            fut.add_done_callback = MagicMock()
            return fut

        with patch("asyncio.run_coroutine_threadsafe", side_effect=mock_run_coroutine):
            event = self._make_event("image", {"image_key": "img_v3_abc123"})
            bot._on_message(event)

        assert 'coro' in captured_args
        coro = captured_args['coro']
        coro.close()

    def test_file_message_builds_attachment_meta(self, bot):
        captured = {}
        def mock_run(coro, loop):
            captured['coro'] = coro
            fut = MagicMock()
            fut.add_done_callback = MagicMock()
            return fut

        with patch("asyncio.run_coroutine_threadsafe", side_effect=mock_run):
            event = self._make_event("file", {"file_key": "file_abc", "file_name": "report.pdf"})
            bot._on_message(event)

        assert 'coro' in captured
        captured['coro'].close()

    def test_audio_message_unsupported(self, bot):
        captured = {}
        def mock_run(coro, loop):
            captured['coro'] = coro
            fut = MagicMock()
            fut.add_done_callback = MagicMock()
            return fut

        with patch("asyncio.run_coroutine_threadsafe", side_effect=mock_run):
            event = self._make_event("audio", {"duration": 5000})
            bot._on_message(event)

        assert 'coro' in captured
        captured['coro'].close()

    def test_text_message_no_attachment(self, bot):
        captured = {}
        def mock_run(coro, loop):
            captured['coro'] = coro
            fut = MagicMock()
            fut.add_done_callback = MagicMock()
            return fut

        with patch("asyncio.run_coroutine_threadsafe", side_effect=mock_run):
            event = self._make_event("text", {"text": "hello world"})
            bot._on_message(event)

        assert 'coro' in captured
        captured['coro'].close()


# ===========================================================================
# 4. Gateway — _build_cmd_claude with images/files
# ===========================================================================

class TestBuildCmdClaudeAttachments:
    """Test ClaudeCodeBridge._build_cmd_claude with image_paths and file_paths."""

    @pytest.fixture
    def bridge(self):
        cfg = {"aiden_cmd": "aiden", "model": "test-model", "target": "claude"}
        return ClaudeCodeBridge(cfg)

    def test_no_attachments(self, bridge):
        cmd = bridge._build_cmd_claude("hello", None)
        args_str = " ".join(cmd)
        assert "--images" not in args_str
        assert "[附件:" not in args_str

    def test_single_image(self, bridge):
        cmd = bridge._build_cmd_claude("hello", None, image_paths=["/tmp/img.png"])
        args_str = " ".join(cmd)
        assert "--images /tmp/img.png" in args_str

    def test_multiple_images(self, bridge):
        cmd = bridge._build_cmd_claude("hello", None, image_paths=["/tmp/a.png", "/tmp/b.jpg"])
        args_str = " ".join(cmd)
        assert "--images /tmp/a.png" in args_str
        assert "--images /tmp/b.jpg" in args_str

    def test_single_file(self, bridge):
        cmd = bridge._build_cmd_claude("analyze this", None, file_paths=["/tmp/doc.pdf"])
        args_str = " ".join(cmd)
        assert "[附件: /tmp/doc.pdf]" in args_str
        assert "analyze this" in args_str

    def test_multiple_files(self, bridge):
        cmd = bridge._build_cmd_claude("read these", None, file_paths=["/tmp/a.pdf", "/tmp/b.docx"])
        args_str = " ".join(cmd)
        assert "[附件: /tmp/a.pdf]" in args_str
        assert "[附件: /tmp/b.docx]" in args_str

    def test_images_and_files_combined(self, bridge):
        cmd = bridge._build_cmd_claude(
            "check this", None,
            image_paths=["/tmp/screenshot.png"],
            file_paths=["/tmp/report.pdf"],
        )
        args_str = " ".join(cmd)
        assert "--images /tmp/screenshot.png" in args_str
        assert "[附件: /tmp/report.pdf]" in args_str

    def test_images_with_session_resume(self, bridge):
        cmd = bridge._build_cmd_claude("hello", "session_123", image_paths=["/tmp/img.png"])
        args_str = " ".join(cmd)
        assert "--resume session_123" in args_str
        assert "--images /tmp/img.png" in args_str

    def test_none_image_paths_ignored(self, bridge):
        cmd = bridge._build_cmd_claude("hello", None, image_paths=None)
        args_str = " ".join(cmd)
        assert "--images" not in args_str

    def test_empty_image_paths_ignored(self, bridge):
        cmd = bridge._build_cmd_claude("hello", None, image_paths=[])
        args_str = " ".join(cmd)
        assert "--images" not in args_str


# ===========================================================================
# 5. Gateway — _build_cmd_codex with images/files
# ===========================================================================

class TestBuildCmdCodexAttachments:
    """Test ClaudeCodeBridge._build_cmd_codex with image_paths and file_paths."""

    @pytest.fixture
    def bridge(self):
        cfg = {"aiden_cmd": "aiden", "model": "test-model", "target": "codex", "provider": "codex"}
        return ClaudeCodeBridge(cfg)

    def test_no_attachments(self, bridge):
        cmd = bridge._build_cmd_codex("hello", None)
        args_str = " ".join(cmd)
        assert "--image" not in args_str

    def test_single_image_uses_singular_flag(self, bridge):
        cmd = bridge._build_cmd_codex("hello", None, image_paths=["/tmp/img.png"])
        args_str = " ".join(cmd)
        assert "--image /tmp/img.png" in args_str
        assert "--images" not in args_str

    def test_multiple_images(self, bridge):
        cmd = bridge._build_cmd_codex("hello", None, image_paths=["/tmp/a.png", "/tmp/b.jpg"])
        args_str = " ".join(cmd)
        assert "--image /tmp/a.png" in args_str
        assert "--image /tmp/b.jpg" in args_str

    def test_file_prepend(self, bridge):
        cmd = bridge._build_cmd_codex("analyze", None, file_paths=["/tmp/doc.pdf"])
        args_str = " ".join(cmd)
        assert "[附件: /tmp/doc.pdf]" in args_str

    def test_images_with_session_resume(self, bridge):
        cmd = bridge._build_cmd_codex("hello", "sess_abc", image_paths=["/tmp/img.png"])
        args_str = " ".join(cmd)
        assert "resume" in args_str
        assert "--image /tmp/img.png" in args_str


# ===========================================================================
# 6. Gateway — _build_cmd dispatch
# ===========================================================================

class TestBuildCmdDispatch:
    def test_claude_provider_dispatches(self):
        bridge = ClaudeCodeBridge({"aiden_cmd": "aiden", "model": "m", "target": "claude"})
        cmd = bridge._build_cmd("hi", None, image_paths=["/tmp/x.png"])
        args_str = " ".join(cmd)
        assert "--images /tmp/x.png" in args_str

    def test_codex_provider_dispatches(self):
        bridge = ClaudeCodeBridge({"aiden_cmd": "aiden", "model": "m", "target": "codex", "provider": "codex"})
        cmd = bridge._build_cmd("hi", None, image_paths=["/tmp/x.png"])
        args_str = " ".join(cmd)
        assert "--image /tmp/x.png" in args_str
        assert "--images" not in args_str


# ===========================================================================
# 7. Gateway — stream_ask only passes images on first attempt
# ===========================================================================

class TestStreamAskRetryImagePaths(unittest.IsolatedAsyncioTestCase):

    def _make_bridge(self):
        cfg = {"aiden_cmd": "echo", "model": "m", "target": "claude"}
        return ClaudeCodeBridge(cfg)

    async def test_images_only_on_first_attempt(self):
        bridge = self._make_bridge()
        call_count = 0
        received_kwargs = []

        async def mock_stream_ask_once(topic_id, prompt, *, image_paths=None, file_paths=None):
            nonlocal call_count
            call_count += 1
            received_kwargs.append({"image_paths": image_paths, "file_paths": file_paths})
            if call_count == 1:
                result = StreamResult(stop_reason="tool_use", session_id="sess_1")
                yield {"type": "final", "result": result}
            else:
                result = StreamResult(result_text="done", stop_reason="end_turn", session_id="sess_1")
                yield {"type": "final", "result": result}

        bridge._stream_ask_once = mock_stream_ask_once

        events = []
        async for event in bridge.stream_ask("topic_1", "hello", image_paths=["/tmp/img.png"]):
            events.append(event)

        assert call_count == 2
        assert received_kwargs[0]["image_paths"] == ["/tmp/img.png"]
        assert received_kwargs[1]["image_paths"] is None


# ===========================================================================
# 8. MiraBridge — stream_ask uploads attachments
# ===========================================================================

class TestMiraBridgeAttachments(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self._tmp_dir)

        client = AsyncMock()
        client.create_session = AsyncMock(return_value="session_123")

        from mira_client import FileInfo
        async def mock_upload(path):
            name = os.path.basename(path)
            return FileInfo(
                file_name=name,
                url=f"https://mira.example.com/files/{name}",
                uri=f"mira://files/{name}",
                mime_type="application/octet-stream",
            )
        client.upload_file = AsyncMock(side_effect=mock_upload)

        async def mock_chat(session_id, content, *, model=None, mode=None, attachments=None):
            from mira_client import MiraEvent
            mock_chat.last_attachments = attachments
            evt = MiraEvent(
                event="content",
                data={"content": {"result": "I see the image", "input_tokens": 100, "output_tokens": 50}},
                text="I see the image",
            )
            yield evt
        mock_chat.last_attachments = None
        client.chat = mock_chat

        self.mock_client = client

        from mira_bridge import MiraBridge
        b = object.__new__(MiraBridge)
        b.client = client
        b.model = "test-model"
        b.mode = "quick"
        b.timeout = 30
        b.session_store_path = "/tmp/test-mira-sessions.json"
        b._session_lock = asyncio.Lock()
        b._sessions = {}
        self.bridge = b

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    async def test_image_upload_and_pass(self):
        img_path = self.tmp_path / "test.png"
        img_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

        events = []
        async for event in self.bridge.stream_ask("topic_1", "what is this?", image_paths=[str(img_path)]):
            events.append(event)

        self.mock_client.upload_file.assert_called_once_with(str(img_path))
        attachments = self.mock_client.chat.last_attachments
        assert attachments is not None
        assert len(attachments) == 1
        assert attachments[0]["file_name"] == "test.png"

    async def test_file_upload_and_pass(self):
        doc_path = self.tmp_path / "report.pdf"
        doc_path.write_bytes(b'%PDF-1.4 fake content')

        events = []
        async for event in self.bridge.stream_ask("topic_1", "summarize", file_paths=[str(doc_path)]):
            events.append(event)

        self.mock_client.upload_file.assert_called_once_with(str(doc_path))
        attachments = self.mock_client.chat.last_attachments
        assert attachments is not None
        assert len(attachments) == 1
        assert attachments[0]["file_name"] == "report.pdf"

    async def test_mixed_uploads(self):
        img = self.tmp_path / "photo.jpg"
        doc = self.tmp_path / "data.csv"
        img.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 50)
        doc.write_bytes(b'a,b,c\n1,2,3')

        events = []
        async for event in self.bridge.stream_ask(
            "topic_1", "analyze", image_paths=[str(img)], file_paths=[str(doc)]
        ):
            events.append(event)

        assert self.mock_client.upload_file.call_count == 2
        attachments = self.mock_client.chat.last_attachments
        assert len(attachments) == 2

    async def test_no_attachments_passes_none(self):
        events = []
        async for event in self.bridge.stream_ask("topic_1", "just text"):
            events.append(event)

        attachments = self.mock_client.chat.last_attachments
        assert attachments is None

    async def test_upload_failure_graceful(self):
        self.mock_client.upload_file = AsyncMock(side_effect=Exception("upload failed"))

        img = self.tmp_path / "bad.png"
        img.write_bytes(b'\x89PNG' + b'\x00' * 50)

        events = []
        async for event in self.bridge.stream_ask("topic_1", "try this", image_paths=[str(img)]):
            events.append(event)

        final_events = [e for e in events if e["type"] == "final"]
        assert len(final_events) == 1
        attachments = self.mock_client.chat.last_attachments
        assert attachments is None


# ===========================================================================
# 9. FileInfo.to_attachment format
# ===========================================================================

class TestFileInfoToAttachment:
    def test_basic(self):
        from mira_client import FileInfo
        fi = FileInfo(
            file_name="test.png",
            url="https://example.com/test.png",
            uri="mira://test.png",
            mime_type="image/png",
        )
        att = fi.to_attachment()
        assert att == {
            "file_name": "test.png",
            "url": "https://example.com/test.png",
            "mime_type": "image/png",
            "uri": "mira://test.png",
        }

    def test_all_required_keys(self):
        from mira_client import FileInfo
        fi = FileInfo(file_name="f", url="u", uri="r", mime_type="m")
        att = fi.to_attachment()
        assert set(att.keys()) == {"file_name", "url", "mime_type", "uri"}


# ===========================================================================
# 10. Integration-style: attachment_meta flow
# ===========================================================================

class TestEndToEndAttachmentFlow:

    def test_image_only_message_generates_prompt(self):
        content = {"image_key": "img_v3_test"}
        attachment_meta = []
        image_key = content.get("image_key", "")
        if image_key:
            attachment_meta.append({"type": "image", "key": image_key, "name": ""})
        text = ""

        if not text and attachment_meta:
            names = [a.get("name") or a["key"] for a in attachment_meta]
            text = f"[用户发送了附件: {', '.join(names)}]"

        assert text == "[用户发送了附件: img_v3_test]"
        assert len(attachment_meta) == 1
        assert attachment_meta[0]["type"] == "image"

    def test_file_message_generates_prompt(self):
        content = {"file_key": "file_abc", "file_name": "design.pdf"}
        attachment_meta = []
        file_key = content.get("file_key", "")
        file_name = content.get("file_name", "attachment")
        if file_key:
            attachment_meta.append({"type": "file", "key": file_key, "name": file_name, "msg_id": "m1"})
        text = ""

        if not text and attachment_meta:
            names = [a.get("name") or a["key"] for a in attachment_meta]
            text = f"[用户发送了附件: {', '.join(names)}]"

        assert text == "[用户发送了附件: design.pdf]"
        assert attachment_meta[0]["type"] == "file"
        assert attachment_meta[0]["name"] == "design.pdf"

    def test_unsupported_type_text(self):
        for msg_type in ("audio", "video", "media"):
            text = f"[不支持的消息类型: {msg_type}，请发送文字、图片或文件]"
            assert "不支持" in text
            assert msg_type in text


# ===========================================================================
# 11. Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_build_cmd_claude_special_chars_in_file_path(self):
        bridge = ClaudeCodeBridge({"aiden_cmd": "aiden", "model": "m", "target": "claude"})
        cmd = bridge._build_cmd_claude("hi", None, file_paths=["/tmp/my file.pdf"])
        args_str = " ".join(cmd)
        assert "[附件: /tmp/my file.pdf]" in args_str

    def test_build_cmd_codex_special_chars_in_image_path(self):
        bridge = ClaudeCodeBridge({"aiden_cmd": "aiden", "model": "m", "target": "codex", "provider": "codex"})
        cmd = bridge._build_cmd_codex("hi", None, image_paths=["/tmp/my image.png"])
        args_str = " ".join(cmd)
        assert "--image /tmp/my image.png" in args_str

    def test_guess_image_ext_with_gif87a(self):
        from feishu_bot import FeishuBot
        data = b'GIF87a' + b'\x00' * 100
        assert FeishuBot._guess_image_ext(data) == ".gif"

    def test_file_path_sanitization(self):
        name = "../../etc/passwd"
        safe_name = name.replace("/", "_").replace("..", "_")
        assert "/" not in safe_name
        assert ".." not in safe_name

    def test_empty_image_key_not_added(self):
        content = {"image_key": ""}
        attachment_meta = []
        image_key = content.get("image_key", "")
        if image_key:
            attachment_meta.append({"type": "image", "key": image_key, "name": ""})
        assert len(attachment_meta) == 0

    def test_empty_file_key_not_added(self):
        content = {"file_key": "", "file_name": "test.pdf"}
        attachment_meta = []
        file_key = content.get("file_key", "")
        if file_key:
            attachment_meta.append({"type": "file", "key": file_key, "name": "test.pdf"})
        assert len(attachment_meta) == 0
