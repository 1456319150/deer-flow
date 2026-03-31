import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock

from gateway import ClaudeCodeBridge, StreamResult, ToolCall, UsageSummary
from weixin_bot import WeixinBot
from weixin_channel import (
    ContextTokenStore,
    SyncCursor,
    WeixinAccount,
    WeixinChannel,
    strip_markdown,
    load_account,
    save_account,
)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self, **kwargs):
        return self.payload


class _FakeSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []
        self.closed = False

    def post(self, url, **kwargs):
        self.requests.append({"url": url, **kwargs})
        if not self.payloads:
            raise AssertionError("No fake payload left for request")
        return _FakeResponse(self.payloads.pop(0))

    async def close(self):
        self.closed = True


class TestWeixinHelpers(unittest.TestCase):
    def test_strip_markdown_keeps_plain_text_readable(self):
        text = "# Title\n**bold** [link](https://example.com)\n> quote"
        output = strip_markdown(text)
        self.assertIn("【Title】", output)
        self.assertIn("bold", output)
        self.assertIn("link (https://example.com)", output)
        self.assertNotIn("**", output)

    def test_save_and_load_account_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            account = WeixinAccount(bot_token="token-123", baseurl="https://wx.example")
            save_account(account, path)
            loaded = load_account(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.bot_token, "token-123")
            self.assertEqual(loaded.baseurl, "https://wx.example")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_context_token_store_persists_values(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store = ContextTokenStore(path)
            store.set("user-1", "ctx-1")

            reloaded = ContextTokenStore(path)
            self.assertEqual(reloaded.get("user-1"), "ctx-1")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_sync_cursor_persists_updates(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cursor = SyncCursor(path)
            cursor.update("buf-123")

            reloaded = SyncCursor(path)
            self.assertEqual(reloaded.buf, "buf-123")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_split_text_prefers_paragraph_boundaries(self):
        text = "para1\n\n" + ("x" * 20) + "\n\npara3"
        chunks = WeixinChannel._split_text(text, max_len=16)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk for chunk in chunks))
        self.assertEqual("".join(chunks), text)


class TestWeixinChannel(unittest.IsolatedAsyncioTestCase):
    async def test_get_updates_parses_messages_and_updates_state(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as ctx_f, tempfile.NamedTemporaryFile(suffix=".json", delete=False) as cursor_f:
            ctx_path = ctx_f.name
            cursor_path = cursor_f.name
        try:
            ctx_store = ContextTokenStore(ctx_path)
            sync_cursor = SyncCursor(cursor_path)
            account = WeixinAccount(bot_token="token", baseurl="https://wx.example")
            channel = WeixinChannel(account, ctx_store, sync_cursor)
            session = _FakeSession([
                {
                    "get_updates_buf": "buf-next",
                    "msgs": [
                        {
                            "from_user_id": "user-1",
                            "context_token": "ctx-1",
                            "message_type": 1,
                            "msg_id": "msg-1",
                            "timestamp": 1710000000,
                            "item_list": [{"type": 1, "text_item": {"text": "hello"}}],
                        },
                        {
                            "from_user_id": "bot-self",
                            "context_token": "ctx-bot",
                            "message_type": 2,
                            "msg_id": "msg-2",
                            "timestamp": 1710000001,
                            "item_list": [{"type": 1, "text_item": {"text": "skip me"}}],
                        },
                    ],
                }
            ])
            channel._ensure_session = AsyncMock(return_value=session)

            messages = await channel._get_updates()

            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].from_user, "user-1")
            self.assertEqual(messages[0].text, "hello")
            self.assertEqual(messages[0].context_token, "ctx-1")
            self.assertEqual(ctx_store.get("user-1"), "ctx-1")
            self.assertEqual(sync_cursor.buf, "buf-next")
            self.assertTrue(session.requests[0]["url"].endswith("/ilink/bot/getupdates"))
            self.assertEqual(session.requests[0]["json"]["get_updates_buf"], "")
        finally:
            for path in (ctx_path, cursor_path):
                if os.path.exists(path):
                    os.unlink(path)

    async def test_send_text_recovers_context_and_splits_long_messages(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as ctx_f, tempfile.NamedTemporaryFile(suffix=".json", delete=False) as cursor_f:
            ctx_path = ctx_f.name
            cursor_path = cursor_f.name
        try:
            ctx_store = ContextTokenStore(ctx_path)
            ctx_store.set("user-1", "ctx-stored")
            sync_cursor = SyncCursor(cursor_path)
            account = WeixinAccount(bot_token="token", baseurl="https://wx.example")
            channel = WeixinChannel(account, ctx_store, sync_cursor)
            session = _FakeSession([{"ret": 0}, {"ret": 0}])
            channel._ensure_session = AsyncMock(return_value=session)

            original_sleep = asyncio.sleep
            asyncio.sleep = AsyncMock()
            try:
                result = await channel.send_text("user-1", "", "a" * 4001)
            finally:
                asyncio.sleep = original_sleep

            self.assertEqual(result, {"ret": 0})
            self.assertEqual(len(session.requests), 2)
            first_payload = session.requests[0]["json"]["msg"]
            second_payload = session.requests[1]["json"]["msg"]
            self.assertEqual(first_payload["context_token"], "ctx-stored")
            self.assertEqual(second_payload["context_token"], "ctx-stored")
            self.assertEqual(len(first_payload["item_list"][0]["text_item"]["text"]), 4000)
            self.assertEqual(len(second_payload["item_list"][0]["text_item"]["text"]), 1)
        finally:
            for path in (ctx_path, cursor_path):
                if os.path.exists(path):
                    os.unlink(path)

    async def test_send_typing_uses_ticket_from_getconfig(self):
        account = WeixinAccount(bot_token="token", baseurl="https://wx.example")
        channel = WeixinChannel(account)
        session = _FakeSession([
            {"typing_ticket": "ticket-1"},
            {"ret": 0},
        ])
        channel._ensure_session = AsyncMock(return_value=session)

        await channel.send_typing("user-1", "ctx-1", typing=False)

        self.assertEqual(len(session.requests), 2)
        self.assertTrue(session.requests[0]["url"].endswith("/ilink/bot/getconfig"))
        self.assertTrue(session.requests[1]["url"].endswith("/ilink/bot/sendtyping"))
        self.assertEqual(session.requests[1]["json"]["status"], 2)


class TestWeixinBot(unittest.IsolatedAsyncioTestCase):
    async def test_handle_message_routes_to_bridge_and_sends_reply(self):
        bridge = ClaudeCodeBridge({})
        bot = WeixinBot({"enabled": True}, bridge)
        bot._channel = AsyncMock()

        seen = {}

        async def fake_stream_ask(topic_id, prompt):
            seen["topic_id"] = topic_id
            seen["prompt"] = prompt
            yield {"type": "final", "result": StreamResult(assistant_texts=["**你好**"])}

        bot.bridge.stream_ask = fake_stream_ask
        msg = type("Msg", (), {
            "msg_id": "msg-1",
            "timestamp": time.time(),
            "text": "hello from wechat",
            "from_user": "user-1",
            "context_token": "ctx-1",
            "is_empty": False,
        })()

        await bot._handle_message(msg)

        self.assertEqual(seen["topic_id"], "wx_user-1")
        self.assertEqual(seen["prompt"], "hello from wechat")
        self.assertEqual(bot._channel.send_typing.await_count, 2)
        bot._channel.send_text.assert_awaited_once_with("user-1", "ctx-1", "你好")

    async def test_handle_message_sends_usage_as_second_message(self):
        bridge = ClaudeCodeBridge({})
        bot = WeixinBot({"enabled": True}, bridge)
        bot._channel = AsyncMock()

        async def fake_stream_ask(topic_id, prompt):
            yield {
                "type": "final",
                "result": StreamResult(
                    assistant_texts=["完成了"],
                    usage=UsageSummary(
                        input_tokens=1200,
                        output_tokens=34,
                        total_tokens=1234,
                        cost_usd=0.0165,
                    ),
                ),
            }

        bot.bridge.stream_ask = fake_stream_ask
        msg = type("Msg", (), {
            "msg_id": "msg-usage",
            "timestamp": time.time(),
            "text": "hello from wechat",
            "from_user": "user-1",
            "context_token": "ctx-1",
            "is_empty": False,
        })()

        await bot._handle_message(msg)

        self.assertEqual(bot._channel.send_text.await_count, 2)
        first_call = bot._channel.send_text.await_args_list[0].args
        second_call = bot._channel.send_text.await_args_list[1].args
        self.assertEqual(first_call, ("user-1", "ctx-1", "完成了"))
        self.assertEqual(second_call[0], "user-1")
        self.assertEqual(second_call[1], "ctx-1")
        self.assertIn("用量统计", second_call[2])
        self.assertIn("输入 tokens: 1,200", second_call[2])
        self.assertIn("输出 tokens: 34", second_call[2])
        self.assertIn("总 tokens: 1,234", second_call[2])
        self.assertIn("金额: $0.0165", second_call[2])

    async def test_new_command_resets_session(self):
        bridge = ClaudeCodeBridge({})
        bridge._sessions["wx_user-1"] = "sess_123"
        bot = WeixinBot({"enabled": True}, bridge)
        bot._channel = AsyncMock()
        msg = type("Msg", (), {
            "msg_id": "msg-new",
            "timestamp": time.time(),
            "text": "/new",
            "from_user": "user-1",
            "context_token": "ctx-1",
            "is_empty": False,
        })()

        await bot._handle_message(msg)

        self.assertIsNone(bridge.get_session("wx_user-1"))
        bot._channel.send_typing.assert_not_awaited()
        bot._channel.send_text.assert_awaited_once()
        self.assertIn("已重置当前会话", bot._channel.send_text.await_args.args[2])
        self.assertIn("sess_123", bot._channel.send_text.await_args.args[2])

    async def test_session_command_reports_current_session(self):
        bridge = ClaudeCodeBridge({})
        bridge._sessions["wx_user-1"] = "sess_123"
        bot = WeixinBot({"enabled": True}, bridge)
        bot._channel = AsyncMock()
        msg = type("Msg", (), {
            "msg_id": "msg-session",
            "timestamp": time.time(),
            "text": "/session",
            "from_user": "user-1",
            "context_token": "ctx-1",
            "is_empty": False,
        })()

        await bot._handle_message(msg)

        bot._channel.send_typing.assert_not_awaited()
        bot._channel.send_text.assert_awaited_once_with("user-1", "ctx-1", "当前 session: sess_123")

    async def test_session_command_when_empty(self):
        bridge = ClaudeCodeBridge({})
        bot = WeixinBot({"enabled": True}, bridge)
        bot._channel = AsyncMock()
        msg = type("Msg", (), {
            "msg_id": "msg-session-empty",
            "timestamp": time.time(),
            "text": "/session",
            "from_user": "user-1",
            "context_token": "ctx-1",
            "is_empty": False,
        })()

        await bot._handle_message(msg)

        bot._channel.send_text.assert_awaited_once_with("user-1", "ctx-1", "当前还没有活动会话。")

    async def test_reset_alias_supported(self):
        bridge = ClaudeCodeBridge({})
        bridge._sessions["wx_user-1"] = "sess_123"
        bot = WeixinBot({"enabled": True}, bridge)
        bot._channel = AsyncMock()
        msg = type("Msg", (), {
            "msg_id": "msg-reset",
            "timestamp": time.time(),
            "text": "/reset",
            "from_user": "user-1",
            "context_token": "ctx-1",
            "is_empty": False,
        })()

        await bot._handle_message(msg)

        self.assertIsNone(bridge.get_session("wx_user-1"))
        bot._channel.send_text.assert_awaited_once()

    async def test_handle_message_skips_duplicate_ids(self):
        bridge = ClaudeCodeBridge({})
        bot = WeixinBot({"enabled": True}, bridge)
        bot._channel = AsyncMock()

        async def fake_stream_ask(topic_id, prompt):
            yield {"type": "final", "result": StreamResult(assistant_texts=["ok"])}

        bot.bridge.stream_ask = fake_stream_ask
        msg = type("Msg", (), {
            "msg_id": "dup-msg",
            "timestamp": time.time(),
            "text": "hello",
            "from_user": "user-1",
            "context_token": "ctx-1",
            "is_empty": False,
        })()

        await bot._handle_message(msg)
        await bot._handle_message(msg)

        bot._channel.send_text.assert_awaited_once()

    def test_format_reply_includes_tool_summary_and_reply(self):
        result = StreamResult(
            assistant_texts=["done"],
            tool_calls=[
                ToolCall(name="Read", input_text="a"),
                ToolCall(name="Bash", input_text="b"),
            ],
        )

        reply = WeixinBot._format_reply(result)

        self.assertIn("执行了", reply)
        self.assertIn("Read, Bash", reply)
        self.assertIn("done", reply)

    def test_format_usage_summary_returns_empty_when_no_usage(self):
        self.assertEqual(WeixinBot._format_usage_summary(None), "")

    def test_format_usage_summary_includes_cost_and_totals(self):
        usage = UsageSummary(
            input_tokens=1200,
            output_tokens=34,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=400,
            total_tokens=1654,
            cost_usd=0.0165,
        )
        summary = WeixinBot._format_usage_summary(usage)
        self.assertIn("用量统计", summary)
        self.assertIn("输入 tokens: 1,200", summary)
        self.assertIn("缓存写入 tokens: 20", summary)
        self.assertIn("缓存命中 tokens: 400", summary)
        self.assertIn("总 tokens: 1,654", summary)
        self.assertIn("金额: $0.0165", summary)


if __name__ == "__main__":
    unittest.main()
