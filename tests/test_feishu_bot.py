"""Tests for FeishuBot card rendering helpers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from gateway import StreamEvent, StreamResult


# We only test the classmethod/staticmethod helpers — no Feishu SDK needed.
from feishu_bot import FeishuBot


class TestBuildProcessCard:
    """Tests for _build_process_card — the unified tool card builder."""

    def test_empty_blocks(self):
        assert FeishuBot._build_process_card([]) == ""

    def test_single_block(self):
        result = FeishuBot._build_process_card(["**search**\n`hello`"])
        assert "🔧 Tool Calls (1)" in result
        assert "**search**\n`hello`" in result

    def test_multiple_blocks_joined_with_separator(self):
        blocks = ["**tool_a**\n`input_a`", "**tool_b**\n`input_b`"]
        result = FeishuBot._build_process_card(blocks)
        assert "🔧 Tool Calls (2)" in result
        assert "---" in result  # separator between blocks
        assert "input_a" in result
        assert "input_b" in result

    def test_truncation(self):
        # Build blocks that exceed MAX_REPLY
        big_block = "x" * (FeishuBot.MAX_REPLY + 100)
        result = FeishuBot._build_process_card([big_block])
        assert len(result) <= FeishuBot.MAX_REPLY + 100  # header + truncation msg
        assert "truncated" in result


class TestFormatStreamEventToolUse:
    """Tests for _format_stream_event tool_use/tool_result rendering."""

    def test_tool_use_no_emoji_prefix(self):
        """tool_use blocks inside process card should NOT have 🔧 prefix (card header has it)."""
        evt = StreamEvent(kind="tool_use", text="query text", tool_name="mira_search")
        result = FeishuBot._format_stream_event(evt)
        assert "🔧" not in result
        assert "**mira_search**" in result
        assert "`query text`" in result

    def test_tool_result_with_icon(self):
        evt = StreamEvent(kind="tool_result", text="some output", tool_name="mira_search")
        result = FeishuBot._format_stream_event(evt)
        assert "📦" in result
        assert "mira_search" in result
        assert "some output" in result

    def test_tool_use_truncation(self):
        long_input = "a" * (FeishuBot.MAX_TOOL_INPUT + 100)
        evt = StreamEvent(kind="tool_use", text=long_input, tool_name="tool")
        result = FeishuBot._format_stream_event(evt)
        assert "..." in result

    def test_tool_result_truncation(self):
        long_output = "b" * (FeishuBot.MAX_TOOL_OUTPUT + 100)
        evt = StreamEvent(kind="tool_result", text=long_output, tool_name="tool")
        result = FeishuBot._format_stream_event(evt)
        assert "..." in result


class TestMergeToolStreamBlocks:
    """Tests for _merge_tool_stream_blocks."""

    def test_merge_combines_blocks(self):
        use_block = "**search**\n`query`"
        result_evt = StreamEvent(kind="tool_result", text="found 3 results", tool_name="search")
        merged = FeishuBot._merge_tool_stream_blocks(use_block, result_evt)
        assert "**search**" in merged
        assert "found 3 results" in merged

    def test_merge_empty_use_block(self):
        result_evt = StreamEvent(kind="tool_result", text="output", tool_name="t")
        merged = FeishuBot._merge_tool_stream_blocks("", result_evt)
        assert "output" in merged

    def test_merge_empty_result(self):
        merged = FeishuBot._merge_tool_stream_blocks("use block", StreamEvent(kind="tool_result", text="", tool_name="t"))
        assert "use block" in merged  # tool_result header still appended


class TestEventDedupKey:
    """Tests for _event_dedup_key."""

    def test_text_event_has_key(self):
        evt = StreamEvent(kind="text", text="hello")
        key = FeishuBot._event_dedup_key(evt)
        assert key == ("text", "hello")

    def test_result_event_has_key(self):
        evt = StreamEvent(kind="result", text="answer")
        key = FeishuBot._event_dedup_key(evt)
        assert key == ("result", "answer")

    def test_tool_use_no_key(self):
        evt = StreamEvent(kind="tool_use", text="input", tool_name="t")
        assert FeishuBot._event_dedup_key(evt) is None

    def test_empty_text_no_key(self):
        evt = StreamEvent(kind="text", text="   ")
        assert FeishuBot._event_dedup_key(evt) is None



class TestCardOrderingIntegration:
    """Integration-style tests verifying 2-card model (thinking+tools, result)."""

    @pytest.fixture
    def bot(self):
        """Create a FeishuBot with mocked Feishu SDK."""
        cfg = {"app_id": "test", "app_secret": "test"}
        bridge = MagicMock()
        bot = FeishuBot(cfg, bridge)
        bot._reply_card = AsyncMock(side_effect=lambda mid, text: f"card_{id(text)}")
        bot._update_card = AsyncMock()
        bot._react = AsyncMock()
        # Skip image upload in tests
        bot._resolve_images = AsyncMock(side_effect=lambda text: text)
        return bot

    @pytest.mark.asyncio
    async def test_thinking_and_tools_merged_in_one_card(self, bot):
        """Thinking + tool calls should appear in a single combined card."""
        reply_texts = []

        async def capture_reply(msg_id, text):
            reply_texts.append(text)
            return f"card_{len(reply_texts)}"

        bot._reply_card = AsyncMock(side_effect=capture_reply)

        events = [
            {"type": "stream_event", "event": StreamEvent(kind="thinking", text="let me think...")},
            {"type": "stream_event", "event": StreamEvent(kind="tool_use", text="search query",
                                                           tool_name="web_search", tool_use_id="tu_1")},
            {"type": "stream_event", "event": StreamEvent(kind="tool_result", text="search results",
                                                           tool_name="web_search", tool_use_id="tu_1")},
            {"type": "stream_event", "event": StreamEvent(kind="result", text="Here is the answer.")},
            {"type": "final", "result": StreamResult()},
        ]

        async def fake_stream(topic_id, text, **kwargs):
            for e in events:
                yield e

        bot.bridge.stream_ask = fake_stream
        await bot._handle("chat_1", "msg_1", "topic_1", "test query")

        # Final flush should produce a combined card with both Thinking and Tool Calls
        combined = [t for t in reply_texts if "Thinking" in t and "Tool Calls" in t]
        assert len(combined) >= 1, f"Expected combined thinking+tools card, got: {reply_texts}"

    @pytest.mark.asyncio
    async def test_thinking_card_before_result_card(self, bot):
        """The thinking card must be created before the result card."""
        card_order = []

        async def tracking_reply(msg_id, text):
            if "Thinking" in text or "Tool Calls" in text:
                card_order.append("thinking")
            elif "Usage" not in text:
                card_order.append("result")
            return f"card_{len(card_order)}"

        bot._reply_card = AsyncMock(side_effect=tracking_reply)

        events = [
            {"type": "stream_event", "event": StreamEvent(kind="thinking", text="hmm...")},
            {"type": "stream_event", "event": StreamEvent(kind="tool_use", text="q",
                                                           tool_name="search", tool_use_id="tu_1")},
            {"type": "stream_event", "event": StreamEvent(kind="result", text="answer")},
            {"type": "final", "result": StreamResult()},
        ]

        async def fake_stream(topic_id, text, **kwargs):
            for e in events:
                yield e

        bot.bridge.stream_ask = fake_stream
        await bot._handle("chat_1", "msg_1", "topic_1", "test")

        if "thinking" in card_order and "result" in card_order:
            assert card_order.index("thinking") < card_order.index("result"), \
                f"thinking must come before result, got: {card_order}"

    @pytest.mark.asyncio
    async def test_multiple_tools_in_thinking_card(self, bot):
        """Multiple tool_use events should accumulate in the thinking card."""
        reply_texts = []

        async def capture_reply(msg_id, text):
            reply_texts.append(text)
            return f"card_{len(reply_texts)}"

        bot._reply_card = AsyncMock(side_effect=capture_reply)

        events = [
            {"type": "stream_event", "event": StreamEvent(kind="tool_use", text="q1",
                                                           tool_name="search", tool_use_id="tu_1")},
            {"type": "stream_event", "event": StreamEvent(kind="tool_use", text="q2",
                                                           tool_name="fetch", tool_use_id="tu_2")},
            {"type": "stream_event", "event": StreamEvent(kind="result", text="answer")},
            {"type": "final", "result": StreamResult()},
        ]

        async def fake_stream(topic_id, text, **kwargs):
            for e in events:
                yield e

        bot.bridge.stream_ask = fake_stream
        await bot._handle("chat_1", "msg_1", "topic_1", "test")

        tool_cards = [t for t in reply_texts if "Tool Calls" in t]
        assert len(tool_cards) >= 1, f"Expected tool card, got: {reply_texts}"
        final_card = tool_cards[-1]
        assert "search" in final_card
        assert "fetch" in final_card

    @pytest.mark.asyncio
    async def test_tools_only_no_thinking_text(self, bot):
        """Tool calls without thinking should still create a card."""
        reply_texts = []

        async def capture_reply(msg_id, text):
            reply_texts.append(text)
            return f"card_{len(reply_texts)}"

        bot._reply_card = AsyncMock(side_effect=capture_reply)

        events = [
            {"type": "stream_event", "event": StreamEvent(kind="tool_use", text="query",
                                                           tool_name="search", tool_use_id="tu_1")},
            {"type": "stream_event", "event": StreamEvent(kind="result", text="answer")},
            {"type": "final", "result": StreamResult()},
        ]

        async def fake_stream(topic_id, text, **kwargs):
            for e in events:
                yield e

        bot.bridge.stream_ask = fake_stream
        await bot._handle("chat_1", "msg_1", "topic_1", "test")

        tool_cards = [t for t in reply_texts if "Tool Calls" in t]
        assert len(tool_cards) >= 1, "Tools without thinking should still create a card"


# \u2501\u2501 Image resolution tests \u2501\u2501

class TestResolveImages:
    """Test _resolve_images uploads images and replaces URLs with image_keys."""

    @pytest.fixture
    def bot(self):
        cfg = {"app_id": "test", "app_secret": "test"}
        bridge = MagicMock()
        bot = FeishuBot(cfg, bridge)
        bot._image_cache = {}
        return bot

    @pytest.mark.asyncio
    async def test_no_images_unchanged(self, bot):
        text = "Just normal text with [link](https://example.com)"
        result = await FeishuBot._resolve_images(bot, text)
        assert result == text

    @pytest.mark.asyncio
    async def test_empty_text(self, bot):
        assert await FeishuBot._resolve_images(bot, "") == ""
        assert await FeishuBot._resolve_images(bot, None) is None

    @pytest.mark.asyncio
    async def test_successful_upload_replaces_url(self, bot):
        bot._upload_image_to_feishu = AsyncMock(return_value="img_uploaded_key")
        text = "Here is ![chart](https://example.com/chart.png) done"
        result = await FeishuBot._resolve_images(bot, text)
        assert "![chart](img_uploaded_key)" in result
        assert "example.com" not in result

    @pytest.mark.asyncio
    async def test_failed_upload_leaves_url_unchanged(self, bot):
        bot._upload_image_to_feishu = AsyncMock(return_value=None)
        text = "Here is ![chart](https://example.com/chart.png) done"
        result = await FeishuBot._resolve_images(bot, text)
        assert "![chart](https://example.com/chart.png)" in result

    @pytest.mark.asyncio
    async def test_multiple_images_partial_upload(self, bot):
        async def selective_upload(url):
            if "good" in url:
                return "img_good_key"
            return None

        bot._upload_image_to_feishu = AsyncMock(side_effect=selective_upload)
        text = "![a](https://good.com/a.png) and ![b](https://bad.com/b.png)"
        result = await FeishuBot._resolve_images(bot, text)
        assert "![a](img_good_key)" in result
        assert "![b](https://bad.com/b.png)" in result


# \u2501\u2501 Inbound message handling tests \u2501\u2501

class TestExtractText:
    """Test _extract_text handles various message formats."""

    def test_plain_text(self):
        assert FeishuBot._extract_text({"text": "hello"}) == "hello"

    def test_rich_text(self):
        content = {
            "content": [
                [{"tag": "text", "text": "hello"}, {"tag": "text", "text": " world"}]
            ]
        }
        result = FeishuBot._extract_text(content)
        assert "hello" in result
        assert "world" in result

    def test_empty_content(self):
        assert FeishuBot._extract_text({}) == ""


class TestSanitizeForCard:
    """Test _sanitize_for_card strips images that would trigger Feishu error 11310."""

    def test_markdown_image_converted(self):
        text = "Result: ![chart](https://cdn.example.com/chart.png) done"
        result = FeishuBot._sanitize_for_card(text)
        assert "![" not in result
        assert "[chart](https://cdn.example.com/chart.png)" in result
        assert result.startswith("Result: ")
        assert result.endswith(" done")

    def test_markdown_image_with_title(self):
        text = '![alt](https://x.com/a.jpg "My Title")'
        result = FeishuBot._sanitize_for_card(text)
        assert "![" not in result
        assert "[alt](https://x.com/a.jpg)" in result

    def test_markdown_image_empty_alt(self):
        text = "![](https://example.com/img.png)"
        result = FeishuBot._sanitize_for_card(text)
        assert "![" not in result
        assert "[image](https://example.com/img.png)" in result

    def test_html_img_tag(self):
        text = 'See: <img src="https://cdn.example.com/chart.png" width="400"> end'
        result = FeishuBot._sanitize_for_card(text)
        assert "<img" not in result
        assert "[image](https://cdn.example.com/chart.png)" in result

    def test_feishu_image_token(self):
        text = 'Content <image token="boxcnABC123XYZ" /> more'
        result = FeishuBot._sanitize_for_card(text)
        assert "<image" not in result
        assert "[image:boxcnABC123XYZ]" in result

    def test_multiple_images_mixed(self):
        text = "![a](url1.png) text <img src='url2.jpg'/> end"
        result = FeishuBot._sanitize_for_card(text)
        assert "![" not in result
        assert "<img" not in result
        assert "[a](url1.png)" in result
        assert "[image](url2.jpg)" in result

    def test_no_images_unchanged(self):
        text = "Normal **bold** text with [link](https://example.com)"
        result = FeishuBot._sanitize_for_card(text)
        assert result == text

    def test_empty_and_none(self):
        assert FeishuBot._sanitize_for_card("") == ""
        assert FeishuBot._sanitize_for_card(None) is None

    def test_regular_link_not_affected(self):
        """Ensure [text](url) links are NOT converted."""
        text = "Click [here](https://example.com/page) for more"
        result = FeishuBot._sanitize_for_card(text)
        assert result == text

    def test_card_integrates_sanitization(self):
        """Verify _card() calls _sanitize_for_card."""
        import json
        card_json = FeishuBot._card("Hello ![img](https://x.com/a.png) world")
        card = json.loads(card_json)
        content = card["elements"][0]["content"]
        assert "![" not in content
        assert "[img](https://x.com/a.png)" in content
