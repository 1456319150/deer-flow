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
    """Integration-style tests verifying card creation order."""

    @pytest.fixture
    def bot(self):
        """Create a FeishuBot with mocked Feishu SDK."""
        cfg = {"app_id": "test", "app_secret": "test"}
        bridge = MagicMock()
        bot = FeishuBot(cfg, bridge)
        # Mock the API methods
        bot._reply_card = AsyncMock(side_effect=lambda mid, text: f"card_{id(text)}")
        bot._update_card = AsyncMock()
        bot._react = AsyncMock()
        return bot

    @pytest.mark.asyncio
    async def test_card_creation_order_thinking_tools_result(self, bot):
        """Cards must be created in order: thinking -> process -> result."""
        card_creation_order = []
        original_reply = bot._reply_card

        async def tracking_reply(msg_id, text):
            if "Thinking" in text:
                card_creation_order.append("thinking")
            elif "🔧 Tool Calls" in text:
                card_creation_order.append("process")
            elif "Thinking" not in text and "🔧" not in text and "Usage" not in text:
                card_creation_order.append("result")
            return f"card_{len(card_creation_order)}"

        bot._reply_card = AsyncMock(side_effect=tracking_reply)

        # Simulate stream events in realistic order
        events = [
            {"type": "stream_event", "event": StreamEvent(kind="thinking", text="let me think...")},
            {"type": "stream_event", "event": StreamEvent(kind="tool_use", text="search query",
                                                           tool_name="web_search", tool_use_id="tu_1")},
            {"type": "stream_event", "event": StreamEvent(kind="tool_result", text="search results",
                                                           tool_name="web_search", tool_use_id="tu_1")},
            {"type": "stream_event", "event": StreamEvent(kind="result", text="Here is the answer.")},
            {"type": "final", "result": StreamResult()},
        ]

        async def fake_stream(topic_id, text):
            for e in events:
                yield e

        bot.bridge.stream_ask = fake_stream

        await bot._handle("chat_1", "msg_1", "topic_1", "test query")

        # Verify order: thinking must come before process, process before result
        if "thinking" in card_creation_order and "process" in card_creation_order:
            assert card_creation_order.index("thinking") < card_creation_order.index("process"), \
                f"thinking must be before process, got: {card_creation_order}"
        if "process" in card_creation_order and "result" in card_creation_order:
            assert card_creation_order.index("process") < card_creation_order.index("result"), \
                f"process must be before result, got: {card_creation_order}"

    @pytest.mark.asyncio
    async def test_multiple_tools_accumulate_in_single_card(self, bot):
        """Multiple tool_use events should accumulate into one process card, not separate cards."""
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

        async def fake_stream(topic_id, text):
            for e in events:
                yield e

        bot.bridge.stream_ask = fake_stream

        await bot._handle("chat_1", "msg_1", "topic_1", "test")

        # Find the process card — should contain BOTH tools
        process_cards = [t for t in reply_texts if "🔧 Tool Calls" in t]
        assert len(process_cards) >= 1, f"Expected at least 1 process card, got texts: {reply_texts}"
        # The final process card should mention both tools
        final_process = process_cards[-1]
        assert "search" in final_process
        assert "fetch" in final_process


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Image / file sanitization for Feishu cards
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


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
