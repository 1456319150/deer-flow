"""Unit tests for gateway.py — stream-json parsing + rich formatting."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from feishu_bot import FeishuBot
from gateway import (
    ClaudeCodeBridge,
    StreamEvent,
    StreamResult,
    StreamState,
    ToolCall,
    _preview_text,
    load_config,
    load_dotenv,
)


# ===========================================================================
# StreamResult & ToolCall
# ===========================================================================

class TestStreamResult(unittest.TestCase):
    """Tests for the StreamResult dataclass."""

    def test_preview_text_escapes_newlines_and_truncates(self):
        self.assertEqual(_preview_text("a\nb", limit=10), "a\\nb")
        self.assertEqual(_preview_text("123456", limit=4), "1234...")

    def test_reply_text_prefers_result(self):
        r = StreamResult(result_text="from result", assistant_texts=["from assistant"])
        self.assertEqual(r.reply_text, "from result")

    def test_reply_text_falls_back_to_assistant(self):
        r = StreamResult(assistant_texts=["hello", "world"])
        self.assertEqual(r.reply_text, "hello\n\nworld")

    def test_reply_text_empty(self):
        r = StreamResult()
        self.assertEqual(r.reply_text, "")

    def test_is_empty_when_no_text_no_tools(self):
        r = StreamResult()
        self.assertTrue(r.is_empty)

    def test_not_empty_with_text(self):
        r = StreamResult(assistant_texts=["hi"])
        self.assertFalse(r.is_empty)

    def test_not_empty_with_tools(self):
        r = StreamResult(tool_calls=[ToolCall(name="Bash", input_text="ls")])
        self.assertFalse(r.is_empty)


# ===========================================================================
# _parse_stream
# ===========================================================================

class TestParseStream(unittest.TestCase):
    """Tests for ClaudeCodeBridge._parse_stream with full event coverage."""

    @staticmethod
    def _build_stream(*events: dict) -> str:
        return "\n".join(json.dumps(e) for e in events)

    def test_basic_assistant_text(self):
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello!"}]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(result.assistant_texts, ["Hello!"])

    def test_thinking_extracted(self):
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "Let me analyze this..."}
            ]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(result.thinking, ["Let me analyze this..."])
        self.assertEqual(result.assistant_texts, [])

    def test_redacted_thinking_ignored(self):
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [
                {"type": "redacted_thinking", "data": "encrypted_blob"}
            ]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(result.thinking, [])

    def test_tool_use_extracted(self):
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tool_1", "name": "Bash",
                 "input": {"command": "ls -la"}}
            ]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "Bash")
        self.assertEqual(result.tool_calls[0].input_text, "ls -la")

    def test_tool_use_with_file_path(self):
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tool_2", "name": "Read",
                 "input": {"file_path": "/src/main.py"}}
            ]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(result.tool_calls[0].input_text, "/src/main.py")

    def test_tool_use_with_generic_input(self):
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tool_3", "name": "Edit",
                 "input": {"file": "a.py", "changes": "fix bug"}}
            ]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        # Falls through to json.dumps
        self.assertIn("a.py", result.tool_calls[0].input_text)

    def test_tool_result_paired(self):
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tool_1", "name": "Bash",
                 "input": {"command": "echo hi"}}
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tool_1",
                 "content": "hi"}
            ]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].output_text, "hi")

    def test_tool_result_list_content(self):
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t1", "name": "Bash",
                 "input": {"command": "cat file.txt"}}
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}
            ]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(result.tool_calls[0].output_text, "line1\nline2")

    def test_full_workflow(self):
        """Full realistic stream: system → thinking → tool_use → tool_result → text → result."""
        raw = self._build_stream(
            {"type": "system", "subtype": "init", "session_id": "abc123"},
            {"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "I need to check the directory"}
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t1", "name": "Bash",
                 "input": {"command": "ls /project"}}
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": "README.md\nsrc/\ntests/"}
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "项目包含 README、src 和 tests 目录"}
            ]}},
            {"type": "result", "result": "", "session_id": "sess_xyz"}
        )
        result = ClaudeCodeBridge._parse_stream(raw)

        self.assertEqual(result.thinking, ["I need to check the directory"])
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "Bash")
        self.assertEqual(result.tool_calls[0].input_text, "ls /project")
        self.assertEqual(result.tool_calls[0].output_text, "README.md\nsrc/\ntests/")
        self.assertEqual(result.assistant_texts, ["项目包含 README、src 和 tests 目录"])
        self.assertEqual(result.result_text, "")
        self.assertEqual(result.session_id, "sess_xyz")
        # reply_text should use assistant_texts since result is empty
        self.assertEqual(result.reply_text, "项目包含 README、src 和 tests 目录")

    def test_multiple_tool_calls(self):
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t1", "name": "Read",
                 "input": {"file_path": "a.py"}}
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "print('a')"}
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t2", "name": "Read",
                 "input": {"file_path": "b.py"}}
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": "print('b')"}
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Both files contain print statements."}
            ]}},
            {"type": "result", "result": "", "session_id": "s1"}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].output_text, "print('a')")
        self.assertEqual(result.tool_calls[1].output_text, "print('b')")

    def test_mixed_content_in_single_message(self):
        """A single assistant message with both text and tool_use blocks."""
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Let me check that."},
                {"type": "tool_use", "id": "t1", "name": "Bash",
                 "input": {"command": "pwd"}}
            ]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(result.assistant_texts, ["Let me check that."])
        self.assertEqual(len(result.tool_calls), 1)

    def test_session_id_extraction(self):
        raw = self._build_stream(
            {"type": "result", "result": "", "session_id": "sess_123"}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(result.session_id, "sess_123")

    def test_session_id_from_system_init(self):
        raw = self._build_stream(
            {"type": "system", "subtype": "init", "session_id": "sess_init"}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(result.session_id, "sess_init")

    def test_noise_lines_ignored(self):
        raw = "=== Claude Code ===\nLoading...\n" + self._build_stream(
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi"}]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(result.assistant_texts, ["Hi"])

    def test_no_events_returns_empty(self):
        result = ClaudeCodeBridge._parse_stream("")
        self.assertTrue(result.is_empty)
        self.assertIsNone(result.session_id)


# ===========================================================================
# _format_result
# ===========================================================================

class TestFormatResult(unittest.TestCase):
    """Tests for FeishuBot._format_result rich formatting."""

    def test_format_stream_event_variants(self):
        self.assertIn("💭 Thinking", FeishuBot._format_stream_event(StreamEvent(kind="thinking", text="analyzing")))
        self.assertIn("🔧 Tool Use: Bash", FeishuBot._format_stream_event(StreamEvent(kind="tool_use", tool_name="Bash", text="pwd")))
        self.assertIn("📦 Tool Result: Bash", FeishuBot._format_stream_event(StreamEvent(kind="tool_result", tool_name="Bash", text="/repo")))
        self.assertIn("✅ Result", FeishuBot._format_stream_event(StreamEvent(kind="result", text="done")))
        self.assertEqual("reply", FeishuBot._format_stream_event(StreamEvent(kind="text", text="reply")))

    def test_format_stream_transcript_joins_blocks(self):
        output = FeishuBot._format_stream_transcript(["a", "b"])
        self.assertIn("a", output)
        self.assertIn("---", output)
        self.assertIn("b", output)

    def test_reply_only(self):
        r = StreamResult(assistant_texts=["Simple answer"])
        output = FeishuBot._format_result(r)
        self.assertEqual(output, "Simple answer")

    def test_thinking_shown(self):
        r = StreamResult(
            thinking=["Analyzing the question..."],
            assistant_texts=["The answer is 42."]
        )
        output = FeishuBot._format_result(r)
        self.assertIn("💭 Thinking", output)
        self.assertIn("Analyzing the question", output)
        self.assertIn("The answer is 42.", output)

    def test_tool_calls_shown(self):
        r = StreamResult(
            tool_calls=[
                ToolCall(name="Bash", input_text="ls -la", output_text="total 5\nfoo\nbar"),
                ToolCall(name="Read", input_text="/src/main.py", output_text="import os"),
            ],
            assistant_texts=["Found 2 files."]
        )
        output = FeishuBot._format_result(r)
        self.assertIn("🔧 Tool Calls (2)", output)
        self.assertIn("1. Bash", output)
        self.assertIn("ls -la", output)
        self.assertIn("2. Read", output)
        self.assertIn("/src/main.py", output)
        self.assertIn("Found 2 files.", output)
        # Separator between process and reply
        self.assertIn("---", output)

    def test_full_format(self):
        r = StreamResult(
            thinking=["Need to check files"],
            tool_calls=[ToolCall(name="Bash", input_text="ls", output_text="a.py b.py")],
            assistant_texts=["Project has 2 files."]
        )
        output = FeishuBot._format_result(r)
        # All sections present in order
        thinking_pos = output.index("💭 Thinking")
        tools_pos = output.index("🔧 Tool Calls")
        reply_pos = output.index("Project has 2 files.")
        self.assertLess(thinking_pos, tools_pos)
        self.assertLess(tools_pos, reply_pos)

    def test_empty_result(self):
        r = StreamResult()
        output = FeishuBot._format_result(r)
        self.assertIn("未生成文字回复", output)

    def test_tool_calls_only_no_text(self):
        """Tools ran but no text response — should show tools + fallback."""
        r = StreamResult(
            tool_calls=[ToolCall(name="Bash", input_text="make build", output_text="OK")]
        )
        output = FeishuBot._format_result(r)
        self.assertIn("🔧 Tool Calls", output)
        # is_empty is False because tool_calls exist, but reply_text is empty
        # So we show tool calls but no fallback message
        self.assertIn("make build", output)

    def test_thinking_truncated(self):
        r = StreamResult(
            thinking=["x" * 5000],
            assistant_texts=["Done"]
        )
        output = FeishuBot._format_result(r)
        self.assertIn("thinking truncated", output)

    def test_tool_calls_overflow(self):
        """More than MAX_TOOL_CALLS_SHOWN shows '... and N more'."""
        calls = [ToolCall(name=f"Tool{i}", input_text=f"cmd{i}") for i in range(15)]
        r = StreamResult(tool_calls=calls, assistant_texts=["Done"])
        output = FeishuBot._format_result(r)
        self.assertIn("5 more tool calls", output)

    def test_long_tool_input_truncated(self):
        r = StreamResult(
            tool_calls=[ToolCall(name="Bash", input_text="x" * 1000, output_text="ok")],
            assistant_texts=["Done"]
        )
        output = FeishuBot._format_result(r)
        self.assertIn("...", output)


# ===========================================================================
# _build_cmd
# ===========================================================================

class TestBuildCmd(unittest.TestCase):
    """Tests for ClaudeCodeBridge._build_cmd."""

    def setUp(self):
        self.bridge = ClaudeCodeBridge({"model": "gpt-5.4", "target": "claude"})

    def test_basic_cmd_structure(self):
        cmd = self.bridge._build_cmd("hello", None)
        self.assertIn("ttadk", cmd[0])
        self.assertIn("code", cmd)
        self.assertIn("-t", cmd)

    def test_stream_json_and_verbose(self):
        cmd = self.bridge._build_cmd("hello", None)
        args_str = " ".join(cmd)
        self.assertIn("--output-format stream-json", args_str)
        self.assertIn("--verbose", args_str)

    def test_no_system_prompt(self):
        """--system-prompt should NOT be in the command (breaks Claude Code)."""
        cmd = self.bridge._build_cmd("hello", None)
        args_str = " ".join(cmd)
        self.assertNotIn("--system-prompt", args_str)

    def test_session_resume(self):
        cmd = self.bridge._build_cmd("hello", "sess_123")
        args_str = " ".join(cmd)
        self.assertIn("--resume sess_123", args_str)

    def test_no_resume_without_session(self):
        cmd = self.bridge._build_cmd("hello", None)
        args_str = " ".join(cmd)
        self.assertNotIn("--resume", args_str)

    def test_allowed_tools_in_args(self):
        bridge = ClaudeCodeBridge({"allowed_tools": "Bash,Read"})
        cmd = bridge._build_cmd("hello", None)
        args_str = " ".join(cmd)
        self.assertIn("--allowedTools Bash,Read", args_str)

    def test_newline_escaping(self):
        cmd = self.bridge._build_cmd("line1\nline2", None)
        args_str = " ".join(cmd)
        self.assertNotIn("\n", args_str)

    def test_single_quote_escaping(self):
        cmd = self.bridge._build_cmd("it's a test", None)
        args_str = " ".join(cmd)
        self.assertIn("'\"'\"'", args_str)


class TestStreamingHelpers(unittest.TestCase):

    def test_consume_stream_line_emits_text_event(self):
        state = StreamState()
        events = ClaudeCodeBridge._consume_stream_line(
            state,
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}}),
        )
        self.assertEqual([event["type"] for event in events], ["stream_event"])
        self.assertEqual(events[0]["event"].kind, "text")
        self.assertEqual(events[0]["event"].text, "Hello")
        self.assertEqual(state.result.assistant_texts, ["Hello"])

    def test_consume_stream_line_emits_thinking_tool_and_result_events(self):
        state = StreamState()
        events = ClaudeCodeBridge._consume_stream_line(
            state,
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "Need to inspect"},
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pwd"}},
            ]}}),
        )
        self.assertEqual([event["event"].kind for event in events], ["thinking", "tool_use"])
        result_events = ClaudeCodeBridge._consume_stream_line(
            state,
            json.dumps({"type": "assistant", "message": {"content": [{
                "type": "tool_result", "tool_use_id": "t1", "content": "repo"
            }]}}),
        )
        self.assertEqual(result_events[0]["event"].kind, "tool_result")
        self.assertEqual(result_events[0]["event"].tool_name, "Bash")
        self.assertEqual(state.result.tool_calls[0].output_text, "repo")

    def test_consume_stream_line_emits_nonempty_result_event(self):
        state = StreamState()
        events = ClaudeCodeBridge._consume_stream_line(
            state,
            json.dumps({"type": "result", "result": "Done", "session_id": "sess_1"}),
        )
        self.assertEqual([event["type"] for event in events], ["stream_event", "session"])
        self.assertEqual(events[0]["event"].kind, "result")
        self.assertEqual(events[0]["event"].text, "Done")
        self.assertEqual(state.result.session_id, "sess_1")


# ===========================================================================
# ask() — mocked subprocess
# ===========================================================================

class TestBridgeAskMocked(unittest.TestCase):
    """Test ask() with mocked subprocess."""

    def test_missing_cmd_returns_error(self):
        bridge = ClaudeCodeBridge({"ttadk_cmd": "nonexistent_cmd_99"})
        result = asyncio.run(bridge.ask("t1", "hi"))
        self.assertIsInstance(result, StreamResult)
        self.assertIn("not found", result.reply_text)


class TestSessionPersistence(unittest.TestCase):

    def test_load_sessions_from_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"topic1": "sess_a"}, f)
            f.flush()
            bridge = ClaudeCodeBridge({"session_store_path": f.name})
        self.assertEqual(bridge._sessions, {"topic1": "sess_a"})
        os.unlink(f.name)

    def test_invalid_session_file_returns_empty(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("not-json")
            f.flush()
            bridge = ClaudeCodeBridge({"session_store_path": f.name})
        self.assertEqual(bridge._sessions, {})
        os.unlink(f.name)

    def test_remember_session_persists(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)
        bridge = ClaudeCodeBridge({"session_store_path": path})
        asyncio.run(bridge._remember_session("topic1", "sess_a"))
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"topic1": "sess_a"})
        os.unlink(path)


# ===========================================================================
# Session Management
# ===========================================================================

class TestSessionManagement(unittest.TestCase):
    """Test session_id tracking across topics."""

    def test_session_stored(self):
        bridge = ClaudeCodeBridge({})
        bridge._sessions["topic1"] = "sess_a"
        self.assertEqual(bridge._sessions.get("topic1"), "sess_a")

    def test_different_topics(self):
        bridge = ClaudeCodeBridge({})
        bridge._sessions["topic1"] = "sess_a"
        bridge._sessions["topic2"] = "sess_b"
        self.assertNotEqual(bridge._sessions["topic1"], bridge._sessions["topic2"])


# ===========================================================================
# Config / Dotenv / Extract Text
# ===========================================================================

class TestFeishuStreaming(unittest.IsolatedAsyncioTestCase):

    async def test_handle_stream_events_reply_one_card_per_event(self):
        bridge = ClaudeCodeBridge({})
        bot = FeishuBot({"app_id": "app", "app_secret": "secret"}, bridge)

        async def fake_stream_ask(topic_id, prompt):
            yield {"type": "stream_event", "event": StreamEvent(kind="thinking", text="先思考")}
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_use", tool_name="Bash", text="pwd")}
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_result", tool_name="Bash", text="/repo")}
            yield {"type": "stream_event", "event": StreamEvent(kind="text", text="最终回复")}
            yield {"type": "final", "result": StreamResult(assistant_texts=["最终回复"])}

        bot.bridge.stream_ask = fake_stream_ask
        bot._reply_card = AsyncMock(return_value="card_1")
        bot._react = AsyncMock()

        await bot._handle("chat", "msg", "topic", "hello")

        self.assertEqual(bot._reply_card.await_count, 4)
        rendered_contents = [call.args[1] for call in bot._reply_card.await_args_list]
        self.assertIn("💭 Thinking", rendered_contents[0])
        self.assertIn("🔧 Tool Use: Bash", rendered_contents[1])
        self.assertIn("📦 Tool Result: Bash", rendered_contents[2])
        self.assertEqual("最终回复", rendered_contents[3])
        self.assertEqual(bot._react.await_count, 2)

    async def test_handle_skips_duplicate_result_event(self):
        bridge = ClaudeCodeBridge({})
        bot = FeishuBot({"app_id": "app", "app_secret": "secret"}, bridge)

        async def fake_stream_ask(topic_id, prompt):
            yield {"type": "stream_event", "event": StreamEvent(kind="text", text="同一段回复")}
            yield {"type": "stream_event", "event": StreamEvent(kind="result", text="同一段回复")}
            yield {"type": "final", "result": StreamResult(assistant_texts=["同一段回复"], result_text="同一段回复")}

        bot.bridge.stream_ask = fake_stream_ask
        bot._reply_card = AsyncMock(return_value="card_1")
        bot._react = AsyncMock()

        await bot._handle("chat", "msg", "topic", "hello")

        self.assertEqual(bot._reply_card.await_count, 1)
        self.assertNotIn("✅ Result", bot._reply_card.await_args.args[1])

    async def test_handle_without_stream_text_appends_final_reply(self):
        bridge = ClaudeCodeBridge({})
        bot = FeishuBot({"app_id": "app", "app_secret": "secret"}, bridge)

        async def fake_stream_ask(topic_id, prompt):
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_use", tool_name="Bash", text="git push origin main")}
            yield {"type": "final", "result": StreamResult(assistant_texts=["已推送到 origin/main"])}

        bot.bridge.stream_ask = fake_stream_ask
        bot._reply_card = AsyncMock(return_value="card_1")
        bot._react = AsyncMock()

        await bot._handle("chat", "msg", "topic", "hello")

        rendered_contents = [call.args[1] for call in bot._reply_card.await_args_list]
        self.assertIn("🔧 Tool Use: Bash", rendered_contents[0])
        self.assertIn("已推送到 origin/main", rendered_contents[-1])

    async def test_handle_without_stream_events_falls_back_to_final_result(self):
        bridge = ClaudeCodeBridge({})
        bot = FeishuBot({"app_id": "app", "app_secret": "secret"}, bridge)

        async def fake_stream_ask(topic_id, prompt):
            yield {"type": "final", "result": StreamResult(tool_calls=[ToolCall(name="Bash", input_text="pwd")])}

        bot.bridge.stream_ask = fake_stream_ask
        bot._reply_card = AsyncMock(return_value="card_1")
        bot._react = AsyncMock()

        await bot._handle("chat", "msg", "topic", "hello")

        bot._reply_card.assert_awaited_once()
        self.assertIn("🔧 Tool Calls", bot._reply_card.await_args.args[1])


class TestLoadDotenv(unittest.TestCase):

    def test_basic_vars(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("FOO=bar\nBAZ=qux\n")
            f.flush()
            os.environ.pop("FOO", None)
            os.environ.pop("BAZ", None)
            load_dotenv(f.name)
        self.assertEqual(os.environ.get("FOO"), "bar")
        self.assertEqual(os.environ.get("BAZ"), "qux")
        os.environ.pop("FOO", None)
        os.environ.pop("BAZ", None)

    def test_quoted_values(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write('KEY="hello world"\n')
            f.flush()
            os.environ.pop("KEY", None)
            load_dotenv(f.name)
        self.assertEqual(os.environ.get("KEY"), "hello world")
        os.environ.pop("KEY", None)

    def test_comments_and_blanks(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("# comment\n\nVAR=val\n")
            f.flush()
            os.environ.pop("VAR", None)
            load_dotenv(f.name)
        self.assertEqual(os.environ.get("VAR"), "val")
        os.environ.pop("VAR", None)

    def test_existing_env_not_overridden(self):
        os.environ["EXISTING"] = "original"
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("EXISTING=new\n")
            f.flush()
            load_dotenv(f.name)
        self.assertEqual(os.environ.get("EXISTING"), "original")
        os.environ.pop("EXISTING", None)

    def test_missing_file_no_error(self):
        load_dotenv("/nonexistent/.env")


class TestLoadConfig(unittest.TestCase):

    def test_env_substitution(self):
        import tempfile
        os.environ["TEST_APP_ID"] = "myid"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("feishu:\n  app_id: ${TEST_APP_ID}\n")
            f.flush()
            cfg = load_config(f.name)
        self.assertEqual(cfg["feishu"]["app_id"], "myid")
        os.environ.pop("TEST_APP_ID", None)

    def test_missing_env_becomes_empty(self):
        import tempfile
        os.environ.pop("UNSET_VAR_XYZ", None)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("key: ${UNSET_VAR_XYZ}\n")
            f.flush()
            cfg = load_config(f.name)
        self.assertFalse(cfg["key"])


class TestExtractText(unittest.TestCase):

    def test_plain_text(self):
        self.assertEqual(FeishuBot._extract_text({"text": "hello"}), "hello")

    def test_rich_text_single_paragraph(self):
        content = {"content": [[{"tag": "text", "text": "Hi"}, {"tag": "text", "text": " there"}]]}
        self.assertEqual(FeishuBot._extract_text(content), "Hi  there")

    def test_rich_text_multiple_paragraphs(self):
        content = {"content": [
            [{"tag": "text", "text": "Para 1"}],
            [{"tag": "text", "text": "Para 2"}],
        ]}
        self.assertEqual(FeishuBot._extract_text(content), "Para 1\n\nPara 2")

    def test_at_mention_included(self):
        content = {"content": [[{"tag": "at", "text": "@bot"}, {"tag": "text", "text": " help"}]]}
        self.assertIn("help", FeishuBot._extract_text(content))

    def test_non_text_tags_ignored(self):
        content = {"content": [[{"tag": "img", "src": "http://x"}, {"tag": "text", "text": "ok"}]]}
        self.assertEqual(FeishuBot._extract_text(content), "ok")

    def test_empty_content(self):
        self.assertEqual(FeishuBot._extract_text({}), "")


class TestCard(unittest.TestCase):

    def test_card_structure(self):
        card = json.loads(FeishuBot._card("hello"))
        self.assertTrue(card["config"]["wide_screen_mode"])
        self.assertEqual(card["elements"][0]["content"], "hello")

    def test_card_with_special_chars(self):
        card = json.loads(FeishuBot._card('test "quotes" & <tags>'))
        self.assertIn("quotes", card["elements"][0]["content"])


if __name__ == "__main__":
    unittest.main()
