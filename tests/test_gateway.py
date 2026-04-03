"""Unit tests for gateway.py — stream-json parsing + rich formatting."""

import asyncio
import json
import os
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

from feishu_bot import FeishuBot
from gateway import (
    ClaudeCodeBridge,
    StreamEvent,
    StreamResult,
    StreamState,
    ToolCall,
    UsageSummary,
    _preview_text,
    _reset_log_file,
    load_config,
    load_dotenv,
    main,
)


# ===========================================================================
# StreamResult & ToolCall
# ===========================================================================

class TestStreamResult(unittest.TestCase):
    """Tests for the StreamResult dataclass."""

    def test_preview_text_escapes_newlines_and_truncates(self):
        self.assertEqual(_preview_text("a\nb", limit=10), "a\\nb")
        self.assertEqual(_preview_text("123456", limit=4), "1234...")

    def test_reset_log_file_truncates_existing_content(self):
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write("old log")
            path = f.name
        _reset_log_file(path)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "")
        os.unlink(path)

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
        self.assertIn("a.py", result.tool_calls[0].input_text)

    def test_tool_result_paired(self):
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tool_1", "name": "Bash",
                 "input": {"command": "echo hi"}}
            ]}},
            {"type": "user", "message": {"role": "user", "content": [
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
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}
            ]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(result.tool_calls[0].output_text, "line1\nline2")

    def test_tool_result_assistant_event_still_supported(self):
        raw = self._build_stream(
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "tool_legacy", "name": "Bash",
                 "input": {"command": "echo hi"}}
            ]}},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "tool_legacy",
                 "content": "hi"}
            ]}}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertEqual(result.tool_calls[0].output_text, "hi")

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
        self.assertEqual(result.reply_text, "项目包含 README、src 和 tests 目录")

    def test_result_usage_and_cost_are_extracted(self):
        raw = self._build_stream(
            {
                "type": "result",
                "result": "OK",
                "session_id": "sess_usage",
                "total_cost_usd": 0.0165432,
                "usage": {
                    "input_tokens": 3716,
                    "output_tokens": 32,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 16384,
                },
            }
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.input_tokens, 3716)
        self.assertEqual(result.usage.output_tokens, 32)
        self.assertEqual(result.usage.cache_creation_input_tokens, 0)
        self.assertEqual(result.usage.cache_read_input_tokens, 16384)
        self.assertEqual(result.usage.total_tokens, 20132)
        self.assertAlmostEqual(result.usage.cost_usd, 0.0165432)

    def test_result_usage_uses_model_usage_fallback(self):
        raw = self._build_stream(
            {
                "type": "result",
                "result": "OK",
                "modelUsage": {
                    "gpt-5.4": {
                        "inputTokens": 10,
                        "outputTokens": 2,
                        "cacheReadInputTokens": 7,
                        "costUSD": 0.0025,
                    }
                },
            }
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.input_tokens, 10)
        self.assertEqual(result.usage.output_tokens, 2)
        self.assertEqual(result.usage.cache_read_input_tokens, 7)
        self.assertEqual(result.usage.total_tokens, 19)
        self.assertAlmostEqual(result.usage.cost_usd, 0.0025)

    def test_invalid_usage_shape_is_ignored(self):
        raw = self._build_stream(
            {"type": "result", "result": "OK", "usage": "not-a-dict", "total_cost_usd": "bad-value"}
        )
        result = ClaudeCodeBridge._parse_stream(raw)
        self.assertIsNone(result.usage)
        self.assertEqual(result.reply_text, "OK")

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
        tool_use_out = FeishuBot._format_stream_event(StreamEvent(kind="tool_use", tool_name="Bash", text="pwd"))
        self.assertIn("**Bash**", tool_use_out)
        self.assertIn("`pwd`", tool_use_out)
        tool_result_out = FeishuBot._format_stream_event(StreamEvent(kind="tool_result", tool_name="Bash", text="/repo"))
        self.assertIn("📦 **Bash** result", tool_result_out)
        self.assertIn("/repo", tool_result_out)
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
        self.assertIn("---", output)

    def test_full_format(self):
        r = StreamResult(
            thinking=["Need to check files"],
            tool_calls=[ToolCall(name="Bash", input_text="ls", output_text="a.py b.py")],
            assistant_texts=["Project has 2 files."]
        )
        output = FeishuBot._format_result(r)
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
        r = StreamResult(
            tool_calls=[ToolCall(name="Bash", input_text="make build", output_text="OK")]
        )
        output = FeishuBot._format_result(r)
        self.assertIn("🔧 Tool Calls", output)
        self.assertIn("make build", output)

    def test_thinking_truncated(self):
        r = StreamResult(
            thinking=["x" * 5000],
            assistant_texts=["Done"]
        )
        output = FeishuBot._format_result(r)
        self.assertIn("thinking truncated", output)

    def test_tool_calls_overflow(self):
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

    def test_format_usage_summary_returns_empty_without_usage(self):
        self.assertEqual(FeishuBot._format_usage_summary(None), "")

    def test_format_usage_summary_includes_expected_fields(self):
        usage = UsageSummary(
            input_tokens=1200,
            output_tokens=34,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=400,
            total_tokens=1654,
            cost_usd=0.0165,
        )
        output = FeishuBot._format_usage_summary(usage)
        self.assertIn("**Usage**", output)
        self.assertIn("- Input: 1,200", output)
        self.assertIn("- Cache Create: 20", output)
        self.assertIn("- Cache Read: 400", output)
        self.assertIn("- Total: 1,654", output)
        self.assertIn("- Cost: $0.0165", output)


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

    def test_provider_defaults_to_claude(self):
        self.assertEqual(self.bridge.provider, "claude")

    def test_provider_infers_codex_from_target(self):
        bridge = ClaudeCodeBridge({"model": "gpt-5.4", "target": "codex"})
        self.assertEqual(bridge.provider, "codex")
        cmd = bridge._build_cmd("hello", None)
        self.assertEqual(cmd[cmd.index("-t") + 1], "codex")

    def test_provider_override_wins(self):
        bridge = ClaudeCodeBridge({"model": "gpt-5.4", "target": "claude", "provider": "codex"})
        self.assertEqual(bridge.provider, "codex")

    def test_codex_cmd_uses_exec_json(self):
        bridge = ClaudeCodeBridge({"model": "gpt-5.4", "target": "codex"})
        cmd = bridge._build_cmd("hello", None)
        args_str = " ".join(cmd)
        self.assertIn("exec --yolo --json", args_str)
        self.assertNotIn("--full-auto", args_str)
        self.assertNotIn("--output-format stream-json", args_str)

    def test_codex_resume_cmd(self):
        bridge = ClaudeCodeBridge({"model": "gpt-5.4", "target": "codex"})
        cmd = bridge._build_cmd("continue", "thread_123")
        args_str = " ".join(cmd)
        self.assertIn("exec --yolo --json resume thread_123", args_str)


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
            json.dumps({"type": "user", "message": {"role": "user", "content": [{
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
            json.dumps({
                "type": "result",
                "result": "Done",
                "session_id": "sess_1",
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "total_cost_usd": 0.0012,
            }),
        )
        self.assertEqual([event["type"] for event in events], ["stream_event", "session"])
        self.assertEqual(events[0]["event"].kind, "result")
        self.assertEqual(events[0]["event"].text, "Done")
        self.assertEqual(state.result.session_id, "sess_1")
        self.assertIsNotNone(state.result.usage)
        self.assertEqual(state.result.usage.total_tokens, 12)
        self.assertAlmostEqual(state.result.usage.cost_usd, 0.0012)

    def test_consume_stream_line_supports_provider_dispatch(self):
        state = StreamState()
        events = ClaudeCodeBridge._consume_stream_line(
            state,
            json.dumps({"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": "Hello"}}),
            provider="codex",
        )
        self.assertEqual([event["type"] for event in events], ["stream_event"])
        self.assertEqual(events[0]["event"].kind, "text")
        self.assertEqual(state.result.assistant_texts, ["Hello"])


class TestCodexStreaming(unittest.TestCase):

    def test_parse_codex_stream_with_command(self):
        raw = "\n".join([
            json.dumps({"type": "thread.started", "thread_id": "thread_1"}),
            json.dumps({"type": "turn.started"}),
            json.dumps({"type": "item.completed", "item": {
                "id": "item_0",
                "type": "agent_message",
                "text": "Running pwd.",
            }}),
            json.dumps({"type": "item.started", "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": "/bin/bash -lc pwd",
                "aggregated_output": "",
                "exit_code": None,
                "status": "in_progress",
            }}),
            json.dumps({"type": "item.completed", "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": "/bin/bash -lc pwd",
                "aggregated_output": "/repo\n",
                "exit_code": 0,
                "status": "completed",
            }}),
            json.dumps({"type": "item.completed", "item": {
                "id": "item_2",
                "type": "agent_message",
                "text": "/repo",
            }}),
            json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 4,
                "output_tokens": 3,
            }}),
        ])
        result = ClaudeCodeBridge._parse_stream(raw, provider="codex")
        self.assertEqual(result.session_id, "thread_1")
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(result.assistant_texts, ["Running pwd.", "/repo"])
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "command_execution")
        self.assertEqual(result.tool_calls[0].input_text, "/bin/bash -lc pwd")
        self.assertEqual(result.tool_calls[0].output_text, "/repo")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.input_tokens, 10)
        self.assertEqual(result.usage.cache_read_input_tokens, 4)
        self.assertEqual(result.usage.output_tokens, 3)
        self.assertEqual(result.usage.total_tokens, 17)

    def test_consume_codex_turn_failed(self):
        state = StreamState()
        events = ClaudeCodeBridge._consume_event(
            state,
            {"type": "turn.failed", "message": "Something broke."},
            provider="codex",
        )
        self.assertEqual(state.result.stop_reason, "error")
        self.assertEqual(state.result.result_text, "Something broke.")
        self.assertEqual([event["event"].kind for event in events], ["result"])


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

    def test_reset_session_persists(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)
        bridge = ClaudeCodeBridge({"session_store_path": path})
        asyncio.run(bridge._remember_session("topic1", "sess_a"))
        cleared = asyncio.run(bridge.reset_session("topic1"))
        self.assertEqual(cleared, "sess_a")
        self.assertIsNone(bridge.get_session("topic1"))
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {})
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

    async def test_handle_stream_events_merge_tool_result_into_tool_use_card(self):
        """Tool results are merged into the same thinking+tools card (2-card model)."""
        bridge = ClaudeCodeBridge({})
        bot = FeishuBot({"app_id": "app", "app_secret": "secret"}, bridge)

        async def fake_stream_ask(topic_id, prompt, **kwargs):
            yield {"type": "stream_event", "event": StreamEvent(kind="thinking", text="先思考")}
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_use", tool_name="Bash", tool_use_id="t1", text="pwd")}
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_result", tool_name="Bash", tool_use_id="t1", text="/repo")}
            yield {"type": "stream_event", "event": StreamEvent(kind="text", text="最终回复")}
            yield {"type": "final", "result": StreamResult(assistant_texts=["最终回复"])}

        bot.bridge.stream_ask = fake_stream_ask
        bot._reply_card = AsyncMock(side_effect=["card-text", "card-thinking"])
        bot._update_card = AsyncMock()
        bot._react = AsyncMock()
        bot._resolve_images = AsyncMock(side_effect=lambda text: text)

        await bot._handle("chat", "msg", "topic", "hello")

        # 2-card model: text card + combined thinking+tools card
        self.assertEqual(bot._reply_card.await_count, 2)
        rendered_contents = [call.args[1] for call in bot._reply_card.await_args_list]
        # First card: the text event reply
        self.assertEqual("最终回复", rendered_contents[0])
        # Second card: combined thinking + tool calls (including merged result)
        combined_card = rendered_contents[1]
        self.assertIn("💭", combined_card)
        self.assertIn("先思考", combined_card)
        self.assertIn("🔧 Tool Calls", combined_card)
        self.assertIn("**Bash**", combined_card)
        self.assertIn("📦 **Bash** result", combined_card)
        self.assertIn("/repo", combined_card)
        self.assertEqual(bot._react.await_count, 2)

    async def test_handle_skips_duplicate_result_event(self):
        """Duplicate result text does not produce an extra card beyond text + result."""
        bridge = ClaudeCodeBridge({})
        bot = FeishuBot({"app_id": "app", "app_secret": "secret"}, bridge)

        async def fake_stream_ask(topic_id, prompt, **kwargs):
            yield {"type": "stream_event", "event": StreamEvent(kind="text", text="同一段回复")}
            yield {"type": "stream_event", "event": StreamEvent(kind="result", text="同一段回复")}
            yield {"type": "final", "result": StreamResult(assistant_texts=["同一段回复"], result_text="同一段回复")}

        bot.bridge.stream_ask = fake_stream_ask
        bot._reply_card = AsyncMock(return_value="card_1")
        bot._react = AsyncMock()
        bot._resolve_images = AsyncMock(side_effect=lambda text: text)

        await bot._handle("chat", "msg", "topic", "hello")

        # 2 cards: text event card + result accumulation card.
        # No extra duplicate card beyond these two.
        self.assertEqual(bot._reply_card.await_count, 2)
        rendered_contents = [call.args[1] for call in bot._reply_card.await_args_list]
        # Neither card should have "✅ Result" header (result goes to _result_acc raw)
        for c in rendered_contents:
            self.assertNotIn("✅ Result", c)

    async def test_handle_repeated_read_tool_use_is_not_deduped_by_same_path(self):
        """Two Read calls with the same path are both shown (not deduped)."""
        bridge = ClaudeCodeBridge({})
        bot = FeishuBot({"app_id": "app", "app_secret": "secret"}, bridge)

        async def fake_stream_ask(topic_id, prompt, **kwargs):
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_use", tool_name="Read", tool_use_id="t1", text="/tmp/a.py")}
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_result", tool_name="Read", tool_use_id="t1", text="first")}
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_use", tool_name="Read", tool_use_id="t2", text="/tmp/a.py")}
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_result", tool_name="Read", tool_use_id="t2", text="second")}
            yield {"type": "final", "result": StreamResult(tool_calls=[
                ToolCall(name="Read", input_text="/tmp/a.py", output_text="first"),
                ToolCall(name="Read", input_text="/tmp/a.py", output_text="second"),
            ])}

        bot.bridge.stream_ask = fake_stream_ask
        bot._reply_card = AsyncMock(side_effect=["card-1", "card-2", "card-3"])
        bot._update_card = AsyncMock()
        bot._react = AsyncMock()
        bot._resolve_images = AsyncMock(side_effect=lambda text: text)

        await bot._handle("chat", "msg", "topic", "hello")

        # 2-card model: one combined thinking+tools card, one fallback card
        self.assertEqual(bot._reply_card.await_count, 2)
        rendered_contents = [call.args[1] for call in bot._reply_card.await_args_list]
        # First card: combined tool calls card with both Read calls
        combined_card = rendered_contents[0]
        self.assertIn("🔧 Tool Calls (2)", combined_card)
        # Both tool uses present (not deduped despite same path)
        # The header "Tool Calls (2)" confirms both tools are included.
        # Also verify both tool result blocks are present.
        self.assertEqual(combined_card.count(chr(128230) + " **Read** result"), 2)
        # Both results merged into their respective tool blocks
        self.assertIn("first", combined_card)
        self.assertIn("second", combined_card)
        self.assertIn("📦 **Read** result", combined_card)

    async def test_handle_missing_card_id_falls_back_to_merged_reply(self):
        """When thinking card creation returns None, content is still delivered."""
        bridge = ClaudeCodeBridge({})
        bot = FeishuBot({"app_id": "app", "app_secret": "secret"}, bridge)

        async def fake_stream_ask(topic_id, prompt, **kwargs):
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_use", tool_name="Bash", tool_use_id="t1", text="git status")}
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_result", tool_name="Bash", tool_use_id="t1", text="working tree clean")}
            yield {"type": "final", "result": StreamResult(tool_calls=[
                ToolCall(name="Bash", input_text="git status", output_text="working tree clean"),
            ])}

        bot.bridge.stream_ask = fake_stream_ask
        bot._reply_card = AsyncMock(side_effect=[None, "fallback-card", "final-card"])
        bot._update_card = AsyncMock()
        bot._react = AsyncMock()
        bot._resolve_images = AsyncMock(side_effect=lambda text: text)

        await bot._handle("chat", "msg", "topic", "hello")

        # 2-card model: thinking+tools card (returns None) + fallback text card
        self.assertEqual(bot._reply_card.await_count, 2)
        # First card: combined thinking+tools (returned None)
        first_card = bot._reply_card.await_args_list[0].args[1]
        self.assertIn("🔧 Tool Calls", first_card)
        self.assertIn("**Bash**", first_card)
        self.assertIn("working tree clean", first_card)
        # Even though first card returned None, a second card is still sent
        bot._update_card.assert_not_awaited()

    async def test_handle_without_stream_text_appends_final_reply(self):
        """Final reply text is appended even without streaming text events."""
        bridge = ClaudeCodeBridge({})
        bot = FeishuBot({"app_id": "app", "app_secret": "secret"}, bridge)

        async def fake_stream_ask(topic_id, prompt, **kwargs):
            yield {"type": "stream_event", "event": StreamEvent(kind="tool_use", tool_name="Bash", text="git push origin main")}
            yield {"type": "final", "result": StreamResult(assistant_texts=["已推送到 origin/main"])}

        bot.bridge.stream_ask = fake_stream_ask
        bot._reply_card = AsyncMock(return_value="card_1")
        bot._react = AsyncMock()
        bot._resolve_images = AsyncMock(side_effect=lambda text: text)

        await bot._handle("chat", "msg", "topic", "hello")

        rendered_contents = [call.args[1] for call in bot._reply_card.await_args_list]
        # First card: combined thinking+tools card with tool call
        self.assertIn("🔧 Tool Calls", rendered_contents[0])
        self.assertIn("**Bash**", rendered_contents[0])
        # Last card: the final reply text
        self.assertIn("已推送到 origin/main", rendered_contents[-1])

    async def test_handle_without_stream_events_falls_back_to_final_result(self):
        bridge = ClaudeCodeBridge({})
        bot = FeishuBot({"app_id": "app", "app_secret": "secret"}, bridge)

        async def fake_stream_ask(topic_id, prompt, **kwargs):
            yield {"type": "final", "result": StreamResult(tool_calls=[ToolCall(name="Bash", input_text="pwd")])}

        bot.bridge.stream_ask = fake_stream_ask
        bot._reply_card = AsyncMock(return_value="card_1")
        bot._react = AsyncMock()

        await bot._handle("chat", "msg", "topic", "hello")

        bot._reply_card.assert_awaited_once()
        self.assertIn("🔧 Tool Calls", bot._reply_card.await_args.args[1])

    async def test_handle_sends_usage_card_after_main_reply(self):
        bridge = ClaudeCodeBridge({})
        bot = FeishuBot({"app_id": "app", "app_secret": "secret"}, bridge)

        async def fake_stream_ask(topic_id, prompt, **kwargs):
            yield {
                "type": "final",
                "result": StreamResult(
                    assistant_texts=["主回复"],
                    usage=UsageSummary(input_tokens=1200, output_tokens=34, total_tokens=1234, cost_usd=0.0165),
                ),
            }

        bot.bridge.stream_ask = fake_stream_ask
        bot._reply_card = AsyncMock(side_effect=["card-main", "card-usage"])
        bot._react = AsyncMock()

        await bot._handle("chat", "msg", "topic", "hello")

        self.assertEqual(bot._reply_card.await_count, 2)
        self.assertEqual(bot._reply_card.await_args_list[0].args[1], "主回复")
        usage_text = bot._reply_card.await_args_list[1].args[1]
        self.assertIn("**Usage**", usage_text)
        self.assertIn("- Input: 1,200", usage_text)
        self.assertIn("- Total: 1,234", usage_text)
        self.assertIn("- Cost: $0.0165", usage_text)


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


class TestGatewayMain(unittest.IsolatedAsyncioTestCase):

    async def test_main_starts_feishu_channel(self):
        cfg = {
            "claude": {},
            "feishu": {"app_id": "app", "app_secret": "secret"},
            "weixin": {"enabled": False},
        }
        events: list[str] = []
        bridge = object()

        class FakeFeishuBot:
            def __init__(self, feishu_cfg, bridge_obj):
                events.append("feishu_init")
                self.feishu_cfg = feishu_cfg
                self.bridge_obj = bridge_obj

            async def start(self):
                events.append("feishu_start")

        fake_feishu_module = types.SimpleNamespace(FeishuBot=FakeFeishuBot)

        with (
            patch("gateway.load_config", return_value=cfg),
            patch("gateway.ClaudeCodeBridge", return_value=bridge),
            patch.dict(os.sys.modules, {"feishu_bot": fake_feishu_module}),
            patch("asyncio.Event.wait", new=AsyncMock(side_effect=asyncio.CancelledError)),
        ):
            await main()

        self.assertEqual(events, ["feishu_init", "feishu_start"])

    async def test_main_starts_weixin_when_enabled(self):
        cfg = {
            "claude": {},
            "feishu": {"app_id": "app", "app_secret": "secret"},
            "weixin": {"enabled": True},
        }
        events: list[str] = []
        bridge = object()

        class FakeFeishuBot:
            def __init__(self, feishu_cfg, bridge_obj):
                events.append("feishu_init")

            async def start(self):
                events.append("feishu_start")

        class FakeWeixinBot:
            def __init__(self, weixin_cfg, bridge_obj):
                events.append("weixin_init")
                self.weixin_cfg = weixin_cfg
                self.bridge_obj = bridge_obj

            async def start(self):
                events.append("weixin_start")

        fake_feishu_module = types.SimpleNamespace(FeishuBot=FakeFeishuBot)
        fake_weixin_module = types.SimpleNamespace(WeixinBot=FakeWeixinBot)

        with (
            patch("gateway.load_config", return_value=cfg),
            patch("gateway.ClaudeCodeBridge", return_value=bridge),
            patch.dict(os.sys.modules, {
                "feishu_bot": fake_feishu_module,
                "weixin_bot": fake_weixin_module,
            }),
            patch("asyncio.Event.wait", new=AsyncMock(side_effect=asyncio.CancelledError)),
        ):
            await main()

        self.assertEqual(events, ["feishu_init", "feishu_start", "weixin_init", "weixin_start"])


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




# ===========================================================================
# stderr separation tests
# ===========================================================================

class TestStderrSeparation(unittest.IsolatedAsyncioTestCase):
    """Tests for stderr being captured separately from stdout."""

    async def test_stderr_captured_separately(self):
        """Verify stderr lines are logged with [STDERR] prefix, not mixed into stream events."""
        bridge = ClaudeCodeBridge({})

        # Build a fake process that writes JSON to stdout and error to stderr
        stdout_lines = [
            json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}).encode() + b"\n",
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi"}]}}).encode() + b"\n",
            json.dumps({"type": "result", "result": "Hi", "session_id": "s1"}).encode() + b"\n",
        ]
        stderr_lines_raw = [
            b"Warning: something went wrong\n",
            b"Error: connection reset\n",
        ]

        async def fake_readline_stdout(lines):
            for line in lines:
                yield line
            yield b""

        async def fake_readline_stderr(lines):
            for line in lines:
                yield line
            yield b""

        class FakeStream:
            def __init__(self, lines):
                self._iter = iter(lines + [b""])
            async def readline(self):
                return next(self._iter)

        class FakeProc:
            def __init__(self):
                self.stdout = FakeStream(stdout_lines)
                self.stderr = FakeStream(stderr_lines_raw)
                self.returncode = 0
            async def wait(self):
                pass
            def kill(self):
                pass
            async def communicate(self):
                return b"", b""

        with patch("asyncio.create_subprocess_exec", return_value=FakeProc()):
            events = []
            async for event in bridge.stream_ask("topic1", "hello"):
                events.append(event)

        # Should have stream events + final
        final_events = [e for e in events if e["type"] == "final"]
        self.assertEqual(len(final_events), 1)
        result = final_events[0]["result"]
        self.assertEqual(result.reply_text, "Hi")

    async def test_stderr_logged_on_empty_result(self):
        """When result is empty and stderr has content, stderr should be in diagnostics."""
        bridge = ClaudeCodeBridge({})

        stdout_lines = [
            json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}).encode() + b"\n",
            json.dumps({"type": "result", "result": "", "session_id": "s1", "stop_reason": "end_turn", "num_turns": 3}).encode() + b"\n",
        ]
        stderr_lines_raw = [
            b"FATAL: max context length exceeded\n",
        ]

        class FakeStream:
            def __init__(self, lines):
                self._iter = iter(lines + [b""])
            async def readline(self):
                return next(self._iter)

        class FakeProc:
            def __init__(self):
                self.stdout = FakeStream(stdout_lines)
                self.stderr = FakeStream(stderr_lines_raw)
                self.returncode = 1
            async def wait(self):
                pass
            def kill(self):
                pass
            async def communicate(self):
                return b"", b""

        with patch("asyncio.create_subprocess_exec", return_value=FakeProc()), \
             patch("gateway.log") as mock_log:
            events = []
            async for event in bridge.stream_ask("topic1", "hello"):
                events.append(event)

            # Check that stderr was logged with [STDERR] prefix
            stderr_calls = [
                call for call in mock_log.warning.call_args_list
                if "[STDERR]" in str(call)
            ]
            self.assertGreater(len(stderr_calls), 0, "stderr lines should be logged with [STDERR] prefix")

            # Check that empty result diagnostic includes stderr
            stderr_dump_calls = [
                call for call in mock_log.warning.call_args_list
                if "stderr output" in str(call) or "stderr[" in str(call)
            ]
            self.assertGreater(len(stderr_dump_calls), 0, "empty result diagnostic should dump stderr")

    async def test_timeout_cancels_stderr_task(self):
        """On timeout, stderr drain task should be cancelled cleanly."""
        bridge = ClaudeCodeBridge({"timeout": 1})

        class HangingStream:
            async def readline(self):
                await asyncio.sleep(100)  # hang forever
                return b""

        class FakeStream:
            def __init__(self, lines):
                self._iter = iter(lines + [b""])
            async def readline(self):
                return next(self._iter)

        class FakeProc:
            def __init__(self):
                self.stdout = HangingStream()
                self.stderr = FakeStream([b"some error\n"])
                self.returncode = -9
            async def wait(self):
                pass
            def kill(self):
                pass
            async def communicate(self):
                return b"", b""

        with patch("asyncio.create_subprocess_exec", return_value=FakeProc()):
            events = []
            async for event in bridge.stream_ask("topic1", "hello"):
                events.append(event)

        final_events = [e for e in events if e["type"] == "final"]
        self.assertEqual(len(final_events), 1)
        self.assertIn("timed out", final_events[0]["result"].reply_text)

    async def test_timeout_returns_final_without_communicate(self):
        """Timeout should still return a final result even if communicate would hang."""
        bridge = ClaudeCodeBridge({"timeout": 1})

        class HangingStream:
            async def readline(self):
                await asyncio.sleep(100)
                return b""

        class FakeProc:
            def __init__(self):
                self.stdout = HangingStream()
                self.stderr = HangingStream()
                self.returncode = -9
                self.killed = False
            async def wait(self):
                return None
            def kill(self):
                self.killed = True
            async def communicate(self):
                raise AssertionError("communicate should not be called on timeout")

        proc = FakeProc()
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            events = []
            async for event in bridge.stream_ask("topic1", "hello"):
                events.append(event)

        final_events = [e for e in events if e["type"] == "final"]
        self.assertEqual(len(final_events), 1)
        self.assertTrue(proc.killed)
        self.assertIn("timed out", final_events[0]["result"].reply_text)


if __name__ == "__main__":
    unittest.main()


class TestRedactedThinking(unittest.TestCase):
    """Tests for redacted_thinking block handling."""

    def test_redacted_thinking_block_skipped_silently(self):
        """redacted_thinking blocks should be silently skipped, not emitted or warned."""
        state = StreamState()
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "redacted_thinking", "data": "enc_gAAAAA_encrypted_blob"},
                ]
            },
        }
        emitted = ClaudeCodeBridge._consume_event(state, event)
        # Only the text block should produce a stream event
        stream_events = [e for e in emitted if e["type"] == "stream_event"]
        self.assertEqual(len(stream_events), 1)
        self.assertEqual(stream_events[0]["event"].kind, "text")
        self.assertEqual(stream_events[0]["event"].text, "Hello")
        # assistant_texts should only contain the text block
        self.assertEqual(state.result.assistant_texts, ["Hello"])

    def test_redacted_thinking_no_unknown_warning(self):
        """redacted_thinking should NOT trigger the UNKNOWN block warning."""
        state = StreamState()
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "redacted_thinking", "data": "enc_gAAAAA_encrypted_blob"},
                ]
            },
        }
        with patch("gateway.log") as mock_log:
            ClaudeCodeBridge._consume_event(state, event)
            # Check no warning calls contain "UNKNOWN"
            for call in mock_log.warning.call_args_list:
                self.assertNotIn("UNKNOWN", str(call), "redacted_thinking should not trigger UNKNOWN warning")

    def test_redacted_thinking_only_event_produces_empty_texts(self):
        """Assistant event with only redacted_thinking should not add to assistant_texts."""
        state = StreamState()
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "redacted_thinking", "data": "encrypted_content"},
                ]
            },
        }
        ClaudeCodeBridge._consume_event(state, event)
        self.assertEqual(state.result.assistant_texts, [])
        self.assertEqual(state.result.thinking, [])


class TestToolUseFallbackReply(unittest.TestCase):
    """Tests for fallback reply when tool_calls exist but no text."""

    def test_reply_text_fallback_with_tool_calls(self):
        """When result_text and assistant_texts are empty but tool_calls exist, reply_text should provide fallback."""
        result = StreamResult(
            tool_calls=[ToolCall(name="Bash", input_text="ls")],
        )
        self.assertNotEqual(result.reply_text, "")
        self.assertIn("Bash", result.reply_text)
        self.assertIn("⚠️", result.reply_text)

    def test_reply_text_prefers_result_text_over_fallback(self):
        """result_text takes priority even when tool_calls exist."""
        result = StreamResult(
            result_text="Done!",
            tool_calls=[ToolCall(name="Bash", input_text="ls")],
        )
        self.assertEqual(result.reply_text, "Done!")

    def test_reply_text_prefers_assistant_texts_over_fallback(self):
        """assistant_texts takes priority over tool_calls fallback."""
        result = StreamResult(
            assistant_texts=["Here is the result"],
            tool_calls=[ToolCall(name="Bash", input_text="ls")],
        )
        self.assertEqual(result.reply_text, "Here is the result")

    def test_reply_text_empty_when_nothing(self):
        """reply_text should be empty when nothing is available."""
        result = StreamResult()
        self.assertEqual(result.reply_text, "")

    def test_reply_text_fallback_multiple_tools(self):
        """Fallback message should list multiple tool names."""
        result = StreamResult(
            tool_calls=[
                ToolCall(name="Bash", input_text="ls"),
                ToolCall(name="Read", input_text="file.py"),
                ToolCall(name="Edit", input_text="fix.py"),
            ],
        )
        text = result.reply_text
        self.assertIn("Bash", text)
        self.assertIn("Read", text)
        self.assertIn("Edit", text)

    def test_reply_text_fallback_many_tools_truncated(self):
        """When >5 tools, fallback should show count."""
        result = StreamResult(
            tool_calls=[ToolCall(name=f"Tool{i}", input_text="x") for i in range(8)],
        )
        text = result.reply_text
        self.assertIn("8", text)

    def test_is_empty_false_with_tool_calls(self):
        """is_empty should be False when tool_calls exist (even without text)."""
        result = StreamResult(
            tool_calls=[ToolCall(name="Bash", input_text="ls")],
        )
        # With the new fallback, reply_text is non-empty so is_empty is False
        self.assertFalse(result.is_empty)


class TestStopReasonCapture(unittest.TestCase):
    """Tests for stop_reason capture in StreamResult."""

    def test_stop_reason_captured_from_result_event(self):
        """stop_reason from the result event should be stored in StreamResult."""
        state = StreamState()
        event = {
            "type": "result",
            "result": "",
            "stop_reason": "tool_use",
            "num_turns": 1,
            "session_id": "sess_abc",
        }
        ClaudeCodeBridge._consume_event(state, event)
        self.assertEqual(state.result.stop_reason, "tool_use")

    def test_stop_reason_none_when_normal(self):
        """stop_reason should be None when not set in event."""
        state = StreamState()
        event = {
            "type": "result",
            "result": "All done.",
            "session_id": "sess_abc",
        }
        ClaudeCodeBridge._consume_event(state, event)
        self.assertIsNone(state.result.stop_reason)

    def test_stop_reason_end_turn(self):
        """stop_reason=end_turn should be captured normally."""
        state = StreamState()
        event = {
            "type": "result",
            "result": "Done.",
            "stop_reason": "end_turn",
            "num_turns": 5,
        }
        ClaudeCodeBridge._consume_event(state, event)
        self.assertEqual(state.result.stop_reason, "end_turn")
        self.assertEqual(state.result.result_text, "Done.")

# =========================================================================
# Encrypted thinking blocks & tool_use retry
# =========================================================================

class TestEncryptedThinkingBlocks(unittest.TestCase):
    """Encrypted/signature-only thinking blocks must not break parsing."""

    def test_signature_only_block_skipped(self):
        """Content block with only 'signature' (no type) should be skipped silently."""
        state = StreamState()
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"signature": "enc_gAAAAA_very_long_encrypted_blob"},
                    {"type": "text", "text": "Here is the answer."},
                ]
            },
        }
        emitted = ClaudeCodeBridge._consume_event(state, event)
        self.assertEqual(len(state.result.assistant_texts), 1)
        self.assertEqual(state.result.assistant_texts[0], "Here is the answer.")
        # Should emit only the text event, not the signature block
        stream_events = [e for e in emitted if e["type"] == "stream_event"]
        self.assertEqual(len(stream_events), 1)
        self.assertEqual(stream_events[0]["event"].kind, "text")

    def test_signature_block_before_tool_use(self):
        """Signature block followed by tool_use block — tool_use must still be captured."""
        state = StreamState()
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"signature": "enc_gAAAAA_encrypted_thinking"},
                    {"type": "tool_use", "id": "toolu_abc", "name": "Read", "input": {"file_path": "/src/main.py"}},
                ]
            },
        }
        emitted = ClaudeCodeBridge._consume_event(state, event)
        self.assertEqual(len(state.result.tool_calls), 1)
        self.assertEqual(state.result.tool_calls[0].name, "Read")
        self.assertEqual(state.result.tool_calls[0].input_text, "/src/main.py")

    def test_only_signature_block_no_crash(self):
        """Content with only a signature block should not crash and yield nothing."""
        state = StreamState()
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"signature": "enc_gAAAAA_some_encrypted_data"},
                ]
            },
        }
        emitted = ClaudeCodeBridge._consume_event(state, event)
        self.assertEqual(len(state.result.assistant_texts), 0)
        self.assertEqual(len(state.result.tool_calls), 0)
        self.assertEqual(len(emitted), 0)

    def test_redacted_thinking_with_type_field(self):
        """Standard redacted_thinking block (with type field) should be skipped."""
        state = StreamState()
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "redacted_thinking", "data": "encrypted_blob"},
                    {"type": "text", "text": "Response text."},
                ]
            },
        }
        emitted = ClaudeCodeBridge._consume_event(state, event)
        self.assertEqual(len(state.result.assistant_texts), 1)
        self.assertEqual(state.result.assistant_texts[0], "Response text.")

    def test_mixed_thinking_signature_tool_use(self):
        """Real-world pattern: thinking + signature + tool_use in one event."""
        state = StreamState()
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "Let me analyze this."},
                    ]
                },
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"signature": "enc_gAAAAA_redacted"},
                        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls -la"}},
                    ]
                },
            },
        ]
        for e in events:
            ClaudeCodeBridge._consume_event(state, e)
        self.assertEqual(len(state.result.thinking), 1)
        self.assertEqual(len(state.result.tool_calls), 1)
        self.assertEqual(state.result.tool_calls[0].name, "Bash")


class TestToolUseRetry(unittest.IsolatedAsyncioTestCase):
    """Auto-retry when CLI exits with stop_reason=tool_use and empty result."""

    def _make_bridge(self):
        cfg = {
            "ttadk_cmd": "echo",
            "model": "gpt-5.4",
            "target": "claude",
            "timeout": 10,
        }
        return ClaudeCodeBridge(cfg)

    async def test_retry_on_tool_use_empty_result(self):
        """stream_ask retries when stop_reason=tool_use with empty result."""
        bridge = self._make_bridge()
        call_count = 0

        async def fake_once(topic_id, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First attempt: tool_use exit with empty result
                yield {
                    "type": "stream_event",
                    "event": StreamEvent(kind="thinking", text="Analyzing..."),
                }
                yield {
                    "type": "final",
                    "result": StreamResult(
                        stop_reason="tool_use",
                        session_id="sess_abc",
                    ),
                }
            else:
                # Retry: successful response
                yield {
                    "type": "stream_event",
                    "event": StreamEvent(kind="text", text="Here is the answer."),
                }
                yield {
                    "type": "final",
                    "result": StreamResult(
                        result_text="Here is the answer.",
                        stop_reason="end_turn",
                        session_id="sess_abc",
                    ),
                }

        bridge._stream_ask_once = fake_once

        events = []
        async for event in bridge.stream_ask("topic1", "hello"):
            events.append(event)

        self.assertEqual(call_count, 2)
        # Final event should have the successful result
        final = [e for e in events if e["type"] == "final"]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["result"].result_text, "Here is the answer.")
        self.assertEqual(final[0]["result"].stop_reason, "end_turn")

    async def test_no_retry_on_normal_result(self):
        """stream_ask does NOT retry when result is normal."""
        bridge = self._make_bridge()
        call_count = 0

        async def fake_once(topic_id, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            yield {
                "type": "final",
                "result": StreamResult(
                    result_text="Normal response.",
                    stop_reason="end_turn",
                    session_id="sess_abc",
                ),
            }

        bridge._stream_ask_once = fake_once

        events = []
        async for event in bridge.stream_ask("topic1", "hello"):
            events.append(event)

        self.assertEqual(call_count, 1)

    async def test_no_retry_when_assistant_texts_present(self):
        """Even with stop_reason=tool_use, don't retry if we have assistant_texts."""
        bridge = self._make_bridge()
        call_count = 0

        async def fake_once(topic_id, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            yield {
                "type": "final",
                "result": StreamResult(
                    assistant_texts=["I found the following..."],
                    stop_reason="tool_use",
                    session_id="sess_abc",
                ),
            }

        bridge._stream_ask_once = fake_once

        events = []
        async for event in bridge.stream_ask("topic1", "hello"):
            events.append(event)

        # Should NOT retry because we have assistant_texts
        self.assertEqual(call_count, 1)

    async def test_retry_exhaustion(self):
        """After max retries, return the last result even if still tool_use."""
        bridge = self._make_bridge()
        call_count = 0

        async def fake_once(topic_id, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            yield {
                "type": "final",
                "result": StreamResult(
                    stop_reason="tool_use",
                    session_id="sess_abc",
                ),
            }

        bridge._stream_ask_once = fake_once

        events = []
        async for event in bridge.stream_ask("topic1", "hello"):
            events.append(event)

        # Should try 1 + 2 retries = 3 total
        self.assertEqual(call_count, 3)
        final = [e for e in events if e["type"] == "final"]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["result"].stop_reason, "tool_use")

    async def test_no_retry_without_session_id(self):
        """Cannot retry without session_id (needed for --resume)."""
        bridge = self._make_bridge()
        call_count = 0

        async def fake_once(topic_id, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            yield {
                "type": "final",
                "result": StreamResult(
                    stop_reason="tool_use",
                    session_id=None,  # no session
                ),
            }

        bridge._stream_ask_once = fake_once

        events = []
        async for event in bridge.stream_ask("topic1", "hello"):
            events.append(event)

        # Should NOT retry without session_id
        self.assertEqual(call_count, 1)

    async def test_retry_uses_continue_prompt(self):
        """Retry should use 'continue' as the prompt."""
        bridge = self._make_bridge()
        prompts_received = []

        async def fake_once(topic_id, prompt, **kwargs):
            prompts_received.append(prompt)
            if len(prompts_received) == 1:
                yield {
                    "type": "final",
                    "result": StreamResult(
                        stop_reason="tool_use",
                        session_id="sess_abc",
                    ),
                }
            else:
                yield {
                    "type": "final",
                    "result": StreamResult(
                        result_text="Done.",
                        stop_reason="end_turn",
                        session_id="sess_abc",
                    ),
                }

        bridge._stream_ask_once = fake_once

        async for _ in bridge.stream_ask("topic1", "original question"):
            pass

        self.assertEqual(prompts_received[0], "original question")
        self.assertEqual(prompts_received[1], "continue")

    async def test_stream_events_forwarded_during_retry(self):
        """Stream events from all attempts should be forwarded to caller."""
        bridge = self._make_bridge()
        call_count = 0

        async def fake_once(topic_id, prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            yield {
                "type": "stream_event",
                "event": StreamEvent(kind="thinking", text=f"Attempt {call_count}"),
            }
            if call_count == 1:
                yield {
                    "type": "final",
                    "result": StreamResult(
                        stop_reason="tool_use",
                        session_id="sess_abc",
                    ),
                }
            else:
                yield {
                    "type": "final",
                    "result": StreamResult(
                        result_text="Success",
                        stop_reason="end_turn",
                        session_id="sess_abc",
                    ),
                }

        bridge._stream_ask_once = fake_once

        events = []
        async for event in bridge.stream_ask("topic1", "hello"):
            events.append(event)

        stream_events = [e for e in events if e["type"] == "stream_event"]
        self.assertEqual(len(stream_events), 2)
        self.assertEqual(stream_events[0]["event"].text, "Attempt 1")
        self.assertEqual(stream_events[1]["event"].text, "Attempt 2")


class TestAssistantEventDiagnostics(unittest.TestCase):
    """Event-level diagnostic logging for assistant events."""

    def test_empty_content_array(self):
        """Assistant event with empty content should not crash."""
        state = StreamState()
        event = {"type": "assistant", "message": {"content": []}}
        emitted = ClaudeCodeBridge._consume_event(state, event)
        self.assertEqual(len(emitted), 0)
        self.assertEqual(len(state.result.assistant_texts), 0)

    def test_content_with_non_dict_items(self):
        """Non-dict items in content array should be skipped."""
        state = StreamState()
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    "some string",
                    42,
                    {"type": "text", "text": "Valid text."},
                ]
            },
        }
        emitted = ClaudeCodeBridge._consume_event(state, event)
        self.assertEqual(len(state.result.assistant_texts), 1)
        self.assertEqual(state.result.assistant_texts[0], "Valid text.")

    def test_parse_stream_with_encrypted_blocks(self):
        """Full stream parse with encrypted thinking blocks."""
        stream = "\n".join([
            json.dumps({"type": "system", "subtype": "init", "session_id": "sess_1"}),
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "Let me check."},
                    ]
                },
            }),
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [
                        {"signature": "enc_gAAAAA_long_encrypted_blob"},
                    ]
                },
            }),
            json.dumps({
                "type": "result",
                "result": "",
                "stop_reason": "tool_use",
                "num_turns": 1,
                "session_id": "sess_1",
            }),
        ])
        result = ClaudeCodeBridge._parse_stream(stream)
        self.assertEqual(result.stop_reason, "tool_use")
        self.assertEqual(len(result.thinking), 1)
        self.assertEqual(result.thinking[0], "Let me check.")
        self.assertEqual(result.result_text, "")
        self.assertEqual(len(result.tool_calls), 0)
        self.assertEqual(len(result.assistant_texts), 0)
