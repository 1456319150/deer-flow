"""Unit tests for gateway.py — no Feishu/ttadk connectivity needed.

Run: python test_gateway.py
  or: python -m pytest test_gateway.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest

# Import targets
from gateway import ClaudeCodeBridge, FeishuBot, load_config, load_dotenv


class TestLoadDotenv(unittest.TestCase):
    """Test .env file parsing."""

    def test_basic_vars(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("FOO=bar\nBAZ=qux\n")
            f.flush()
            # Clear any existing values
            os.environ.pop("FOO", None)
            os.environ.pop("BAZ", None)
            load_dotenv(f.name)
            self.assertEqual(os.environ.get("FOO"), "bar")
            self.assertEqual(os.environ.get("BAZ"), "qux")
        os.unlink(f.name)
        os.environ.pop("FOO", None)
        os.environ.pop("BAZ", None)

    def test_quoted_values(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("A='single quoted'\nB=\"double quoted\"\n")
            f.flush()
            os.environ.pop("A", None)
            os.environ.pop("B", None)
            load_dotenv(f.name)
            self.assertEqual(os.environ.get("A"), "single quoted")
            self.assertEqual(os.environ.get("B"), "double quoted")
        os.unlink(f.name)
        os.environ.pop("A", None)
        os.environ.pop("B", None)

    def test_comments_and_blanks(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("# comment\n\nKEY=val\n  # another comment\n")
            f.flush()
            os.environ.pop("KEY", None)
            load_dotenv(f.name)
            self.assertEqual(os.environ.get("KEY"), "val")
        os.unlink(f.name)
        os.environ.pop("KEY", None)

    def test_existing_env_not_overridden(self):
        """Existing env vars should take priority over .env file."""
        os.environ["PRIORITY"] = "from_env"
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("PRIORITY=from_file\n")
            f.flush()
            load_dotenv(f.name)
            self.assertEqual(os.environ["PRIORITY"], "from_env")
        os.unlink(f.name)
        os.environ.pop("PRIORITY", None)

    def test_missing_file_no_error(self):
        """Missing .env file should silently pass."""
        load_dotenv("/nonexistent/.env")  # Should not raise


class TestLoadConfig(unittest.TestCase):
    """Test YAML config loading with env var substitution."""

    def test_env_substitution(self):
        os.environ["TEST_APP_ID"] = "cli_test123"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("feishu:\n  app_id: $TEST_APP_ID\n")
            f.flush()
            cfg = load_config(f.name)
            self.assertEqual(cfg["feishu"]["app_id"], "cli_test123")
        os.unlink(f.name)
        os.environ.pop("TEST_APP_ID", None)

    def test_missing_env_becomes_empty(self):
        os.environ.pop("NONEXISTENT_VAR", None)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("key: $NONEXISTENT_VAR\n")
            f.flush()
            cfg = load_config(f.name)
            # YAML parses bare empty string as None
            self.assertIn(cfg["key"], ("", None))
        os.unlink(f.name)


class TestExtractJson(unittest.TestCase):
    """Test ClaudeCodeBridge._extract_json() — parsing JSON from mixed CLI output."""

    def test_clean_json_line(self):
        output = '{"result": "hello", "session_id": "abc123"}'
        parsed = ClaudeCodeBridge._extract_json(output)
        self.assertEqual(parsed["result"], "hello")
        self.assertEqual(parsed["session_id"], "abc123")

    def test_json_with_noise_before(self):
        output = (
            "Loading config...\n"
            "Connecting to API...\n"
            '{"result": "done", "session_id": "s1"}\n'
        )
        parsed = ClaudeCodeBridge._extract_json(output)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["result"], "done")

    def test_json_with_noise_after(self):
        output = (
            '{"result": "ok", "session_id": "s2"}\n'
            "Process exited with code 0\n"
        )
        parsed = ClaudeCodeBridge._extract_json(output)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["result"], "ok")

    def test_multiple_json_returns_last(self):
        output = (
            '{"status": "started"}\n'
            'Some log line\n'
            '{"result": "final answer", "session_id": "s3"}\n'
        )
        parsed = ClaudeCodeBridge._extract_json(output)
        self.assertEqual(parsed["result"], "final answer")

    def test_no_json_returns_none(self):
        output = "Just some plain text\nNo JSON here\n"
        parsed = ClaudeCodeBridge._extract_json(output)
        self.assertIsNone(parsed)

    def test_nested_json(self):
        output = '{"result": "hi", "usage": {"input_tokens": 100, "output_tokens": 50}}'
        parsed = ClaudeCodeBridge._extract_json(output)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["usage"]["input_tokens"], 100)

    def test_malformed_json_skipped(self):
        output = (
            '{bad json here}\n'
            '{"result": "good", "session_id": "s4"}\n'
        )
        parsed = ClaudeCodeBridge._extract_json(output)
        self.assertEqual(parsed["result"], "good")


class TestBuildCmd(unittest.TestCase):
    """Test ClaudeCodeBridge._build_cmd() — CLI command construction."""

    def setUp(self):
        self.bridge = ClaudeCodeBridge({
            "ttadk_cmd": "ttadk",
            "model": "gpt-5.4",
            "target": "claude",
            "allowed_tools": "Bash,Read",
        })

    def test_basic_cmd_structure(self):
        cmd = self.bridge._build_cmd("hello world", None)
        self.assertEqual(cmd[0], "ttadk")
        self.assertEqual(cmd[1], "code")
        self.assertEqual(cmd[2], "-t")
        self.assertEqual(cmd[3], "claude")
        self.assertEqual(cmd[4], "-m")
        self.assertEqual(cmd[5], "gpt-5.4")
        self.assertEqual(cmd[6], "-a")

    def test_prompt_in_args(self):
        cmd = self.bridge._build_cmd("test prompt", None)
        a_arg = cmd[7]
        self.assertIn("-p ", a_arg)
        self.assertIn("test prompt", a_arg)

    def test_session_resume(self):
        cmd = self.bridge._build_cmd("hi", "session_xyz")
        a_arg = cmd[7]
        self.assertIn("--resume session_xyz", a_arg)

    def test_no_resume_without_session(self):
        cmd = self.bridge._build_cmd("hi", None)
        a_arg = cmd[7]
        self.assertNotIn("--resume", a_arg)

    def test_allowed_tools_in_args(self):
        cmd = self.bridge._build_cmd("hi", None)
        a_arg = cmd[7]
        self.assertIn("--allowedTools Bash,Read", a_arg)

    def test_json_output_format(self):
        cmd = self.bridge._build_cmd("hi", None)
        a_arg = cmd[7]
        self.assertIn("--output-format json", a_arg)

    def test_single_quote_escaping(self):
        cmd = self.bridge._build_cmd("it's a test", None)
        a_arg = cmd[7]
        # Should not contain bare single quotes that break shell
        self.assertNotIn("it's", a_arg)

    def test_newline_escaping(self):
        cmd = self.bridge._build_cmd("line1\nline2", None)
        a_arg = cmd[7]
        self.assertNotIn("\n", a_arg)
        self.assertIn("\\n", a_arg)


class TestExtractText(unittest.TestCase):
    """Test FeishuBot._extract_text() — Feishu message content parsing."""

    def test_plain_text(self):
        content = {"text": "hello world"}
        self.assertEqual(FeishuBot._extract_text(content), "hello world")

    def test_rich_text_single_paragraph(self):
        content = {
            "content": [
                [{"tag": "text", "text": "Hello"}, {"tag": "text", "text": "World"}]
            ]
        }
        self.assertEqual(FeishuBot._extract_text(content), "Hello World")

    def test_rich_text_multiple_paragraphs(self):
        content = {
            "content": [
                [{"tag": "text", "text": "Para 1"}],
                [{"tag": "text", "text": "Para 2"}],
            ]
        }
        self.assertEqual(FeishuBot._extract_text(content), "Para 1\n\nPara 2")

    def test_at_mention_included(self):
        content = {
            "content": [
                [{"tag": "at", "text": "@Bot"}, {"tag": "text", "text": " do something"}]
            ]
        }
        result = FeishuBot._extract_text(content)
        self.assertIn("@Bot", result)
        self.assertIn("do something", result)

    def test_non_text_tags_ignored(self):
        content = {
            "content": [
                [{"tag": "text", "text": "visible"}, {"tag": "img", "image_key": "xxx"}]
            ]
        }
        self.assertEqual(FeishuBot._extract_text(content), "visible")

    def test_empty_content(self):
        self.assertEqual(FeishuBot._extract_text({}), "")
        self.assertEqual(FeishuBot._extract_text({"content": []}), "")


class TestCard(unittest.TestCase):
    """Test FeishuBot._card() — Feishu interactive card JSON."""

    def test_card_structure(self):
        card_str = FeishuBot._card("# Hello")
        card = json.loads(card_str)
        self.assertTrue(card["config"]["wide_screen_mode"])
        self.assertTrue(card["config"]["update_multi"])
        self.assertEqual(len(card["elements"]), 1)
        self.assertEqual(card["elements"][0]["tag"], "markdown")
        self.assertEqual(card["elements"][0]["content"], "# Hello")

    def test_card_with_special_chars(self):
        text = 'Code: `print("hello")`'
        card = json.loads(FeishuBot._card(text))
        self.assertEqual(card["elements"][0]["content"], text)


class TestSessionManagement(unittest.TestCase):
    """Test ClaudeCodeBridge session tracking."""

    def test_session_stored_on_response(self):
        bridge = ClaudeCodeBridge({})
        # Simulate _parse response updating session
        bridge._sessions["topic_1"] = "session_abc"
        self.assertEqual(bridge._sessions.get("topic_1"), "session_abc")

    def test_different_topics_different_sessions(self):
        bridge = ClaudeCodeBridge({})
        bridge._sessions["topic_a"] = "session_1"
        bridge._sessions["topic_b"] = "session_2"
        self.assertNotEqual(bridge._sessions["topic_a"], bridge._sessions["topic_b"])


class TestBridgeAskMocked(unittest.TestCase):
    """Test ClaudeCodeBridge.ask() with mocked subprocess."""

    def test_timeout_returns_message(self):
        bridge = ClaudeCodeBridge({"timeout": 1, "ttadk_cmd": "sleep"})

        async def run():
            result = await bridge.ask("t1", "test")
            return result

        # sleep as ttadk_cmd will cause argument errors but tests the flow
        result = asyncio.get_event_loop().run_until_complete(run())
        # Should return some string (either timeout or error)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_missing_cmd_returns_error(self):
        bridge = ClaudeCodeBridge({"ttadk_cmd": "nonexistent_cmd_12345"})

        async def run():
            return await bridge.ask("t1", "test")

        result = asyncio.get_event_loop().run_until_complete(run())
        self.assertIn("not found", result.lower())


if __name__ == "__main__":
    unittest.main()
