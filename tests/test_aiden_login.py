"""Unit tests for AidenLoginDetector — ensures we correctly identify the
aiden SSO login prompt (URL + user code + QR block) from CLI stdout so the
gateway can relay it to end users when running headless.
"""

from __future__ import annotations

import sys
import os
import unittest

# Make gateway.py importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gateway import AidenLoginDetector


# Real sample captured from `aiden x claude` on 2026-04-26
SAMPLE_AIDEN_LOGIN = """- Check user login status...

Scan the following QR code to login:

▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
█ ▄▄▄▄▄ █▄  ▄█▄▀█▀  ▀▀█ ▄▄▄▄▄ █
█ █   █ ██▄▀▄▀ █▄▀▀█▀ █ █   █ █
█ █▄▄▄█ ██▀█ ▀▀▄█▄█▄▄ █ █▄▄▄█ █
█▄▄▄▄▄▄▄█ ▀▄▀▄▀▄▀▄█▄▀▄█▄▄▄▄▄▄▄█
█  ▀ ▄▀▄▀█▄▀██▀  █ ▀ ██▄▀▀█▀▀▄█
█ ▀▀██▀▄█ ▄█ ▄▄▀▄▄ █▀█▄█▄█▀▄█▀█
█  █▀██▄    ▄▄▀▀ ▀▀ ▄▄▀ ▄▀▀▀▀ █
█▄█ ▄▀ ▄▄▀█▀██▄  █▄█ █ ▀ ██ █ █
█ ▄███▀▄ ▄█ █ ▄█▀ ▀▀  ▀█▄▀▄▀▀██
█  ▀▄█ ▄▄▄ ▄ ▀ █▄▀▄▀ ▀█▄█▀ ▀█▄█
█▄▄██▄▄▄▄ ▄ █▄▀█ ▄▄▀▄ ▄▄▄  ▄▄██
█ ▄▄▄▄▄ ██ ▄ ▀█▀   █  █▄█ ▄█▀██
█ █   █ █ ▄▀██ ▀  ██   ▄ ▄▄█  █
█ █▄▄▄█ █▀▄▀█ █▄▄▀▄▀██ ▀ ▄▀ ▄ █
█▄▄▄▄▄▄▄█▄███▄██████▄▄██▄▄▄████


Open the following URL to login:
https://sso.bytedance.com/device?usercode=GHYC-DVGG

Manual login page: https://sso.bytedance.com/device

Or enter code: GHYC-DVGG
- Waiting for login...
"""


class TestAidenLoginDetector(unittest.TestCase):
    def test_extracts_url_from_sample(self):
        d = AidenLoginDetector()
        for line in SAMPLE_AIDEN_LOGIN.splitlines(keepends=True):
            d.feed(line)
        self.assertEqual(d.url, "https://sso.bytedance.com/device?usercode=GHYC-DVGG")
        self.assertTrue(d.triggered)
        self.assertTrue(d.has_actionable)

    def test_extracts_user_code_from_sample(self):
        d = AidenLoginDetector()
        for line in SAMPLE_AIDEN_LOGIN.splitlines(keepends=True):
            d.feed(line)
        self.assertEqual(d.user_code, "GHYC-DVGG")

    def test_captures_qr_block(self):
        d = AidenLoginDetector()
        for line in SAMPLE_AIDEN_LOGIN.splitlines(keepends=True):
            d.feed(line)
        # Should have captured 16 QR rows (top-bottom half-block lines)
        self.assertGreaterEqual(len(d.qr_lines), 10)
        self.assertTrue(all(any(c in "█▀▄▌▐" for c in ln) for ln in d.qr_lines))

    def test_triggers_on_keyword_even_without_url(self):
        d = AidenLoginDetector()
        d.feed("- Check user login status...\n")
        d.feed("Scan the following QR code to login:\n")
        self.assertTrue(d.triggered)
        # But has_actionable is still False — we want URL or code before acting
        self.assertFalse(d.has_actionable)

    def test_no_false_positive_on_normal_output(self):
        d = AidenLoginDetector()
        for line in [
            '{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}\n',
            '{"type":"result","subtype":"success","result":"Done"}\n',
            "Some regular log line\n",
        ]:
            d.feed(line)
        self.assertFalse(d.triggered)
        self.assertFalse(d.has_actionable)
        self.assertIsNone(d.url)

    def test_user_message_contains_url_and_code(self):
        d = AidenLoginDetector()
        for line in SAMPLE_AIDEN_LOGIN.splitlines(keepends=True):
            d.feed(line)
        msg = d.build_user_message("aiden")
        self.assertIn("https://sso.bytedance.com/device?usercode=GHYC-DVGG", msg)
        self.assertIn("GHYC-DVGG", msg)
        self.assertIn("aiden x claude", msg)
        # QR code included as fenced block for terminal viewers
        self.assertIn("```", msg)

    def test_handles_byted_org_variant(self):
        d = AidenLoginDetector()
        d.feed("Open the following URL to login:\n")
        d.feed("https://sso.byted.org/device?usercode=ABCD-1234\n")
        self.assertEqual(d.url, "https://sso.byted.org/device?usercode=ABCD-1234")
        self.assertTrue(d.has_actionable)

    def test_extracts_code_from_standalone_line(self):
        d = AidenLoginDetector()
        d.feed("Or enter code: ABCD-1234\n")
        self.assertEqual(d.user_code, "ABCD-1234")

    def test_context_window_bounded(self):
        d = AidenLoginDetector()
        d.feed("Scan the following QR code to login:\n")  # triggers
        for i in range(100):
            d.feed(f"line {i}\n")
        # Context window should cap around 40 lines
        self.assertLessEqual(len(d.context_lines), 40)




class TestAidenConfirmPrompt(unittest.TestCase):
    """Auto-confirm y/N interactive prompts (update / install / proceed)."""

    def test_update_prompt_detected(self):
        d = AidenLoginDetector()
        self.assertTrue(d.is_confirm_prompt(
            "Update available! Would you like to update? (Y/n) "
        ))

    def test_install_bracket_prompt_detected(self):
        d = AidenLoginDetector()
        self.assertTrue(d.is_confirm_prompt(
            "Install @anthropic-ai/claude-code? [Y/n]"
        ))

    def test_proceed_lowercase_bracket(self):
        d = AidenLoginDetector()
        self.assertTrue(d.is_confirm_prompt(
            "Proceed with installation? (y/N)"
        ))

    def test_bare_keyword_with_question_mark(self):
        d = AidenLoginDetector()
        self.assertTrue(d.is_confirm_prompt("Continue?"))
        self.assertTrue(d.is_confirm_prompt("Download missing dependencies?"))

    def test_login_url_is_not_a_confirm_prompt(self):
        """The SSO login line contains '?usercode=' — must not be confused with y/N."""
        d = AidenLoginDetector()
        self.assertFalse(d.is_confirm_prompt(
            "https://sso.bytedance.com/device?usercode=GHYC-DVGG"
        ))

    def test_login_trigger_keyword_is_not_a_confirm_prompt(self):
        d = AidenLoginDetector()
        self.assertFalse(d.is_confirm_prompt("Scan the following QR code to login:"))
        self.assertFalse(d.is_confirm_prompt("- Waiting for login..."))

    def test_normal_output_not_confirm(self):
        d = AidenLoginDetector()
        self.assertFalse(d.is_confirm_prompt(
            '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}'
        ))
        self.assertFalse(d.is_confirm_prompt("Analysis complete."))
        self.assertFalse(d.is_confirm_prompt("I will update the file for you."))  # "update" without ?

    def test_update_notifier_npm_pattern(self):
        d = AidenLoginDetector()
        # Typical `update-notifier` output
        self.assertTrue(d.is_confirm_prompt(
            "Would you like to upgrade now? (Y/n)"
        ))


if __name__ == "__main__":
    unittest.main()
