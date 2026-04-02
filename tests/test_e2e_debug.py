"""Debug version of E2E test — dumps raw CLI output + parsed JSON structure.

Usage: python test_e2e_debug.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from gateway import ClaudeCodeBridge

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(message)s")


class DebugBridge(ClaudeCodeBridge):
    """Extended bridge that exposes raw CLI output for debugging."""

    async def ask_debug(self, topic_id: str, prompt: str) -> dict:
        """Like ask(), but returns full diagnostic info."""
        session_id = self._sessions.get(topic_id)
        cmd = self._build_cmd(prompt, session_id)

        print(f"\n📋 CMD: {' '.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        raw = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""

        # Parse JSON
        parsed = self._extract_json(raw)

        # Update session (same logic as ask())
        if parsed:
            for key in ("session_id", "conversation_id", "uuid"):
                sid = parsed.get(key)
                if sid:
                    self._sessions[topic_id] = sid
                    break

        return {
            "exit_code": proc.returncode,
            "raw_length": len(raw),
            "raw_last_500": raw[-500:] if raw else "",
            "raw_first_500": raw[:500] if raw else "",
            "parsed": parsed,
            "parsed_keys": list(parsed.keys()) if parsed else None,
            "result_value": repr(parsed.get("result")) if parsed else None,
            "all_string_fields": {
                k: repr(v[:200]) if isinstance(v, str) else repr(v)
                for k, v in (parsed or {}).items()
            },
            "session": self._sessions.get(topic_id),
        }


async def main():
    bridge = DebugBridge({
        "ttadk_cmd": "ttadk",
        "model": "gpt-5.4",
        "target": "claude",
        "timeout": 120,
        "allowed_tools": "Bash,Read,Write,Edit",
        "instruction": "You are a helpful AI assistant. Always respond in Chinese.",
    })

    topic = "debug_topic_001"
    messages = [
        "当前的项目下有skill吗",
        "你的目录是啥",
        "你的项目规则是什么",
    ]

    for i, msg in enumerate(messages, 1):
        print(f"\n{'=' * 70}")
        print(f"📩 Message {i}: {msg}")
        print(f"{'=' * 70}")

        diag = await bridge.ask_debug(topic, msg)

        print(f"\n📊 Exit code: {diag['exit_code']}")
        print(f"📊 Raw output length: {diag['raw_length']}")
        print(f"\n📊 Parsed JSON keys: {diag['parsed_keys']}")
        print(f"📊 result field value: {diag['result_value']}")
        print(f"\n📊 All fields (first 200 chars each):")
        for k, v in (diag['all_string_fields'] or {}).items():
            print(f"   {k}: {v}")

        print(f"\n📊 Raw output FIRST 500 chars:")
        print(diag['raw_first_500'])
        print(f"\n📊 Raw output LAST 500 chars:")
        print(diag['raw_last_500'])

        print(f"\n📎 Session: {diag['session']}")

    print(f"\n{'=' * 70}")
    print("✅ Debug run complete.")


if __name__ == "__main__":
    asyncio.run(main())
