"""Simulate Feishu messages through the gateway (no Feishu needed, real ttadk).

Usage: python test_e2e.py
"""

from __future__ import annotations

import asyncio
import logging

from gateway import ClaudeCodeBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


async def main():
    bridge = ClaudeCodeBridge({
        "ttadk_cmd": "ttadk",
        "model": "gpt-5.4",
        "target": "claude",
        "timeout": 120,
        "allowed_tools": "Bash,Read,Write,Edit",
        # No instruction — let Claude Code use its built-in system prompt
    })

    topic = "test_topic_001"

    messages = [
        "当前的项目下有skill吗",
        "你的目录是啥",
        "你的项目规则是什么",
    ]

    for i, msg in enumerate(messages, 1):
        print(f"\n{'=' * 60}")
        print(f"📩 Message {i}: {msg}")
        print(f"{'=' * 60}")
        reply = await bridge.ask(topic, msg)
        print(f"\n🤖 Reply:\n{reply}")
        print(f"\n📎 Session: {bridge._sessions.get(topic)}")

    print(f"\n{'=' * 60}")
    print("✅ Done.")


if __name__ == "__main__":
    asyncio.run(main())
