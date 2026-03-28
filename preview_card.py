"""Preview what Feishu cards will look like with rich stream-json output.

Run: python3 preview_card.py
No dependencies needed — just prints the formatted Markdown.
"""

from gateway import ClaudeCodeBridge, FeishuBot, StreamResult, ToolCall

# ============================================================
# Scenario 1: Full workflow — thinking + tools + reply
# ============================================================

STREAM_FULL = """
{"type":"system","subtype":"init","session_id":"init_abc"}
{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"用户想知道项目下有没有 skill。我需要先查看目录结构，然后检查 .claude/skills/ 目录。"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"ls -la .claude/skills/ 2>/dev/null || echo 'No skills directory'"}}]}}
{"type":"assistant","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"total 16\\n-rw-r--r-- 1 user user 1234 Mar 27 gateway.md\\n-rw-r--r-- 1 user user 567 Mar 27 debug.md"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":".claude/skills/gateway.md"}}]}}
{"type":"assistant","message":{"content":[{"type":"tool_result","tool_use_id":"t2","content":"---\\ntitle: Gateway Skill\\ndescription: Feishu to Claude Code relay\\n---\\n# Gateway\\nThis skill handles message routing..."}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"当前项目下有 2 个 skill：\\n\\n1. **gateway.md** — Feishu 到 Claude Code 的消息中继技能\\n2. **debug.md** — 调试辅助技能\\n\\n它们位于 `.claude/skills/` 目录下。"}]}}
{"type":"result","result":"","session_id":"sess_xyz123"}
""".strip()

# ============================================================
# Scenario 2: Simple reply — no tools, no thinking
# ============================================================

STREAM_SIMPLE = """
{"type":"assistant","message":{"content":[{"type":"text","text":"你好！我是 Claude Code，有什么可以帮你的？"}]}}
{"type":"result","result":"","session_id":"sess_simple"}
""".strip()

# ============================================================
# Scenario 3: Thinking + reply, no tools
# ============================================================

STREAM_THINKING_ONLY = """
{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"This is a simple greeting, I should respond naturally in Chinese."}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"项目规则定义在 `.claude/CLAUDE.md` 文件中，主要包含：\\n\\n- 代码风格：使用 Black 格式化\\n- 测试要求：所有改动必须附带单元测试\\n- 提交规范：使用 Conventional Commits"}]}}
{"type":"result","result":"","session_id":"sess_think"}
""".strip()

# ============================================================
# Scenario 4: Multi-tool complex workflow
# ============================================================

STREAM_COMPLEX = """
{"type":"system","subtype":"init","session_id":"init_complex"}
{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"用户要我分析项目目录。需要先 ls，再看 README，可能还需要看 package.json 或 pyproject.toml 来了解依赖。"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"find . -maxdepth 2 -type f | head -30"}}]}}
{"type":"assistant","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"./gateway.py\\n./config.yaml\\n./test_gateway.py\\n./test_e2e.py\\n./.env\\n./requirements.txt\\n./.claude/CLAUDE.md\\n./.claude/skills/gateway.md"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Read","input":{"file_path":"requirements.txt"}}]}}
{"type":"assistant","message":{"content":[{"type":"tool_result","tool_use_id":"t2","content":"lark-oapi>=1.3.0\\npyyaml>=6.0"}]}}
{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t3","name":"Read","input":{"file_path":"config.yaml"}}]}}
{"type":"assistant","message":{"content":[{"type":"tool_result","tool_use_id":"t3","content":"feishu:\\n  app_id: ${FEISHU_APP_ID}\\n  app_secret: ${FEISHU_APP_SECRET}\\nclaude:\\n  model: gpt-5.4\\n  target: claude\\n  timeout: 600"}]}}
{"type":"assistant","message":{"content":[{"type":"text","text":"## 项目结构分析\\n\\n这是一个 **Feishu → Claude Code 网关**项目，包含以下文件：\\n\\n| 文件 | 用途 |\\n|---|---|\\n| `gateway.py` | 主程序，飞书 WebSocket + Claude Code 桥接 |\\n| `config.yaml` | 配置文件（飞书凭证 + Claude 参数） |\\n| `test_gateway.py` | 55 个单元测试 |\\n| `test_e2e.py` | 端到端测试 |\\n| `.env` | 环境变量（APP_ID/SECRET） |\\n\\n依赖只有 2 个：`lark-oapi` 和 `pyyaml`，非常轻量。"}]}}
{"type":"result","result":"","session_id":"sess_complex"}
""".strip()


def main():
    scenarios = [
        ("Scenario 1: Full workflow (thinking + 2 tools + reply)", STREAM_FULL),
        ("Scenario 2: Simple reply (text only)", STREAM_SIMPLE),
        ("Scenario 3: Thinking + reply (no tools)", STREAM_THINKING_ONLY),
        ("Scenario 4: Complex multi-tool analysis", STREAM_COMPLEX),
    ]

    for title, stream_data in scenarios:
        result = ClaudeCodeBridge._parse_stream(stream_data)
        card_md = FeishuBot._format_result(result)

        print("=" * 70)
        print(f"  {title}")
        print("=" * 70)
        print()
        print(f"  Thinking blocks: {len(result.thinking)}")
        print(f"  Tool calls:      {len(result.tool_calls)}")
        print(f"  Text blocks:     {len(result.assistant_texts)}")
        print(f"  Session ID:      {result.session_id}")
        print()
        print("--- Feishu Card Content ---")
        print()
        print(card_md)
        print()
        print()


if __name__ == "__main__":
    main()
