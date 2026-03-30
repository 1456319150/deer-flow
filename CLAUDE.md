# Project: Feishu Claude Gateway

你是飞书消息网关的 AI 助手，运行在用户的本地机器上。

## 关键上下文
- 本项目是一个飞书 → Claude Code 的消息网关
- 核心文件: gateway.py（400行，FeishuBot + ClaudeCodeBridge）
- 依赖: lark-oapi, pyyaml
- 配置: config.yaml + .env

## 规则
- 回复使用中文
- 代码修改后必须跑 `python -m unittest test_gateway -v` 验证
- 改动超过 3 行用整文件重写，不要局部 patch
