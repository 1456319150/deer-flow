"""Standalone WeChat login script.

Run this to perform QR login and save credentials to disk.
The gateway will then pick up the saved account on next start.

Usage:
    python weixin_login.py [--account-path .weixin-account.json]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from weixin_channel import qr_login, save_account


async def main(account_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    log = logging.getLogger("weixin_login")

    log.info("=== 微信 iLink Bot 登录 ===")
    log.info("登录成功后凭证将保存到: %s", account_path)

    try:
        account = await qr_login()
        save_account(account, account_path)
        log.info("✅ 登录完成! bot_token 已保存.")
        log.info("现在可以启动 gateway: python gateway.py")
    except RuntimeError as e:
        log.error("❌ 登录失败: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("登录取消.")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WeChat iLink Bot Login")
    parser.add_argument(
        "--account-path",
        default=".weixin-account.json",
        help="Path to save account credentials (default: .weixin-account.json)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.account_path))
