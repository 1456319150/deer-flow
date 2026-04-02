"""Pytest configuration for the tests/ package.

Adds the project root to sys.path so that test modules can import
top-level project modules (gateway, feishu_bot, weixin_bot, etc.)
without path manipulation in each individual test file.
"""

import os
import sys

# Insert the project root (parent of this tests/ directory) at the
# front of sys.path so `from gateway import ...` etc. resolve correctly.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
