"""
pytest configuration for tests/.

Ensures the repository root is on `sys.path` whether tests are invoked via
`pytest` from the repo root, or via `python tests/test_<name>.py` directly.
Without this, imports like `from rl_trainer import ...` would fail when the
test file is run as a script (its sibling directory is `tests/`, not root).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
