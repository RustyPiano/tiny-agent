from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from agent_framework import _config as config


@contextmanager
def sandbox_cwd(workspace_root: Path | str | None = None):
    """Run shell commands in the workspace root so file tools and bash
    tools share the same working directory."""
    original = os.getcwd()
    workspace_value = config.WORKSPACE_ROOT if workspace_root is None else workspace_root
    workspace = Path(workspace_value).resolve()
    try:
        os.chdir(workspace)
        yield workspace
    finally:
        try:
            os.chdir(original)
        except OSError:
            pass
