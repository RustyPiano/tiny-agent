from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import config


@contextmanager
def sandbox_cwd():
    """Run shell commands in the workspace root so file tools and bash
    tools share the same working directory."""
    original = os.getcwd()
    workspace = Path(config.WORKSPACE_ROOT).resolve()
    try:
        os.chdir(workspace)
        yield workspace
    finally:
        try:
            os.chdir(original)
        except OSError:
            pass
