# tests/conftest.py
"""测试配置，临时修改 WORKSPACE_ROOT 以允许测试使用 tmp_path"""

import pytest

from agent_framework import _config as config


@pytest.fixture(autouse=True)
def _set_workspace_root(tmp_path, monkeypatch):
    """自动为每个测试设置 WORKSPACE_ROOT 为 tmp_path"""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
