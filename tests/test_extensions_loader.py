# tests/test_extensions_loader.py
from __future__ import annotations

from pathlib import Path

import pytest

from tools import registry


def _project_extensions_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "extensions"


@pytest.fixture(autouse=True)
def _clear_tool_registry():
    registry._TOOLS.clear()


def test_load_project_hello_tool_extension():
    from extensions.loader import load_extensions

    result = load_extensions(base_dir=_project_extensions_dir())

    assert "tools:hello_tool.py" in result["loaded"]
    assert result["failed"] == []
    assert result["failed_ids"] == []
    assert "hello_tool" in registry.list_tools()
    assert "hello" in registry.execute("hello_tool", {})


def test_extension_failure_does_not_stop_loading(tmp_path):
    from extensions.loader import load_extensions

    ext_root = tmp_path / "extensions"
    tools_dir = ext_root / "tools"
    tools_dir.mkdir(parents=True)

    (tools_dir / "bad_tool.py").write_text(
        "def register():\n    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    (tools_dir / "good_tool.py").write_text(
        "from tools.registry import register as register_tool\n\n"
        "def _good_tool() -> str:\n"
        "    return '[ok] good'\n\n"
        "def register() -> None:\n"
        "    register_tool(\n"
        "        name='tmp_good_tool',\n"
        "        description='temporary good tool',\n"
        "        parameters={},\n"
        "        required=[],\n"
        "        handler=_good_tool,\n"
        "    )\n",
        encoding="utf-8",
    )

    result = load_extensions(base_dir=ext_root)

    assert "tools:bad_tool.py" in result["failed_ids"]
    assert "tools:good_tool.py" in result["loaded"]
    failed = {item["id"]: item for item in result["failed"]}
    assert failed["tools:bad_tool.py"]["category"] == "tools"
    assert failed["tools:bad_tool.py"]["path"].endswith("bad_tool.py")
    assert "boom" in failed["tools:bad_tool.py"]["error"]
    assert "tmp_good_tool" in registry.list_tools()
    assert "[ok] good" in registry.execute("tmp_good_tool", {})


def test_missing_register_contract_reported(tmp_path):
    from extensions.loader import load_extensions

    ext_root = tmp_path / "extensions"
    tools_dir = ext_root / "tools"
    tools_dir.mkdir(parents=True)

    (tools_dir / "no_register.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = load_extensions(base_dir=ext_root)

    assert result["loaded"] == []
    assert "tools:no_register.py" in result["failed_ids"]
    failed = {item["id"]: item for item in result["failed"]}
    assert "缺少 register" in failed["tools:no_register.py"]["error"]


def test_non_callable_register_reported(tmp_path):
    from extensions.loader import load_extensions

    ext_root = tmp_path / "extensions"
    tools_dir = ext_root / "tools"
    tools_dir.mkdir(parents=True)

    (tools_dir / "bad_register.py").write_text("register = 123\n", encoding="utf-8")

    result = load_extensions(base_dir=ext_root)

    assert result["loaded"] == []
    assert "tools:bad_register.py" in result["failed_ids"]
    failed = {item["id"]: item for item in result["failed"]}
    assert "不是可调用对象" in failed["tools:bad_register.py"]["error"]
