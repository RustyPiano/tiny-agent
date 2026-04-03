from __future__ import annotations

from importlib import util
from pathlib import Path
from typing import TypedDict


class FailedExtension(TypedDict):
    id: str
    category: str
    path: str
    error: str


class LoadExtensionsResult(TypedDict):
    loaded: list[str]
    failed: list[FailedExtension]
    failed_ids: list[str]


def _iter_extension_files(base_dir: Path, kind: str) -> list[Path]:
    folder = base_dir / kind
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.py") if p.is_file() and not p.name.startswith("_"))


def _load_module_from_path(module_name: str, path: Path):
    spec = util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法创建模块 spec: {path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_extensions(base_dir: Path | None = None) -> LoadExtensionsResult:
    """按约定目录加载扩展，失败时不中断进程。"""
    root = base_dir or (Path(__file__).resolve().parent)
    loaded: list[str] = []
    failed: list[FailedExtension] = []

    for kind in ("tools", "skills", "providers"):
        for file_path in _iter_extension_files(root, kind):
            extension_id = f"{kind}:{file_path.name}"
            try:
                module_name = f"extensions_dynamic_{kind}_{file_path.stem}"
                module = _load_module_from_path(module_name, file_path)
                register = getattr(module, "register", None)
                if register is None:
                    raise AttributeError("缺少 register() 合约函数")
                if not callable(register):
                    raise TypeError("register 不是可调用对象")
                register()
                loaded.append(extension_id)
            except Exception as e:
                failed.append(
                    {
                        "id": extension_id,
                        "category": kind,
                        "path": str(file_path),
                        "error": str(e),
                    }
                )

    return {
        "loaded": loaded,
        "failed": failed,
        "failed_ids": [item["id"] for item in failed],
    }
