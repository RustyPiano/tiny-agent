# skills/registry.py
import pathlib

_SKILLS: dict[str, str] = {}


def register(name: str, prompt_text: str) -> None:
    _SKILLS[name] = prompt_text.strip()


def register_from_file(name: str, path: str) -> bool:
    """从文件注册 skill，成功返回 True，失败返回 False"""
    p = pathlib.Path(path)
    if not p.exists():
        return False
    try:
        content = p.read_text(encoding="utf-8")
        register(name, content)
        return True
    except Exception:
        return False


def load(skill_names: list[str]) -> str:
    parts = [_SKILLS[n] for n in skill_names if n in _SKILLS]
    return "\n\n".join(parts)


def list_skills() -> list[str]:
    return list(_SKILLS.keys())
