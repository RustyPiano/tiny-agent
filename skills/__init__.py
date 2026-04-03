# skills/__init__.py
import pathlib

from skills.registry import register_from_file

_BUILTIN = pathlib.Path(__file__).parent / "builtin"


def load_builtin_skills() -> None:
    for md_file in _BUILTIN.glob("*.md"):
        register_from_file(md_file.stem, str(md_file))
