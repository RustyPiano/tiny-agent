from __future__ import annotations

from pathlib import Path


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path

    def load_text(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
