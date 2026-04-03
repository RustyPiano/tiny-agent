from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryStore:
    path: Path
    max_lines: int = 200

    def __post_init__(self) -> None:
        if self.max_lines <= 0:
            raise ValueError("max_lines must be > 0")

    def load_text(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def append(self, topic: str, file_ref: str, summary: str) -> None:
        safe_topic = " ".join(topic.splitlines())
        safe_file_ref = " ".join(file_ref.splitlines())
        safe_summary = " ".join(summary.splitlines())
        line = f"{safe_topic}: {safe_file_ref} | {safe_summary[:150]}"
        existing = self.load_text().splitlines()
        kept = [*existing, line][-self.max_lines :]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(kept), encoding="utf-8")
