from pathlib import Path

from agent_framework.core.memory_store import MemoryStore


def test_load_text_returns_empty_string_for_missing_file(tmp_path: Path):
    store = MemoryStore(path=tmp_path / "memory.txt")

    assert store.load_text() == ""


def test_load_text_reads_existing_memory_file(tmp_path: Path):
    path = tmp_path / "memory.txt"
    path.write_text("remember this", encoding="utf-8")

    store = MemoryStore(path=path)

    assert store.load_text() == "remember this"
