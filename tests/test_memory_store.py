from pathlib import Path

import pytest

from core.memory_store import MemoryStore


def test_append_writes_expected_line_format_and_truncates_summary(tmp_path: Path):
    store = MemoryStore(path=tmp_path / "memory.txt")
    long_summary = "x" * 220

    store.append(topic="planning", file_ref="core/runtime.py:42", summary=long_summary)

    assert store.load_text().splitlines() == [f"planning: core/runtime.py:42 | {'x' * 150}"]


def test_append_keeps_only_most_recent_lines_when_over_cap(tmp_path: Path):
    store = MemoryStore(path=tmp_path / "memory.txt", max_lines=3)

    for idx in range(5):
        store.append(topic=f"t{idx}", file_ref=f"f{idx}", summary=f"s{idx}")

    assert store.load_text().splitlines() == [
        "t2: f2 | s2",
        "t3: f3 | s3",
        "t4: f4 | s4",
    ]


def test_rejects_non_positive_max_lines(tmp_path: Path):
    with pytest.raises(ValueError, match="max_lines must be > 0"):
        MemoryStore(path=tmp_path / "memory.txt", max_lines=0)


def test_append_sanitizes_newlines_in_fields_to_keep_single_line(tmp_path: Path):
    store = MemoryStore(path=tmp_path / "memory.txt")

    store.append(topic="plan\nning", file_ref="core\r\nfile.py:1", summary="hello\nworld")

    assert store.load_text().splitlines() == ["plan ning: core file.py:1 | hello world"]
