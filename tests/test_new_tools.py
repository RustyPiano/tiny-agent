import json
import os

from tools.finish_tool import finish
from tools.grep_tool import grep
from tools.list_dir_tool import list_dir
from tools.summarize_tool import summarize


def test_finish_returns_json_payload_containing_response() -> None:
    response = "done"
    payload = finish(response)

    data = json.loads(payload)
    assert data["response"] == response


def test_summarize_returns_non_empty_short_text() -> None:
    text = "hello world " * 100
    result = summarize(text, max_chars=60)

    assert isinstance(result, str)
    assert result.strip() != ""
    assert len(result) <= 60


def test_list_dir_and_grep_basic_behavior(tmp_path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    (root / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (root / "b.md").write_text("gamma\nalpha\n", encoding="utf-8")

    listing = list_dir(str(root))
    assert "a.txt" in listing
    assert "b.md" in listing

    matched = grep("alpha", str(root))
    assert "a.txt" in matched
    assert "b.md" in matched


def test_list_dir_rejects_workspace_escape(tmp_path) -> None:
    outside = tmp_path.parent

    result = list_dir(str(outside))

    assert result.startswith("[error] 路径超出工作空间范围")


def test_grep_rejects_workspace_escape(tmp_path) -> None:
    outside_file = tmp_path.parent / "outside.txt"
    outside_file.write_text("alpha\n", encoding="utf-8")

    result = grep("alpha", str(outside_file))

    assert result.startswith("[error] 路径超出工作空间范围")


def test_grep_rejects_invalid_regex(tmp_path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("alpha\n", encoding="utf-8")

    result = grep("(", str(sample))

    assert result.startswith("[error] 非法正则")


def test_grep_ignores_symlink_file_escaping_workspace(tmp_path) -> None:
    if os.name == "nt":
        return

    root = tmp_path / "demo"
    root.mkdir()
    outside_file = tmp_path.parent / "outside_link_target.txt"
    outside_file.write_text("alpha\n", encoding="utf-8")

    link = root / "outside.txt"
    try:
        link.symlink_to(outside_file)
    except (NotImplementedError, OSError):
        return

    result = grep("alpha", str(root))

    assert result == ""


def test_grep_does_not_follow_symlink_directories(tmp_path) -> None:
    if os.name == "nt":
        return

    root = tmp_path / "demo"
    root.mkdir()
    outside_dir = tmp_path.parent / "outside_dir"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "escape.txt").write_text("alpha\n", encoding="utf-8")

    link_dir = root / "linked_dir"
    try:
        link_dir.symlink_to(outside_dir, target_is_directory=True)
    except (NotImplementedError, OSError):
        return

    result = grep("alpha", str(root))

    assert result == ""
