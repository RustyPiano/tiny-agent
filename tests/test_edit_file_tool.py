from tools.edit_file_tool import edit_file


def test_edit_file_successful_single_replacement(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("hello old world", encoding="utf-8")

    result = edit_file(str(target), "old", "new")

    assert result.startswith("[ok]")
    assert target.read_text(encoding="utf-8") == "hello new world"


def test_edit_file_missing_file(tmp_path):
    target = tmp_path / "missing.txt"

    result = edit_file(str(target), "old", "new")

    assert result.startswith("[error]")
    assert "文件不存在" in result


def test_edit_file_old_str_not_found(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("hello world", encoding="utf-8")

    result = edit_file(str(target), "missing", "new")

    assert result.startswith("[error]")
    assert "old_str" in result


def test_edit_file_rejects_ambiguous_match_when_replace_all_false(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("old\nold\n", encoding="utf-8")

    result = edit_file(str(target), "old", "new", replace_all=False)

    assert result.startswith("[error]")
    assert "replace_all" in result
    assert target.read_text(encoding="utf-8") == "old\nold\n"


def test_edit_file_replace_all_updates_all_matches(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("old\nold\n", encoding="utf-8")

    result = edit_file(str(target), "old", "new", replace_all=True)

    assert result.startswith("[ok]")
    assert target.read_text(encoding="utf-8") == "new\nnew\n"


def test_edit_file_rejects_empty_old_str(tmp_path):
    target = tmp_path / "sample.txt"
    original = "hello world"
    target.write_text(original, encoding="utf-8")

    result = edit_file(str(target), "", "new")

    assert result.startswith("[error]")
    assert "old_str" in result
    assert target.read_text(encoding="utf-8") == original


def test_edit_file_returns_error_for_non_utf8_file(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_bytes(b"\xff\xfe\xfd")

    result = edit_file(str(target), "old", "new")

    assert result.startswith("[error]")
    assert "UTF-8" in result


def test_edit_file_returns_error_when_write_fails(tmp_path, monkeypatch):
    target = tmp_path / "sample.txt"
    original = "hello old world"
    target.write_text(original, encoding="utf-8")

    def fake_write_text(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(target), "write_text", fake_write_text)

    result = edit_file(str(target), "old", "new")

    assert result.startswith("[error]")
    assert "写入" in result
    assert "disk full" in result
    assert target.read_text(encoding="utf-8") == original
