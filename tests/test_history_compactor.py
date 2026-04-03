import pytest

from core.history_compactor import compact_history


def test_returns_original_history_when_within_limit():
    history = ["m1", "m2"]

    result = compact_history(history=history, max_records=11, summarize_fn=lambda records: "unused")

    assert result is history


def test_rejects_invalid_max_records_for_summary_plus_latest_ten_policy():
    with pytest.raises(ValueError, match="max_records must be >= 11"):
        compact_history(history=["m1"], max_records=10, summarize_fn=lambda records: "unused")


def test_compacts_old_records_to_summary_and_keeps_latest_ten_for_valid_max_records():
    history = [f"m{i}" for i in range(15)]
    captured: list[list[str]] = []

    def summarize_fn(records: list[str]) -> str:
        captured.append(records)
        return f"summary({len(records)})"

    result = compact_history(history=history, max_records=11, summarize_fn=summarize_fn)

    assert captured == [history[:-10]]
    assert result == ["summary(5)", *history[-10:]]
