from __future__ import annotations

from collections.abc import Callable


def compact_history(
    history: list[str],
    max_records: int,
    summarize_fn: Callable[[list[str]], str],
) -> list[str]:
    if max_records < 11:
        raise ValueError("max_records must be >= 11 for summary + latest 10 policy")

    if len(history) <= max_records:
        return history

    old_part = history[:-10]
    summary = summarize_fn(old_part)
    return [summary, *history[-10:]]
