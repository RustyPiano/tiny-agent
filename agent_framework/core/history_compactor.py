from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

def compact_history(
    history: list[str],
    max_records: int,
    summarize_fn: Callable[[list[str]], str],
) -> list[str]:
    if max_records < 11:
        raise ValueError("max_records must be >= 11 for summary + latest 10 policy")

    if len(history) <= max_records:
        return history

    old_part = history[: -(max_records - 1)]
    summary = summarize_fn(old_part)
    return [summary, *history[-(max_records - 1) :]]


def compact_message_history(
    history: list[T],
    recent_window_size: int,
    summarize_fn: Callable[[list[T]], str],
    should_keep_with_next: Callable[[T, T], bool] | None = None,
) -> tuple[str, list[T]]:
    if recent_window_size < 1:
        raise ValueError("recent_window_size must be >= 1")

    if len(history) <= recent_window_size:
        return "", history

    window_start = len(history) - recent_window_size
    if should_keep_with_next is not None:
        while window_start > 0 and should_keep_with_next(
            history[window_start - 1], history[window_start]
        ):
            window_start -= 1

    older_history = history[:window_start]
    recent_window = history[window_start:]
    if not older_history:
        return "", history

    return summarize_fn(older_history), recent_window
