from __future__ import annotations


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def should_compact(estimated_tokens: int, soft_limit: int) -> bool:
    return estimated_tokens >= soft_limit
