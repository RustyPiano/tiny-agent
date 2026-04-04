from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RUN_ID_ALLOWED_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_MAX_RUN_ID_LEN = 120


def _parse_non_negative_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return 0
        return max(0, int(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            number = float(text)
        except ValueError:
            return 0
        if not math.isfinite(number):
            return 0
        return max(0, int(number))
    return 0


def _sanitize_run_id(run_id: str) -> str:
    safe = _RUN_ID_ALLOWED_CHARS.sub("_", run_id).strip("._-")
    if not safe:
        safe = "run"
    return safe[:_MAX_RUN_ID_LEN]


@dataclass
class RunMetrics:
    run_id: str
    session_id: str | None
    provider: str
    model: str
    turns: int
    tool_calls: int
    tool_failures: int
    tool_timeouts: int
    start_time: datetime
    end_time: datetime

    @property
    def duration_ms(self) -> int:
        delta = self.end_time - self.start_time
        return max(0, int(delta.total_seconds() * 1000))

    @property
    def tool_success_rate(self) -> float:
        tool_calls = _parse_non_negative_int(self.tool_calls)
        tool_failures = _parse_non_negative_int(self.tool_failures)
        tool_timeouts = _parse_non_negative_int(self.tool_timeouts)
        if tool_calls == 0:
            return 1.0
        successes = tool_calls - tool_failures - tool_timeouts
        clamped = max(0, min(successes, tool_calls))
        return clamped / tool_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "turns": _parse_non_negative_int(self.turns),
            "tool_calls": _parse_non_negative_int(self.tool_calls),
            "tool_failures": _parse_non_negative_int(self.tool_failures),
            "tool_timeouts": _parse_non_negative_int(self.tool_timeouts),
            "start_time": self.start_time.astimezone(UTC).isoformat(),
            "end_time": self.end_time.astimezone(UTC).isoformat(),
            "duration_ms": self.duration_ms,
            "tool_success_rate": self.tool_success_rate,
        }


class MetricsCollector:
    def __init__(self, output_dir: str | Path = "outputs/metrics"):
        self.output_dir = Path(output_dir)

    def record(self, metrics: RunMetrics) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        utc_start = metrics.start_time.astimezone(UTC)
        ts_label = utc_start.strftime("%Y%m%dT%H%M%S%fZ")
        safe_run_id = _sanitize_run_id(metrics.run_id)
        file_name = f"{ts_label}-{safe_run_id}.json"
        file_path = self.output_dir / file_name
        file_path.write_text(
            json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return file_path

    def summary(self, last_n: int = 100) -> dict[str, Any]:
        if not self.output_dir.exists():
            return {}

        files = sorted(
            self.output_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[: max(0, last_n)]
        if not files:
            return {}

        runs: list[dict[str, Any]] = []
        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(data, dict):
                runs.append(data)

        if not runs:
            return {}

        total_runs = len(runs)
        total_turns = sum(_parse_non_negative_int(run.get("turns")) for run in runs)
        total_duration_ms = sum(_parse_non_negative_int(run.get("duration_ms")) for run in runs)
        total_tool_calls = sum(_parse_non_negative_int(run.get("tool_calls")) for run in runs)
        total_tool_failures = sum(_parse_non_negative_int(run.get("tool_failures")) for run in runs)
        total_tool_timeouts = sum(_parse_non_negative_int(run.get("tool_timeouts")) for run in runs)

        provider_counts: dict[str, int] = {}
        model_counts: dict[str, int] = {}
        for run in runs:
            provider = str(run.get("provider", "unknown"))
            model = str(run.get("model", "unknown"))
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            model_counts[model] = model_counts.get(model, 0) + 1

        successes = max(0, total_tool_calls - total_tool_failures - total_tool_timeouts)
        tool_success_rate = 1.0 if total_tool_calls == 0 else successes / total_tool_calls

        return {
            "runs": total_runs,
            "avg_turns": total_turns / total_runs,
            "avg_duration_ms": total_duration_ms / total_runs,
            "tool_calls": total_tool_calls,
            "tool_failures": total_tool_failures,
            "tool_timeouts": total_tool_timeouts,
            "tool_success_rate": tool_success_rate,
            "providers": provider_counts,
            "models": model_counts,
        }
