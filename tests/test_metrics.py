from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta, timezone

from core.metrics import MetricsCollector, RunMetrics


def test_summary_handles_type_invalid_values(tmp_path):
    collector = MetricsCollector(output_dir=tmp_path)
    bad_run = {
        "run_id": "r1",
        "session_id": None,
        "provider": "openai",
        "model": "gpt",
        "turns": None,
        "tool_calls": "3",
        "tool_failures": "oops",
        "tool_timeouts": -1,
        "duration_ms": "bad",
    }
    good_run = {
        "run_id": "r2",
        "session_id": "s2",
        "provider": "anthropic",
        "model": "claude",
        "turns": 2,
        "tool_calls": 4,
        "tool_failures": 1,
        "tool_timeouts": 1,
        "duration_ms": 100,
    }
    (tmp_path / "bad.json").write_text(json.dumps(bad_run), encoding="utf-8")
    (tmp_path / "good.json").write_text(json.dumps(good_run), encoding="utf-8")

    summary = collector.summary()

    assert summary["runs"] == 2
    assert summary["tool_calls"] == 7
    assert summary["tool_failures"] == 1
    assert summary["tool_timeouts"] == 1
    assert summary["avg_turns"] == 1
    assert summary["avg_duration_ms"] == 50


def test_record_sanitizes_and_truncates_filename(tmp_path):
    collector = MetricsCollector(output_dir=tmp_path)
    long_run_id = "bad/id:*?name with spaces" + ("x" * 300)
    metrics = RunMetrics(
        run_id=long_run_id,
        session_id=None,
        provider="openai",
        model="gpt",
        turns=1,
        tool_calls=0,
        tool_failures=0,
        tool_timeouts=0,
        start_time=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
        end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
    )

    path = collector.record(metrics)

    assert path.exists()
    assert len(path.name) <= 180
    assert re.match(r"^[A-Za-z0-9._-]+\.json$", path.name)


def test_record_uses_true_utc_label_for_z_suffix(tmp_path):
    collector = MetricsCollector(output_dir=tmp_path)
    utc8 = timezone(timedelta(hours=8))
    metrics = RunMetrics(
        run_id="run",
        session_id=None,
        provider="openai",
        model="gpt",
        turns=1,
        tool_calls=0,
        tool_failures=0,
        tool_timeouts=0,
        start_time=datetime(2026, 1, 1, 8, 0, 0, tzinfo=utc8),
        end_time=datetime(2026, 1, 1, 8, 0, 1, tzinfo=utc8),
    )

    path = collector.record(metrics)

    assert path.name.startswith("20260101T000000000000Z-")
