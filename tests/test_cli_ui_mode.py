from __future__ import annotations

import builtins

import agent_framework.main as main


class DummySettings:
    provider_type = "openai"
    model = "dummy-model"
    base_url = None

    def validate(self):
        return []

    def to_provider_config(self):
        return {}


def _run_main_with_args(monkeypatch, capsys, args_list, turn_output="[t1] stop=end_turn tools=0"):
    dummy_settings = DummySettings()
    monkeypatch.setattr(main.AgentSettings, "from_env", staticmethod(lambda: dummy_settings))
    monkeypatch.setattr(main, "bootstrap", lambda settings: None)
    monkeypatch.setattr(main, "create_provider", lambda cfg: object())

    captured_kwargs: dict = {}

    def fake_run(user_input: str, **kwargs):
        _ = user_input
        captured_kwargs.update(kwargs)
        kwargs["turn_printer"](turn_output)
        return "done"

    monkeypatch.setattr(main, "run", fake_run)

    inputs = iter(["hello"])

    def fake_input(prompt: str) -> str:
        _ = prompt
        try:
            return next(inputs)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr("sys.argv", ["main.py"] + args_list)

    main.main()
    out = capsys.readouterr().out
    return captured_kwargs, out


def test_ui_mode_concise_does_not_print_turns(monkeypatch, capsys):
    captured_kwargs, out = _run_main_with_args(
        monkeypatch, capsys, ["--ui", "concise"], turn_output="[t1] stop=end_turn tools=0"
    )

    assert captured_kwargs["show_turns"] is False
    assert callable(captured_kwargs["turn_printer"])
    assert "[t1] stop=end_turn tools=0" not in out
    assert "开始处理" in out
    assert "─" * 40 in out


def test_ui_mode_detailed_prints_turns(monkeypatch, capsys):
    captured_kwargs, out = _run_main_with_args(
        monkeypatch, capsys, ["--ui", "detailed"], turn_output="[t1] stop=end_turn tools=0"
    )

    assert captured_kwargs["show_turns"] is True
    assert callable(captured_kwargs["turn_printer"])
    assert "[t1] stop=end_turn tools=0" in out
    assert "Agent: done" in out


def test_show_turns_fallback_to_detailed(monkeypatch, capsys):
    captured_kwargs, out = _run_main_with_args(
        monkeypatch, capsys, ["--show-turns"], turn_output="[t1] stop=end_turn tools=0"
    )

    assert captured_kwargs["show_turns"] is True
    assert callable(captured_kwargs["turn_printer"])
    assert "[t1] stop=end_turn tools=0" in out


def test_ui_concise_takes_priority_over_show_turns(monkeypatch, capsys):
    captured_kwargs, out = _run_main_with_args(
        monkeypatch,
        capsys,
        ["--ui", "concise", "--show-turns"],
        turn_output="[t1] stop=end_turn tools=0",
    )

    assert captured_kwargs["show_turns"] is False
    assert callable(captured_kwargs["turn_printer"])
    assert "[t1] stop=end_turn tools=0" not in out
    assert "开始处理" in out
