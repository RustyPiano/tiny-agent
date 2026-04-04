from agent_framework._config import (
    AgentSettings,
    CONTEXT_SOFT_LIMIT_TOKENS,
    MAX_FILE_READ_LINES,
    MAX_HISTORY_RECORDS,
    MAX_MEMORY_LINES,
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    FeatureFlags,
)


def test_philosophy_constants_defaults():
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY == "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"
    assert MAX_FILE_READ_LINES == 2000
    assert MAX_HISTORY_RECORDS == 15
    assert MAX_MEMORY_LINES == 200
    assert CONTEXT_SOFT_LIMIT_TOKENS == 160000


def test_feature_flags_defaults():
    flags = FeatureFlags()

    assert flags.strict_react_json is True
    assert flags.enable_memory_md is True
    assert flags.enable_sandbox is True
    assert flags.enable_multi_agent is False
    assert flags.enable_daemon is False
    assert flags.enable_pet_mode is False


def test_agent_settings_subagent_flow_default_false() -> None:
    settings = AgentSettings()
    assert settings.enable_subagent_flow is False


def test_agent_settings_from_env_parses_subagent_flow_truthy(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ENABLE_SUBAGENT_FLOW", "yes")
    settings = AgentSettings.from_env()
    assert settings.enable_subagent_flow is True


def test_agent_settings_from_env_parses_subagent_flow_falsy(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ENABLE_SUBAGENT_FLOW", "off")
    settings = AgentSettings.from_env()
    assert settings.enable_subagent_flow is False
