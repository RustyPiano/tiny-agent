from config import (
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
