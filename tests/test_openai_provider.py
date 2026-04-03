from llm.openai_provider import OpenAIProvider, _resolve_api_key


class _DummyCompletions:
    def __init__(self, resp):
        self._resp = resp

    def create(self, **kwargs):
        _ = kwargs
        return self._resp


class _DummyClient:
    def __init__(self, resp, base_url="https://example.invalid/v1"):
        self.base_url = base_url
        self.chat = type("Chat", (), {"completions": _DummyCompletions(resp)})()


class _Obj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _provider_with_resp(resp) -> OpenAIProvider:
    p = OpenAIProvider.__new__(OpenAIProvider)
    p.model = "moonshotai/Kimi-K2.5"
    p.client = _DummyClient(resp, base_url="https://modelscope.example/v1")
    return p


def test_chat_raises_actionable_error_when_choices_none():
    p = _provider_with_resp(_Obj(choices=None))

    try:
        p.chat(messages=[], system="sys", tools=[])
    except RuntimeError as e:
        msg = str(e)
        assert "LLM 响应解析失败" in msg
        assert "choices" in msg
        assert "base_url" in msg
        assert "tools/function calling" in msg
    else:
        raise AssertionError("expected RuntimeError")


def test_chat_raises_actionable_error_when_choices_empty_with_error_payload():
    p = _provider_with_resp(
        _Obj(
            choices=[],
            error={"message": "tool not supported for this model"},
        )
    )

    try:
        p.chat(messages=[], system="sys", tools=[{"name": "x", "input_schema": {}}])
    except RuntimeError as e:
        msg = str(e)
        assert "LLM 响应解析失败" in msg
        assert "tool not supported for this model" in msg
        assert "model='moonshotai/Kimi-K2.5'" in msg
    else:
        raise AssertionError("expected RuntimeError")


def test_chat_parses_message_content_array_and_tool_args_dict():
    response = _Obj(
        choices=[
            _Obj(
                message=_Obj(
                    content=[_Obj(type="text", text="技能有：")],
                    tool_calls=[
                        _Obj(
                            id="call_1",
                            function=_Obj(
                                name="use_skill",
                                arguments={"name": "using-superpowers"},
                            ),
                        )
                    ],
                )
            )
        ]
    )
    p = _provider_with_resp(response)

    r = p.chat(messages=[], system="sys", tools=[])
    assert r.stop_reason == "tool_use"
    assert r.text == "技能有："
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "use_skill"
    assert r.tool_calls[0].inputs == {"name": "using-superpowers"}


def test_chat_marks_tool_call_parse_error_when_arguments_invalid_json():
    bad_args = '{"path": "snake-game.html", "content": "<html>"'
    response = _Obj(
        choices=[
            _Obj(
                message=_Obj(
                    content="",
                    tool_calls=[
                        _Obj(
                            id="call_bad",
                            function=_Obj(
                                name="write_file",
                                arguments=bad_args,
                            ),
                        )
                    ],
                )
            )
        ]
    )
    p = _provider_with_resp(response)

    r = p.chat(messages=[], system="sys", tools=[])
    assert r.stop_reason == "tool_use"
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "write_file"
    assert r.tool_calls[0].inputs == {}
    assert r.tool_calls[0].parse_error is not None
    assert "JSON" in r.tool_calls[0].parse_error
    assert r.tool_calls[0].raw_arguments == bad_args
    assert r.assistant_message["tool_calls"][0]["function"]["arguments"] == bad_args


def test_chat_parses_legacy_function_call_when_tool_calls_empty():
    response = _Obj(
        choices=[
            _Obj(
                message=_Obj(
                    content="",
                    tool_calls=[],
                    function_call=_Obj(
                        name="use_skill",
                        arguments='{"name": "using-superpowers"}',
                    ),
                )
            )
        ]
    )
    p = _provider_with_resp(response)

    r = p.chat(messages=[], system="sys", tools=[])

    assert r.stop_reason == "tool_use"
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "use_skill"
    assert r.tool_calls[0].inputs == {"name": "using-superpowers"}
    assert r.assistant_message["tool_calls"][0]["function"]["name"] == "use_skill"


def test_chat_uses_max_completion_tokens_for_official_openai_endpoint():
    captured: dict = {}

    class _CaptureCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Obj(choices=[_Obj(message=_Obj(content="ok", tool_calls=[]))])

    p = OpenAIProvider.__new__(OpenAIProvider)
    p.model = "gpt-4o-mini"
    p._use_max_completion_tokens = True
    p.client = type(
        "Client",
        (),
        {
            "base_url": "https://api.openai.com/v1",
            "chat": type("Chat", (), {"completions": _CaptureCompletions()})(),
        },
    )()

    r = p.chat(messages=[], system="sys", tools=[])

    assert r.stop_reason == "end_turn"
    assert captured["temperature"] == 0.0
    assert "max_completion_tokens" in captured
    assert "max_tokens" not in captured


def test_chat_uses_max_tokens_for_compat_endpoints():
    captured: dict = {}

    class _CaptureCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Obj(choices=[_Obj(message=_Obj(content="ok", tool_calls=[]))])

    p = OpenAIProvider.__new__(OpenAIProvider)
    p.model = "qwen2.5:14b"
    p._use_max_completion_tokens = False
    p.client = type(
        "Client",
        (),
        {
            "base_url": "http://localhost:11434/v1",
            "chat": type("Chat", (), {"completions": _CaptureCompletions()})(),
        },
    )()

    r = p.chat(messages=[], system="sys", tools=[])

    assert r.stop_reason == "end_turn"
    assert captured["temperature"] == 0.0
    assert "max_tokens" in captured
    assert "max_completion_tokens" not in captured


def test_resolve_api_key_prefers_env_on_official_endpoint():
    assert _resolve_api_key(api_key=None, base_url=None) is None


def test_resolve_api_key_uses_placeholder_for_compat_endpoint():
    assert _resolve_api_key(api_key=None, base_url="http://localhost:11434/v1") == "not-needed"
