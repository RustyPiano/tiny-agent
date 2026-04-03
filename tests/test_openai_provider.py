from llm.openai_provider import OpenAIProvider


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
