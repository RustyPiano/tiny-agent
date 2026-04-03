from llm.anthropic_provider import AnthropicProvider


class _DummyMessages:
    def __init__(self, response):
        self._response = response
        self.captured_kwargs = None

    def create(self, **kwargs):
        self.captured_kwargs = kwargs
        return self._response


class _DummyClient:
    def __init__(self, response):
        self.messages = _DummyMessages(response)


class _Block:
    def __init__(self, type_, **kwargs):
        self.type = type_
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Resp:
    def __init__(self, content):
        self.content = content


def test_chat_passes_temperature_zero_to_anthropic_api():
    response = _Resp([_Block("text", text="ok")])
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = "claude-3-7-sonnet-latest"
    provider.client = _DummyClient(response)

    result = provider.chat(messages=[], system="sys", tools=[])

    assert result.stop_reason == "end_turn"
    assert provider.client.messages.captured_kwargs["temperature"] == 0.0


def test_chat_concatenates_text_blocks_without_inserting_spaces():
    response = _Resp(
        [
            _Block("text", text="alpha"),
            _Block("text", text="beta"),
            _Block("tool_use", id="t1", name="noop", input={}),
            _Block("text", text="gamma"),
        ]
    )
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = "claude-3-7-sonnet-latest"
    provider.client = _DummyClient(response)

    result = provider.chat(messages=[], system="sys", tools=[])

    assert result.text == "alphabetagamma"
