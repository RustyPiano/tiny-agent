from agent_framework.tools import registry


def setup_function():
    registry.clear_before_tool_call_hooks()


def test_before_tool_call_hook_can_modify_arguments_before_execution():
    def _handler(path: str, content: str) -> str:
        return f"{path}:{content}"

    registry.register(
        name="hook_modify_args_tool",
        description="hook modify args",
        parameters={
            "path": {"type": "string", "description": "path"},
            "content": {"type": "string", "description": "content"},
        },
        required=["path", "content"],
        handler=_handler,
    )

    def _hook(name: str, inputs: dict) -> dict:
        if name == "hook_modify_args_tool":
            inputs = dict(inputs)
            inputs["content"] = "patched"
        return inputs

    registry.register_before_tool_call(_hook)
    result = registry.execute("hook_modify_args_tool", {"path": "/tmp/a", "content": "old"})

    assert result == "/tmp/a:patched"


def test_before_tool_call_hook_can_recover_missing_required_parameter():
    def _handler(path: str, content: str) -> str:
        return f"{path}:{content}"

    registry.register(
        name="hook_recover_required_tool",
        description="hook recover required",
        parameters={
            "path": {"type": "string", "description": "path"},
            "content": {"type": "string", "description": "content"},
        },
        required=["path", "content"],
        handler=_handler,
    )

    def _hook(name: str, inputs: dict) -> dict:
        if name == "hook_recover_required_tool" and "content" not in inputs:
            inputs = dict(inputs)
            inputs["content"] = "restored"
        return inputs

    registry.register_before_tool_call(_hook)
    result = registry.execute("hook_recover_required_tool", {"path": "/tmp/b"})

    assert result == "/tmp/b:restored"


def test_before_tool_call_hook_exception_returns_clear_error():
    def _handler(name: str) -> str:
        return f"hello, {name}"

    registry.register(
        name="hook_exception_tool",
        description="hook exception",
        parameters={
            "name": {"type": "string", "description": "name"},
        },
        required=["name"],
        handler=_handler,
    )

    def _hook(_name: str, _inputs: dict) -> dict:
        raise RuntimeError("hook exploded")

    registry.register_before_tool_call(_hook)
    result = registry.execute("hook_exception_tool", {"name": "world"})

    assert result.startswith("[error]")
    assert "beforeToolCall" in result
    assert "hook exploded" in result


def test_before_tool_call_hook_non_dict_return_returns_clear_error():
    def _handler(name: str) -> str:
        return f"hello, {name}"

    registry.register(
        name="hook_non_dict_tool",
        description="hook non dict",
        parameters={
            "name": {"type": "string", "description": "name"},
        },
        required=["name"],
        handler=_handler,
    )

    def _hook(_name: str, _inputs: dict):
        return "not a dict"

    registry.register_before_tool_call(_hook)
    result = registry.execute("hook_non_dict_tool", {"name": "world"})

    assert result.startswith("[error]")
    assert "beforeToolCall hook #1" in result
    assert "dict" in result
    assert "str" in result


def test_before_tool_call_hooks_apply_in_order_and_chain_data():
    def _handler(path: str, content: str, suffix: str) -> str:
        return f"{path}:{content}:{suffix}"

    registry.register(
        name="hook_chain_tool",
        description="hook chain",
        parameters={
            "path": {"type": "string", "description": "path"},
            "content": {"type": "string", "description": "content"},
            "suffix": {"type": "string", "description": "suffix"},
        },
        required=["path", "content", "suffix"],
        handler=_handler,
    )

    def _hook1(name: str, inputs: dict) -> dict:
        assert name == "hook_chain_tool"
        assert "suffix" not in inputs
        return {**inputs, "suffix": "s1"}

    def _hook2(name: str, inputs: dict) -> dict:
        assert name == "hook_chain_tool"
        assert inputs["suffix"] == "s1"
        return {**inputs, "content": f"{inputs['content']}-updated", "suffix": "s2"}

    registry.register_before_tool_call(_hook1)
    registry.register_before_tool_call(_hook2)
    result = registry.execute("hook_chain_tool", {"path": "/tmp/c", "content": "base"})

    assert result == "/tmp/c:base-updated:s2"
