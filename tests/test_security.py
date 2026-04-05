from agent_framework.core.security import SecurityGuard
from agent_framework.tools.bash_tool import run_bash


def test_security_guard_blocks_non_whitelisted_tool() -> None:
    guard = SecurityGuard(allowed_tools={"run_bash"})

    allowed, reason = guard.validate_tool_call("write_file", {"path": "a.txt"})

    assert allowed is False
    assert "write_file" in reason


def test_security_guard_denies_when_whitelist_empty() -> None:
    guard = SecurityGuard(allowed_tools=set())

    allowed, reason = guard.validate_tool_call("run_bash", {"command": "echo hi"})

    assert allowed is False
    assert "白名单为空" in reason


def test_run_bash_blocks_sudo() -> None:
    result = run_bash("sudo ls")

    assert "[blocked]" in result
    assert "sudo" in result.lower()


def test_run_bash_allows_curl() -> None:
    # curl is no longer blocked; pipe-to-shell is blocked instead
    result = run_bash("curl --version")

    assert "[blocked]" not in result


def test_run_bash_allows_curl_with_absolute_path() -> None:
    result = run_bash("/usr/bin/curl --version 2>/dev/null || true")

    assert "[blocked]" not in result


def test_runtime_blocks_non_whitelisted_tool_call() -> None:
    from agent_framework._config import AgentSettings
    from agent_framework.core.context import Context
    from agent_framework.core.logging import RunContext
    from agent_framework.core.runtime import AgentRuntime
    from agent_framework.llm.base import BaseLLMProvider, LLMResponse, ToolCall

    class NoopProvider(BaseLLMProvider):
        def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
            raise AssertionError("should not call provider.chat")

        def format_tool_result(self, tool_call_id: str, content: str) -> dict:
            return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

        def tool_results_as_message(self, results: list[dict]) -> list[dict]:
            return results

    class WhitelistRegistry:
        def execute(self, name: str, inputs: dict) -> str:
            raise AssertionError("blocked call should not execute")

        def get_schemas(self) -> list[dict]:
            return [{"name": "run_bash"}]

    class NoopStore:
        def save(self, session_id, messages, provider_type):
            _ = (session_id, messages, provider_type)

    runtime = AgentRuntime(
        provider=NoopProvider(),
        settings=AgentSettings(),
        ctx=Context(),
        tool_registry=WhitelistRegistry(),
        session_store=NoopStore(),
        system="test-system",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="openai",
    )

    response = LLMResponse(
        text="",
        tool_calls=[ToolCall(id="call_1", name="write_file", inputs={"path": "x"})],
        stop_reason="tool_use",
        assistant_message={"role": "assistant", "content": ""},
    )

    results = runtime.execute_tools(response)

    assert len(results) == 1
    assert "[blocked]" in results[0]["content"]
    assert "write_file" in results[0]["content"]
