from __future__ import annotations

from agent_framework._config import AgentSettings
from agent_framework.core.context import Context
from agent_framework.core.logging import RunContext
from agent_framework.core.runtime import AgentRuntime
from agent_framework.core.subagent_flow import SubagentFlowState, advance_flow
from agent_framework.llm.base import BaseLLMProvider, LLMResponse


class _NoopProvider(BaseLLMProvider):
    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        raise AssertionError("chat should not be called")

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        return [{"role": "user", "content": results}]


def test_transition_matrix_is_deterministic() -> None:
    cases = [
        ("implement", "DONE", "advance_phase", "spec_review"),
        ("implement", "DONE_WITH_CONCERNS", "advance_phase", "spec_review"),
        ("implement", "NEEDS_CONTEXT", "wait_for_context", "implement"),
        ("implement", "BLOCKED", "escalate_blocker", "implement"),
        ("spec_review", "DONE", "advance_phase", "quality_review"),
        ("spec_review", "DONE_WITH_CONCERNS", "advance_phase", "quality_review"),
        ("spec_review", "NEEDS_CONTEXT", "wait_for_context", "spec_review"),
        ("spec_review", "BLOCKED", "escalate_blocker", "spec_review"),
        ("quality_review", "DONE", "mark_task_complete", "task_complete"),
        (
            "quality_review",
            "DONE_WITH_CONCERNS",
            "mark_task_complete",
            "task_complete",
        ),
        ("quality_review", "NEEDS_CONTEXT", "wait_for_context", "quality_review"),
        ("quality_review", "BLOCKED", "escalate_blocker", "quality_review"),
    ]

    for phase, status, next_action, next_phase in cases:
        result = advance_flow(phase=phase, status=status)
        assert result == {
            "ok": True,
            "next_action": next_action,
            "phase": next_phase,
            "message": "",
        }


def test_subagent_flow_state_advances_index_only_after_task_completion() -> None:
    state = SubagentFlowState(tasks=["task_001", "task_002"])

    first = state.handle_payload(
        {
            "task_id": "task_001",
            "phase": "implement",
            "status": "DONE",
        }
    )
    assert first["ok"] is True
    assert state.current_index == 0
    assert state.phase == "spec_review"

    second = state.handle_payload(
        {
            "task_id": "task_001",
            "phase": "spec_review",
            "status": "DONE",
        }
    )
    assert second["ok"] is True
    assert state.current_index == 0
    assert state.phase == "quality_review"

    third = state.handle_payload(
        {
            "task_id": "task_001",
            "phase": "quality_review",
            "status": "DONE",
        }
    )
    assert third["ok"] is True
    assert third["next_action"] == "mark_task_complete"
    assert state.current_index == 1
    assert state.phase == "implement"


def test_subagent_flow_state_keeps_last_task_at_task_complete() -> None:
    state = SubagentFlowState(tasks=["task_001"], phase="quality_review")

    result = state.handle_payload(
        {
            "task_id": "task_001",
            "phase": "quality_review",
            "status": "DONE_WITH_CONCERNS",
        }
    )

    assert result["ok"] is True
    assert result["phase"] == "task_complete"
    assert state.current_index == 0
    assert state.phase == "task_complete"


def test_runtime_handle_flow_result_ingests_payload_via_runtime_api() -> None:
    runtime = AgentRuntime(
        provider=_NoopProvider(),
        settings=AgentSettings(enable_subagent_flow=True),
        ctx=Context(),
        tool_registry=None,
        session_store=None,
        system="test",
        run_ctx=RunContext(),
        session_id=None,
        provider_type="unknown",
    )

    runtime.enable_subagent_flow(tasks=["task_001"])
    result = runtime.handle_flow_result(
        {
            "task_id": "task_001",
            "phase": "implement",
            "status": "NEEDS_CONTEXT",
            "details": "waiting for product clarification",
        }
    )

    assert result == {
        "ok": True,
        "next_action": "wait_for_context",
        "phase": "implement",
        "message": "waiting for product clarification",
    }
