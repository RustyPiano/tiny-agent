from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from core.context import Context
from core.logging import RunContext, log_event
from core.policies import DefaultRuntimePolicy, RuntimePolicy, Step
from core.react_protocol import ReactDecision, parse_react_json
from llm.base import BaseLLMProvider, LLMResponse, ToolCall

if TYPE_CHECKING:
    from config import AgentSettings


@dataclass
class TurnState:
    turn: int
    response: LLMResponse | None = None


class AgentRuntime:
    def __init__(
        self,
        provider: BaseLLMProvider,
        settings: AgentSettings,
        ctx: Context,
        tool_registry,
        session_store,
        system: str,
        run_ctx: RunContext,
        session_id: str | None,
        provider_type: str,
        policy: RuntimePolicy | None = None,
    ):
        self.provider = provider
        self.settings = settings
        self.ctx = ctx
        self.tool_registry = tool_registry
        self.session_store = session_store
        self.system = system
        self.run_ctx = run_ctx
        self.session_id = session_id
        self.provider_type = provider_type
        self.policy = policy or DefaultRuntimePolicy()
        self.state = TurnState(turn=0)

    def next_step(self, response: LLMResponse | None) -> Step:
        return self.policy.next_step(self.state.turn, self.settings.max_turns, response)

    def call_llm(self) -> LLMResponse:
        return self.provider.chat(
            messages=self.ctx.get(),
            system=self.system,
            tools=self.tool_registry.get_schemas(),
        )

    def parse_react_decision(self, raw: str) -> ReactDecision | None:
        if not isinstance(raw, str) or not raw.strip():
            return None

        allowed_actions: set[str] = set()
        get_schemas = getattr(self.tool_registry, "get_schemas", None)
        if callable(get_schemas):
            schemas = get_schemas() or []
            for schema in schemas:
                if isinstance(schema, dict):
                    schema_name = schema.get("name")
                    if isinstance(schema_name, str) and schema_name:
                        allowed_actions.add(schema_name)

        # Optional backward-compatible fallback.
        list_tools = getattr(self.tool_registry, "list_tools", None)
        if callable(list_tools):
            allowed_actions.update(name for name in list_tools() if isinstance(name, str) and name)

        allowed_actions.add("NONE")
        try:
            return parse_react_json(raw, allowed_actions)
        except ValueError:
            return None

    def execute_tools(self, response: LLMResponse) -> list[dict]:
        results: list[dict] = []
        tool_calls = list(response.tool_calls)
        # ReAct JSON fallback is only considered when tool_calls is empty.
        if not tool_calls:
            react_decision = self.parse_react_decision(response.text)
            if react_decision and react_decision.action != "NONE":
                react_inputs = (
                    react_decision.action_input
                    if isinstance(react_decision.action_input, dict)
                    else {}
                )
                tool_calls = [
                    ToolCall(
                        id=f"react_{uuid4().hex}",
                        name=react_decision.action,
                        inputs=react_inputs,
                    )
                ]

        for tc in tool_calls:
            inputs_obj = tc.inputs if isinstance(tc.inputs, dict) else {}
            execute_inputs = dict(inputs_obj)
            log_inputs = dict(inputs_obj)

            parse_error = tc.parse_error
            raw_arguments = tc.raw_arguments
            if not isinstance(tc.inputs, dict):
                parse_error = (
                    parse_error
                    or "tool inputs 必须是 object，"
                    f"实际: {type(tc.inputs).__name__}"
                )
                if raw_arguments is None:
                    raw_arguments = str(tc.inputs)

            if parse_error:
                execute_inputs["_tool_parse_error"] = parse_error
                if raw_arguments is not None:
                    execute_inputs["_tool_raw_arguments"] = raw_arguments
                log_inputs = {
                    **log_inputs,
                    "_tool_parse_error": parse_error,
                    "_tool_raw_arguments_preview": (raw_arguments or "")[:200],
                }

            log_event("tool_call", self.run_ctx, tool=tc.name, inputs=log_inputs)
            output = self.tool_registry.execute(tc.name, execute_inputs)
            results.append(self.provider.format_tool_result(tc.id, output))
            output_text = str(output)
            log_event("tool_result", self.run_ctx, tool=tc.name, output_preview=output_text[:120])
        return results

    def persist_session(self) -> None:
        if self.session_id:
            self.session_store.save(self.session_id, self.ctx.snapshot(), self.provider_type)

    def final_response_text(self) -> str:
        text = self.state.response.text if self.state.response else ""
        return text or "[无文本输出]"

    def run(self) -> str:
        while True:
            step = self.next_step(self.state.response)
            if step == Step.MAX_TURNS:
                self.persist_session()
                log_event("run_end", self.run_ctx, stop_reason="max_turns")
                return f"[warn] 达到最大轮次 ({self.settings.max_turns})，任务可能未完成。"
            if step == Step.CALL_LLM:
                self.state.turn += 1
                self.run_ctx.turn = self.state.turn
                log_event("turn_start", self.run_ctx)
                response = self.call_llm()
                self.state.response = response
                self.ctx.add_assistant(response.assistant_message)
                continue
            if step == Step.EXECUTE_TOOLS:
                current_response = self.state.response
                if current_response is None:
                    self.persist_session()
                    log_event("run_end", self.run_ctx, stop_reason="invalid_state")
                    return (
                        "[error] Agent runtime state invalid: "
                        "missing response before tool execution."
                    )
                results = self.execute_tools(current_response)
                packaged = self.provider.tool_results_as_message(results)
                self.ctx.add_tool_results(packaged)
                self.state.response = None
                continue
            if step == Step.PERSIST:
                self.persist_session()
                stop_reason = self.state.response.stop_reason if self.state.response else "unknown"
                log_event("run_end", self.run_ctx, stop_reason=stop_reason)
                return self.final_response_text()
            if step == Step.DONE:
                self.persist_session()
                log_event("run_end", self.run_ctx, stop_reason="end_turn")
                return self.final_response_text()
            raise RuntimeError(f"Unhandled runtime step: {step!r}")
