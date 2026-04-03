from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import config
from core.context import Context
from core.context_budget import estimate_tokens, should_compact
from core.logging import RunContext, log_event
from core.message_assembler import assemble_messages
from core.policies import DefaultRuntimePolicy, RuntimePolicy, Step
from core.react_protocol import ReactDecision, parse_react_json
from core.sandbox import sandbox_cwd
from core.security import SecurityGuard
from llm.base import BaseLLMProvider, LLMResponse, ToolCall

if TYPE_CHECKING:
    from config import AgentSettings


@dataclass
class TurnState:
    turn: int
    response: LLMResponse | None = None
    react_format_retries: int = 0


REACT_FORMAT_MAX_RETRIES = 3
REACT_FORMAT_RETRY_OBSERVATION = (
    "Observation: ReAct output parse failed. Reply with strict JSON only, "
    'exact keys {"thought","action","action_input"}, action must be allowed tool name or "NONE", '
    'and action_input must be an object or the string "NONE".'
)
REACT_FORMAT_RETRY_EXHAUSTED_PAYLOAD = (
    '{"thought":"format_retry_exhausted","action":"NONE","action_input":"NONE"}'
)


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
        allowed_tools: set[str] = set()
        if tool_registry is not None:
            list_tools = getattr(tool_registry, "list_tools", None)
            if callable(list_tools):
                allowed_tools.update(name for name in list_tools() if isinstance(name, str) and name)
            get_schemas = getattr(tool_registry, "get_schemas", None)
            if callable(get_schemas):
                schemas = get_schemas() or []
                for schema in schemas:
                    if isinstance(schema, dict):
                        schema_name = schema.get("name")
                        if isinstance(schema_name, str) and schema_name:
                            allowed_tools.add(schema_name)
        self.security_guard = SecurityGuard(allowed_tools=allowed_tools)

    def next_step(self, response: LLMResponse | None) -> Step:
        return self.policy.next_step(self.state.turn, self.settings.max_turns, response)

    def call_llm(self) -> LLMResponse:
        messages = self.ctx.get()
        current_task = self._latest_user_text(messages)
        last_observation = self._latest_non_user_text(messages)
        estimated_tokens = self._estimate_context_tokens(messages)
        compacted_history = ""
        if should_compact(estimated_tokens, config.CONTEXT_SOFT_LIMIT_TOKENS):
            compacted_history = self._compact_history(messages)
        dynamic_system = assemble_messages(
            static_system_prompt=self.system,
            memory_text="",
            compacted_history=compacted_history,
            last_observation=last_observation,
            current_task=current_task,
        )
        return self.provider.chat(
            messages=messages,
            system=dynamic_system,
            tools=self.tool_registry.get_schemas(),
        )

    def _estimate_context_tokens(self, messages: list[dict]) -> int:
        prompt_text = "\n".join(self._message_text_for_budget(message) for message in messages)
        return estimate_tokens(f"{self.system}\n{prompt_text}")

    def _message_text_for_budget(self, message: dict) -> str:
        try:
            return json.dumps(message, ensure_ascii=False, sort_keys=True)
        except TypeError:
            content = message.get("content", "") if isinstance(message, dict) else message
            if isinstance(content, str):
                return content
            try:
                return json.dumps(content, ensure_ascii=False, sort_keys=True)
            except TypeError:
                return self._message_text_for_prompt(message)

    def _latest_user_text(self, messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            if self._is_synthetic_user_tool_result_message(message):
                continue
            fallback_text = self._message_text_for_prompt(message)
            if fallback_text.strip():
                return fallback_text
        return ""

    def _is_synthetic_user_tool_result_message(self, message: dict) -> bool:
        if message.get("role") != "user":
            return False

        content = message.get("content")
        if not isinstance(content, list) or not content:
            return False

        saw_tool_result = False
        for block in content:
            if not isinstance(block, dict):
                return False
            if block.get("type") != "tool_result":
                return False
            saw_tool_result = True
        return saw_tool_result

    def _latest_non_user_text(self, messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                return self._message_text_for_prompt(message)
        return ""

    def _message_text_for_prompt(self, message: dict) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            content_type = content.get("type")
            if isinstance(content_type, str):
                return f"[{content_type} payload]"
            keys = ", ".join(sorted(str(k) for k in content.keys()))
            return f"[structured payload: {{{keys}}}]"

        if isinstance(content, list):
            if not content:
                return "[empty content blocks]"

            summary_parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    text = block.strip()
                    if text:
                        summary_parts.append(f"text({len(text)} chars)")
                    continue
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if isinstance(block_type, str):
                        if block_type == "tool_result":
                            tool_use_id = block.get("tool_use_id")
                            if isinstance(tool_use_id, str) and tool_use_id:
                                summary_parts.append(f"tool_result:{tool_use_id}")
                            else:
                                summary_parts.append("tool_result")
                        elif block_type == "tool_use":
                            name = block.get("name")
                            if isinstance(name, str) and name:
                                summary_parts.append(f"tool_use:{name}")
                            else:
                                summary_parts.append("tool_use")
                        else:
                            summary_parts.append(block_type)
                    else:
                        keys = ", ".join(sorted(str(k) for k in block.keys()))
                        summary_parts.append(f"block{{{keys}}}")
                    continue
                summary_parts.append(type(block).__name__)

            if summary_parts:
                return "[content blocks: " + ", ".join(summary_parts) + "]"
            return "[content blocks]"

        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return f"[{type(content).__name__} content]"

    def _compact_history(self, messages: list[dict]) -> str:
        lines: list[str] = []
        for message in messages[-config.MAX_COMPACT_HISTORY_MESSAGES:]:
            role = message.get("role", "unknown")
            text = self._message_text_for_prompt(message)
            lines.append(f"{role}: {text}")
        return "\n".join(lines)

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

            allowed, reason = self.security_guard.validate_tool_call(tc.name, execute_inputs)
            if not allowed:
                blocked_output = f"[blocked] {reason}"
                results.append(self.provider.format_tool_result(tc.id, blocked_output))
                log_event("tool_result", self.run_ctx, tool=tc.name, output_preview=blocked_output)
                continue

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
            try:
                if tc.name == "run_bash":
                    with sandbox_cwd():
                        output = self.tool_registry.execute(tc.name, execute_inputs)
                else:
                    output = self.tool_registry.execute(tc.name, execute_inputs)
            except Exception as exc:
                output = f"[tool_error] {type(exc).__name__}: {exc}"
            results.append(self.provider.format_tool_result(tc.id, output))
            output_text = str(output)
            log_event("tool_result", self.run_ctx, tool=tc.name, output_preview=output_text[:120])
        return results

    def _should_retry_react_format(self, response: LLMResponse) -> bool:
        if response.stop_reason != "tool_use":
            return False
        if response.tool_calls:
            return False
        decision = self.parse_react_decision(response.text)
        if decision is None:
            return True
        return decision.action == "NONE"

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
                if self._should_retry_react_format(current_response):
                    self.state.react_format_retries += 1
                    if self.state.react_format_retries > REACT_FORMAT_MAX_RETRIES:
                        self.persist_session()
                        log_event("run_end", self.run_ctx, stop_reason="react_format_retry_exhausted")
                        return REACT_FORMAT_RETRY_EXHAUSTED_PAYLOAD
                    self.ctx.add_user(REACT_FORMAT_RETRY_OBSERVATION)
                    self.state.response = None
                    continue

                results = self.execute_tools(current_response)
                self.state.react_format_retries = 0
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
