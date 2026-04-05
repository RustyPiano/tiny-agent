from __future__ import annotations

import difflib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anthropic
import openai

from agent_framework import _config as config
from agent_framework.core.context import Context
from agent_framework.core.context_budget import (
    estimate_payload_tokens,
    serialize_for_budget,
    should_compact,
)
from agent_framework.core.history_compactor import compact_message_history
from agent_framework.core.logging import RunContext, log_event
from agent_framework.core.memory_store import MemoryStore
from agent_framework.core.message_assembler import (
    assemble_compacted_history_message,
    assemble_messages,
)
from agent_framework.core.policies import DefaultRuntimePolicy, RuntimePolicy, Step
from agent_framework.core.react_protocol import ReactDecision, parse_react_json_with_error
from agent_framework.core.sandbox import sandbox_cwd
from agent_framework.core.security import SecurityGuard
from agent_framework.core.subagent_flow import SubagentFlowState
from agent_framework.llm.base import BaseLLMProvider, LLMResponse, ToolCall

if TYPE_CHECKING:
    from agent_framework._config import AgentSettings


@dataclass
class TurnState:
    turn: int
    response: LLMResponse | None = None
    react_format_retries: int = 0
    end_turn_react_parse_retries: int = 0
    end_turn_empty_retries: int = 0


REACT_FORMAT_MAX_RETRIES = 3
REACT_FORMAT_RETRY_OBSERVATION = (
    "Observation: ReAct output parse failed. Reply with strict JSON only, "
    'exact keys {"thought","action","action_input"}, action must be allowed tool name or "NONE", '
    'and action_input must be an object or the string "NONE".'
)
REACT_FORMAT_RETRY_EXHAUSTED_PAYLOAD = (
    '{"thought":"format_retry_exhausted","action":"NONE","action_input":"NONE"}'
)
END_TURN_REACT_PARSE_MAX_RETRIES = 2
END_TURN_REACT_PARSE_RETRY_EXHAUSTED_PAYLOAD = (
    '{"thought":"end_turn_parse_retry_exhausted","action":"NONE","action_input":"NONE"}'
)
END_TURN_EMPTY_MAX_RETRIES = 2
END_TURN_EMPTY_RETRY_EXHAUSTED_MESSAGE = "[warn] 模型在 end_turn 返回空响应，重试 2 次后仍为空。"


class AgentRuntime:
    _DYNAMIC_CONTEXT_PREVIEW_CHARS = 200

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
        show_turns: bool = False,
        turn_printer: Callable[[str], None] | None = None,
        ui_event_printer: Callable[[str], None] | None = None,
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
        self.show_turns = show_turns
        self.turn_printer = turn_printer
        self.ui_event_printer = ui_event_printer
        self.state = TurnState(turn=0)
        self._last_tool_statuses: list[str] = []
        self._last_executed_tools_count: int = 0
        self._finish_requested: bool = False
        self._finish_response: str = ""
        self._hb_stop: threading.Event | None = None
        self._hb_thread: threading.Thread | None = None
        self._tool_call_log: list[dict] = []
        self._subagent_flow_state: SubagentFlowState | None = None
        allowed_tools: set[str] = set()
        if tool_registry is not None:
            list_tools = getattr(tool_registry, "list_tools", None)
            if callable(list_tools):
                allowed_tools.update(
                    name for name in list_tools() if isinstance(name, str) and name
                )
            get_schemas = getattr(tool_registry, "get_schemas", None)
            if callable(get_schemas):
                schemas = get_schemas() or []
                for schema in schemas:
                    if isinstance(schema, dict):
                        schema_name = schema.get("name")
                        if isinstance(schema_name, str) and schema_name:
                            allowed_tools.add(schema_name)
        self.security_guard = SecurityGuard(allowed_tools=allowed_tools)

    def enable_subagent_flow(self, tasks: list[str]) -> None:
        self._subagent_flow_state = SubagentFlowState(tasks=list(tasks))

    def handle_flow_result(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {
                "ok": False,
                "next_action": "wait_for_context",
                "phase": "implement",
                "message": "payload must be a dict",
            }

        if self._subagent_flow_state is None:
            return {
                "ok": False,
                "next_action": "wait_for_context",
                "phase": "implement",
                "message": "subagent flow not enabled",
            }

        return self._subagent_flow_state.handle_payload(payload)

    def _start_heartbeat(self, tool_name: str, start_time: float) -> None:
        if self.ui_event_printer is None:
            return
        self._hb_stop = threading.Event()
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, args=(tool_name, start_time), daemon=True
        )
        self._hb_thread.start()

    def _heartbeat_loop(self, tool_name: str, start_time: float) -> None:
        hb_stop = self._hb_stop
        printer = self.ui_event_printer
        if hb_stop is None or printer is None:
            return
        while not hb_stop.wait(10):
            elapsed = int(time.time() - start_time)
            printer(f"  ↳ {tool_name} 运行中 {elapsed}s…")

    def _stop_heartbeat(self) -> None:
        if self._hb_stop is not None:
            self._hb_stop.set()
        if self._hb_thread is not None:
            self._hb_thread.join(timeout=1)

    def _truncate(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "…"

    def _line_count(self, text: str) -> int:
        if not text:
            return 0
        return len(text.splitlines())

    def _tool_status(self, output_text: str) -> str:
        if output_text.startswith("[timeout]"):
            return "timeout"
        if output_text.startswith("[blocked]"):
            return "blocked"
        if output_text.startswith("[error]") or output_text.startswith("[tool_error]"):
            return "error"
        return "ok"

    def _preview_first_line(self, output_text: str, max_len: int = 100) -> str:
        lines = output_text.splitlines()
        first = ""
        for line in lines:
            stripped = line.strip()
            if stripped:
                first = stripped
                break
        if not first:
            first = output_text.strip()
        return self._truncate(first, max_len)

    def _safe_read_text(self, path_value) -> str | None:
        if not isinstance(path_value, str) or not path_value:
            return None
        try:
            p = Path(path_value).resolve()
            if not p.exists() or not p.is_file():
                return None
            return p.read_text(encoding="utf-8")
        except Exception:
            return None

    def _line_delta(self, before_text: str, after_text: str) -> tuple[int, int]:
        before_lines = before_text.splitlines()
        after_lines = after_text.splitlines()
        diff = difflib.ndiff(before_lines, after_lines)
        added = 0
        removed = 0
        for line in diff:
            if line.startswith("+ "):
                added += 1
            elif line.startswith("- "):
                removed += 1
        return added, removed

    def _parse_output_json_dict(self, output_text: str) -> dict | None:
        try:
            payload = json.loads(output_text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _job_error_suffix(self, payload: dict) -> str:
        status_value = payload.get("status")
        if not isinstance(status_value, str):
            return ""
        if status_value.lower() not in {"failed", "error", "timeout", "cancelled"}:
            return ""
        error_value = payload.get("error")
        if error_value in (None, ""):
            error_value = payload.get("message")
        if error_value in (None, ""):
            return ""
        return f" error={self._truncate(str(error_value), 80)}"

    def _emit_tool_detail(
        self,
        tool_name: str,
        inputs: dict,
        output_text: str,
        edit_before_text: str | None = None,
    ) -> None:
        if self.ui_event_printer is None:
            return

        status = self._tool_status(output_text)

        if tool_name == "use_skill":
            skill_name = inputs.get("name", "")
            detail = f"    ↳ skill={skill_name}" if skill_name else "    ↳ skill=<unknown>"
            self.ui_event_printer(detail)
            return

        if tool_name == "list_dir":
            path = inputs.get("path", "")
            if status != "ok":
                self.ui_event_printer(
                    f"    ↳ path={path} 状态={status} 输出={self._preview_first_line(output_text)}"
                )
                return
            payload = self._parse_output_json_dict(output_text)
            if isinstance(payload, dict) and payload.get("tool") == "list_dir":
                entries = payload.get("preview", [])
                if isinstance(entries, list):
                    shown = [str(item).strip() for item in entries if str(item).strip()]
                    next_offset = payload.get("next_offset")
                    truncated = bool(payload.get("truncated"))
                    preview = ", ".join(shown[:8])
                    suffix = ""
                    if truncated and isinstance(next_offset, int):
                        suffix = f", ... (继续 offset={next_offset})"
                    self.ui_event_printer(
                        f"    ↳ path={path} 结果({len(shown)}): {preview}{suffix}"
                    )
                    return
            entries = [line.strip() for line in output_text.splitlines() if line.strip()]
            total = len(entries)
            shown = entries[:8]
            suffix = f", ... (+{total - len(shown)})" if total > len(shown) else ""
            preview = ", ".join(shown)
            self.ui_event_printer(f"    ↳ path={path} 结果({total}): {preview}{suffix}")
            return

        if tool_name == "write_file":
            path = inputs.get("path", "")
            mode = inputs.get("mode", "overwrite")
            content = inputs.get("content", "")
            lines = self._line_count(content) if isinstance(content, str) else 0
            self.ui_event_printer(f"    ↳ path={path} 写入 {lines} 行 ({mode})")
            return

        if tool_name == "read_file":
            path = inputs.get("path", "")
            if status != "ok":
                self.ui_event_printer(
                    f"    ↳ path={path} 状态={status} 输出={self._preview_first_line(output_text)}"
                )
                return
            lines = self._line_count(output_text)
            self.ui_event_printer(f"    ↳ path={path} 读取 {lines} 行")
            return

        if tool_name == "edit_file":
            path = inputs.get("path", "")
            if status != "ok":
                self.ui_event_printer(
                    f"    ↳ path={path} 状态={status} 输出={self._preview_first_line(output_text)}"
                )
                return
            after_text = self._safe_read_text(path)
            if edit_before_text is None or after_text is None:
                self.ui_event_printer(f"    ↳ path={path} +? / -?")
                return
            added, removed = self._line_delta(edit_before_text, after_text)
            self.ui_event_printer(f"    ↳ path={path} +{added} / -{removed}")
            return

        if tool_name == "run_bash":
            command = inputs.get("command", "")
            cmd_preview = self._truncate(command, 80) if isinstance(command, str) else ""
            self.ui_event_printer(f"    ↳ cmd={cmd_preview}")
            self.ui_event_printer(
                f"    ↳ 状态={status} 输出={self._preview_first_line(output_text)}"
            )
            return

        if tool_name == "finish":
            lines = self._line_count(output_text)
            self.ui_event_printer(f"    ↳ 最终回复 {lines} 行")
            return

        if tool_name == "run_job":
            payload = self._parse_output_json_dict(output_text)
            if isinstance(payload, dict):
                operation = inputs.get("operation") or "<unknown>"
                job_id = payload.get("job_id") or inputs.get("job_id") or "<unknown>"
                job_status = payload.get("status") or "unknown"
                activity = payload.get("activity") or "unknown"
                next_poll = payload.get("recommended_poll_after_s")
                exit_code = payload.get("exit_code")
                output_tail = payload.get("output_tail", "")
                if not isinstance(output_tail, str):
                    output_tail = str(output_tail)
                preview_text = self._preview_first_line(output_tail, max_len=80)
                next_part = f" next={next_poll}" if next_poll is not None else ""
                exit_part = f" exit={exit_code}" if exit_code is not None else ""
                preview_part = f" preview={preview_text}" if preview_text else ""
                self.ui_event_printer(
                    f"    ↳ job={job_id} op={operation} status={job_status} activity={activity}"
                    f"{next_part}{exit_part}{preview_part}{self._job_error_suffix(payload)}"
                )
                return

        self.ui_event_printer(f"    ↳ 状态={status} 输出={self._preview_first_line(output_text)}")

    def next_step(self, response: LLMResponse | None) -> Step:
        return self.policy.next_step(self.state.turn, self.settings.max_turns, response)

    def call_llm(self) -> LLMResponse:
        messages = self.ctx.get()
        provider_messages, dynamic_system, tools, _estimated_tokens = self._build_provider_payload(
            messages
        )

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                return self.provider.chat(
                    messages=provider_messages,
                    system=dynamic_system,
                    tools=tools,
                    max_tokens=self.settings.max_tokens,
                )
            except (
                openai.APIConnectionError,
                openai.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.RateLimitError,
            ) as exc:
                log_event(
                    "llm_call_error",
                    self.run_ctx,
                    error=type(exc).__name__,
                    message=str(exc),
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                )
                if attempt < max_attempts - 1:
                    time.sleep(min(2**attempt, 30))
                else:
                    err_detail = f"{type(exc).__name__}: {exc}"
                    msg = (
                        f"[network_error] LLM call failed "
                        f"after {max_attempts} attempts: {err_detail}"
                    )
                    return LLMResponse(
                        text=msg,
                        tool_calls=[],
                        stop_reason="end_turn",
                        assistant_message={
                            "role": "assistant",
                            "content": (
                                f"[network_error] LLM call failed after {max_attempts} attempts"
                            ),
                        },
                    )

    def _estimate_context_tokens(self, messages: list[dict]) -> int:
        return self._build_provider_payload(messages)[3]

    def _tool_schemas(self) -> list[dict]:
        get_schemas = getattr(self.tool_registry, "get_schemas", None)
        if not callable(get_schemas):
            return []
        schemas = get_schemas() or []
        return [schema for schema in schemas if isinstance(schema, dict)]

    def _build_provider_payload(
        self,
        messages: list[dict],
    ) -> tuple[list[dict], str, list[dict], int]:
        tools = self._tool_schemas()
        memory_text = self._load_memory_text()
        current_task = self._summarize_dynamic_context_text(self._latest_user_text(messages))
        last_observation = self._summarize_dynamic_context_text(
            self._latest_non_user_text(messages)
        )
        raw_system = assemble_messages(
            static_system_prompt=self.system,
            memory_text=memory_text,
            compacted_history="",
            last_observation=last_observation,
            current_task=current_task,
        )
        raw_tokens = estimate_payload_tokens(
            raw_system,
            messages,
            tools,
            provider_type=self.provider_type,
        )
        soft_limit = getattr(
            self.settings,
            "context_soft_limit_tokens",
            config.CONTEXT_SOFT_LIMIT_TOKENS,
        )
        if not should_compact(raw_tokens, soft_limit):
            return list(messages), raw_system, tools, raw_tokens

        compacted_history, compacted_messages = self._build_compacted_payload_messages(messages)
        compacted_system = assemble_messages(
            static_system_prompt=self.system,
            memory_text=memory_text,
            compacted_history=compacted_history,
            last_observation=last_observation,
            current_task=current_task,
        )
        compacted_tokens = estimate_payload_tokens(
            compacted_system,
            compacted_messages,
            tools,
            provider_type=self.provider_type,
        )
        if compacted_tokens > raw_tokens:
            return list(messages), raw_system, tools, raw_tokens
        return compacted_messages, compacted_system, tools, compacted_tokens

    def _message_text_for_budget(self, message: dict) -> str:
        payload = serialize_for_budget(message)
        if payload:
            return payload
        content = message.get("content", "") if isinstance(message, dict) else message
        if isinstance(content, str):
            return content
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

    def _summarize_dynamic_context_text(self, text: str) -> str:
        compact = " ".join(str(text).split())
        if not compact:
            return ""
        if len(compact) <= self._DYNAMIC_CONTEXT_PREVIEW_CHARS:
            return compact
        return (
            compact[: self._DYNAMIC_CONTEXT_PREVIEW_CHARS]
            + f"… [truncated {len(compact)} chars]"
        )

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

    def _build_compacted_payload_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        if len(messages) <= 1:
            return "", list(messages)

        history_messages = messages[:-1]
        compacted_history, recent_window = compact_message_history(
            history=history_messages,
            recent_window_size=config.MAX_COMPACT_HISTORY_MESSAGES,
            summarize_fn=self._summarize_history_messages,
            should_keep_with_next=self._messages_must_stay_together,
        )
        if not compacted_history.strip():
            return "", list(messages)

        compacted_message = assemble_compacted_history_message(compacted_history)
        if compacted_message is None:
            return "", list(messages)
        return compacted_history, [compacted_message, *recent_window, messages[-1]]

    def _compact_history(self, messages: list[dict]) -> str:
        compacted_history, _ = self._build_compacted_payload_messages(messages)
        return compacted_history

    def _summarize_history_messages(self, messages: list[dict]) -> str:
        lines: list[str] = []
        for message in messages:
            role = message.get("role", "unknown")
            text = self._message_text_for_prompt(message)
            lines.append(f"{role}: {text}")
        return self._summarize_history_records(lines)

    def _messages_must_stay_together(self, left: dict, right: dict) -> bool:
        if left.get("role") == "tool" and right.get("role") == "tool":
            return True
        left_tool_ids = self._assistant_tool_call_ids(left)
        if not left_tool_ids:
            return False
        right_result_ids = self._tool_result_ids(right)
        return any(tool_id in right_result_ids for tool_id in left_tool_ids)

    def _assistant_tool_call_ids(self, message: dict) -> set[str]:
        if message.get("role") != "assistant":
            return set()

        tool_ids: set[str] = set()
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_id = block.get("id")
                if isinstance(tool_id, str) and tool_id:
                    tool_ids.add(tool_id)

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_id = tool_call.get("id")
                if isinstance(tool_id, str) and tool_id:
                    tool_ids.add(tool_id)
        return tool_ids

    def _tool_result_ids(self, message: dict) -> set[str]:
        result_ids: set[str] = set()
        if message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                result_ids.add(tool_call_id)

        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_use_id = block.get("tool_use_id")
                if isinstance(tool_use_id, str) and tool_use_id:
                    result_ids.add(tool_use_id)
        return result_ids

    def _summarize_history_records(self, records: list[str]) -> str:
        if not records:
            return ""
        preview_lines = [f"- {self._truncate(record, 120)}" for record in records[:5]]
        remaining = len(records) - len(preview_lines)
        if remaining > 0:
            preview_lines.append(f"- ... ({remaining} more earlier messages)")
        return "\n".join(preview_lines)

    def _load_memory_text(self) -> str:
        settings_root = getattr(self.settings, "workspace_root", None)
        if isinstance(settings_root, Path):
            try:
                return MemoryStore(path=settings_root / "MEMORY.md").load_text()
            except Exception:
                return ""

        candidates: list[Path] = [Path(config.WORKSPACE_ROOT) / "MEMORY.md"]

        for candidate in candidates:
            try:
                text = MemoryStore(path=candidate).load_text()
            except Exception:
                continue
            if text:
                return text
        return ""

    def parse_react_decision(self, raw: str) -> ReactDecision | None:
        if not isinstance(raw, str) or not raw.strip():
            return None

        decision, _ = self.parse_react_decision_with_error(raw)
        return decision

    def _allowed_react_actions(self) -> set[str]:
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
        return allowed_actions

    def parse_react_decision_with_error(self, raw: str) -> tuple[ReactDecision | None, str | None]:
        if not isinstance(raw, str) or not raw.strip():
            return None, None

        allowed_actions = self._allowed_react_actions()
        return parse_react_json_with_error(raw, allowed_actions)

    def _looks_like_react_json(self, raw: str) -> bool:
        if not isinstance(raw, str):
            return False
        stripped = raw.strip()
        if not stripped.startswith("{"):
            return False
        return '"thought"' in stripped or '"action"' in stripped or '"action_input"' in stripped

    def _end_turn_parse_error_observation(self, parse_error: str) -> str:
        return (
            "Observation: ReAct output parse failed at end_turn. "
            f"Error: {parse_error}. "
            "Reply with strict JSON only, exact keys "
            '{"thought","action","action_input"}, action must be allowed tool name or "NONE", '
            'and action_input must be an object or the string "NONE".'
        )

    def _end_turn_empty_response_observation(self) -> str:
        return (
            "Observation: assistant returned an empty response at end_turn. "
            "Reply with either a non-empty final answer for the user, "
            'or strict ReAct JSON with keys {"thought","action","action_input"} if more work is needed.'
        )

    def _parse_finish_output(self, tc, output_text, parse_error) -> dict | None:
        if parse_error:
            return None
        try:
            parsed = json.loads(output_text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, dict):
            return None
        resp_text = parsed.get("response", "")
        if not isinstance(resp_text, str) or not resp_text:
            return None
        return parsed

    def execute_tools(self, response: LLMResponse) -> list[dict]:
        results: list[dict] = []
        tool_statuses: list[str] = []
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

        self._last_executed_tools_count = len(tool_calls)

        for tc in tool_calls:
            inputs_obj = tc.inputs if isinstance(tc.inputs, dict) else {}
            execute_inputs = dict(inputs_obj)
            log_inputs = dict(inputs_obj)
            detail_inputs = dict(inputs_obj)

            allowed, reason = self.security_guard.validate_tool_call(tc.name, execute_inputs)
            if not allowed:
                blocked_output = f"[blocked] {reason}"
                results.append(self.provider.format_tool_result(tc.id, blocked_output))
                tool_statuses.append(f"{tc.name}:blocked")
                self._emit_tool_detail(tc.name, detail_inputs, blocked_output)
                log_event(
                    "tool_result",
                    self.run_ctx,
                    tool=tc.name,
                    output_preview=blocked_output,
                )
                continue

            parse_error = tc.parse_error
            raw_arguments = tc.raw_arguments
            if not isinstance(tc.inputs, dict):
                parse_error = (
                    parse_error or f"tool inputs 必须是 object，实际: {type(tc.inputs).__name__}"
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
            self._tool_call_log.append({"tool": tc.name, "inputs": dict(log_inputs)})
            if self.ui_event_printer is not None:
                self.ui_event_printer(f"  • 工具 {tc.name} 开始")
            tool_start = time.time()
            edit_before_text = (
                self._safe_read_text(execute_inputs.get("path")) if tc.name == "edit_file" else None
            )
            self._start_heartbeat(tc.name, tool_start)
            try:
                if tc.name == "run_bash":
                    with sandbox_cwd(getattr(self.settings, "workspace_root", None)):
                        output = self.tool_registry.execute(tc.name, execute_inputs)
                else:
                    output = self.tool_registry.execute(tc.name, execute_inputs)
            except Exception as exc:
                output = f"[tool_error] {type(exc).__name__}: {exc}"
                tool_statuses.append(f"{tc.name}:error")
                self._stop_heartbeat()
                if self.ui_event_printer is not None:
                    elapsed = time.time() - tool_start
                    self.ui_event_printer(f"  ✗ {tc.name} 失败（{elapsed:.1f}s）")
            else:
                tool_statuses.append(f"{tc.name}:ok")
                self._stop_heartbeat()
                if self.ui_event_printer is not None:
                    elapsed = time.time() - tool_start
                    self.ui_event_printer(f"  ✓ {tc.name} 完成（{elapsed:.1f}s）")
            results.append(self.provider.format_tool_result(tc.id, output))
            output_text = str(output)
            self._emit_tool_detail(tc.name, detail_inputs, output_text, edit_before_text)
            if (
                tc.name == "finish"
                and not output_text.startswith("[error]")
                and not output_text.startswith("[tool_error]")
            ):
                parsed = self._parse_finish_output(tc, output_text, parse_error)
                if parsed is not None:
                    resp_text = parsed.get("response", "")
                    self._finish_requested = True
                    self._finish_response = resp_text
            log_event(
                "tool_result",
                self.run_ctx,
                tool=tc.name,
                output_preview=output_text[:120],
            )
        self._last_tool_statuses = tool_statuses
        return results

    def _print_turn_summary(self, response: LLMResponse) -> None:
        if not self.show_turns or self.turn_printer is None:
            return

        tools_count = self._last_executed_tools_count
        if tools_count == 0 and self._last_tool_statuses:
            tools_count = len(self._last_tool_statuses)

        summary = f"[t{self.state.turn}] stop={response.stop_reason} tools={tools_count}"
        if self._last_tool_statuses:
            summary += " " + " ".join(self._last_tool_statuses)
        self._last_tool_statuses = []
        self._last_executed_tools_count = 0
        self.turn_printer(summary)

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

    def _build_execution_summary(self) -> str:
        if not self._tool_call_log:
            return ""
        lines = []
        tools_used: dict[str, int] = {}
        files_written: list[str] = []
        files_read: list[str] = []
        files_edited: list[str] = []
        commands: list[str] = []
        skills_used: list[str] = []

        for entry in self._tool_call_log:
            tool = entry["tool"]
            inputs = entry.get("inputs", {})
            tools_used[tool] = tools_used.get(tool, 0) + 1

            if tool == "write_file":
                path = inputs.get("path", "")
                if path and path not in files_written:
                    files_written.append(path)
            elif tool == "read_file":
                path = inputs.get("path", "")
                if path and path not in files_read:
                    files_read.append(path)
            elif tool == "edit_file":
                path = inputs.get("path", "")
                if path and path not in files_edited:
                    files_edited.append(path)
            elif tool == "run_bash":
                cmd = inputs.get("command", "")
                if cmd and cmd not in commands:
                    commands.append(cmd)
            elif tool == "use_skill":
                skill = inputs.get("name", "")
                if skill and skill not in skills_used:
                    skills_used.append(skill)

        if skills_used:
            lines.append(f"Skills: {', '.join(skills_used)}")
        if files_written:
            lines.append(f"写入文件: {', '.join(files_written)}")
        if files_edited:
            lines.append(f"编辑文件: {', '.join(files_edited)}")
        if files_read:
            lines.append(f"读取文件: {', '.join(files_read)}")
        if commands:
            cmd_preview = [c[:60] + ("…" if len(c) > 60 else "") for c in commands[:5]]
            lines.append(f"执行命令: {', '.join(cmd_preview)}")

        tool_summary = ", ".join(f"{t}×{c}" for t, c in tools_used.items())
        lines.append(f"工具调用: {tool_summary}")
        return "\n".join(lines)

    def run(self) -> str:
        self._finish_requested = False
        self._finish_response = ""
        while True:
            step = self.next_step(self.state.response)
            if step == Step.MAX_TURNS:
                self.persist_session()
                log_event("run_end", self.run_ctx, stop_reason="max_turns")
                if self.ui_event_printer is not None:
                    self.ui_event_printer("  ⚠ 达到最大轮次")
                return f"[warn] 达到最大轮次 ({self.settings.max_turns})，任务可能未完成。"
            if step == Step.CALL_LLM:
                self.state.turn += 1
                self.run_ctx.turn = self.state.turn
                log_event("turn_start", self.run_ctx)
                if self.ui_event_printer is not None:
                    self.ui_event_printer(f"  ▸ 第{self.state.turn}轮：分析中…")
                response = self.call_llm()
                self.state.response = response
                log_event(
                    "assistant_decision",
                    self.run_ctx,
                    stop_reason=response.stop_reason,
                    tool_calls_count=len(response.tool_calls),
                )
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
                        log_event(
                            "run_end",
                            self.run_ctx,
                            stop_reason="react_format_retry_exhausted",
                        )
                        return REACT_FORMAT_RETRY_EXHAUSTED_PAYLOAD
                    self.ctx.add_user(REACT_FORMAT_RETRY_OBSERVATION)
                    self.state.response = None
                    continue

                results = self.execute_tools(current_response)
                if self._finish_requested:
                    self.persist_session()
                    log_event("run_end", self.run_ctx, stop_reason="finish")
                    if self.ui_event_printer is not None:
                        summary = self._build_execution_summary()
                        if summary:
                            self.ui_event_printer(f"\n  📋 执行摘要\n{summary}")
                        self.ui_event_printer("  ✓ 任务完成")
                    return self._finish_response
                self.state.react_format_retries = 0
                packaged = self.provider.tool_results_as_message(results)
                self.ctx.add_tool_results(packaged)
                self._print_turn_summary(current_response)
                self.state.response = None
                self.persist_session()
                continue
            if step == Step.PERSIST:
                self.persist_session()
                stop_reason = self.state.response.stop_reason if self.state.response else "unknown"
                log_event("run_end", self.run_ctx, stop_reason=stop_reason)
                return self.final_response_text()
            if step == Step.DONE:
                current_response = self.state.response
                if current_response is not None:
                    react_decision, parse_error = self.parse_react_decision_with_error(
                        current_response.text
                    )
                    if react_decision is not None and react_decision.action != "NONE":
                        results = self.execute_tools(current_response)
                        if self._finish_requested:
                            self.persist_session()
                            log_event("run_end", self.run_ctx, stop_reason="finish")
                            if self.ui_event_printer is not None:
                                summary = self._build_execution_summary()
                                if summary:
                                    self.ui_event_printer(f"\n  📋 执行摘要\n{summary}")
                                self.ui_event_printer("  ✓ 任务完成")
                            return self._finish_response
                        packaged = self.provider.tool_results_as_message(results)
                        self.ctx.add_tool_results(packaged)
                        self._print_turn_summary(current_response)
                        self.state.end_turn_react_parse_retries = 0
                        self.state.response = None
                        continue
                    if (
                        react_decision is not None
                        and react_decision.action == "NONE"
                        and react_decision.thought.strip()
                    ):
                        self._print_turn_summary(current_response)
                        self.persist_session()
                        log_event("run_end", self.run_ctx, stop_reason="end_turn")
                        if self.ui_event_printer is not None:
                            summary = self._build_execution_summary()
                            if summary:
                                self.ui_event_printer(f"\n  📋 执行摘要\n{summary}")
                            self.ui_event_printer("  ✓ 任务完成")
                        return react_decision.thought
                    if (
                        react_decision is None
                        and parse_error
                        and self._looks_like_react_json(current_response.text)
                    ):
                        self.state.end_turn_react_parse_retries += 1
                        if (
                            self.state.end_turn_react_parse_retries
                            > END_TURN_REACT_PARSE_MAX_RETRIES
                        ):
                            self.persist_session()
                            log_event(
                                "run_end",
                                self.run_ctx,
                                stop_reason="end_turn_react_parse_retry_exhausted",
                            )
                            return END_TURN_REACT_PARSE_RETRY_EXHAUSTED_PAYLOAD
                        self.ctx.add_user(self._end_turn_parse_error_observation(parse_error))
                        self.state.response = None
                        continue
                    if not str(current_response.text or "").strip():
                        self.state.end_turn_empty_retries += 1
                        if self.state.end_turn_empty_retries > END_TURN_EMPTY_MAX_RETRIES:
                            self.persist_session()
                            log_event(
                                "run_end",
                                self.run_ctx,
                                stop_reason="end_turn_empty_retry_exhausted",
                            )
                            return END_TURN_EMPTY_RETRY_EXHAUSTED_MESSAGE
                        self.ctx.add_user(self._end_turn_empty_response_observation())
                        self.state.response = None
                        continue
                    self.state.end_turn_empty_retries = 0
                self.state.end_turn_react_parse_retries = 0
                self.state.end_turn_empty_retries = 0
                if current_response is not None:
                    self._print_turn_summary(current_response)
                self.persist_session()
                log_event("run_end", self.run_ctx, stop_reason="end_turn")
                if self.ui_event_printer is not None:
                    summary = self._build_execution_summary()
                    if summary:
                        self.ui_event_printer(f"\n  📋 执行摘要\n{summary}")
                    self.ui_event_printer("  ✓ 任务完成")
                return self.final_response_text()
            raise RuntimeError(f"Unhandled runtime step: {step!r}")
