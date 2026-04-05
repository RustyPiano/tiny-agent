# core/agent.py
from __future__ import annotations

from typing import TYPE_CHECKING

from agent_framework.core.context import Context
from agent_framework.core.logging import RunContext, log_event
from agent_framework.core.prompt_builder import build_system_prompt
from agent_framework.core.runtime import AgentRuntime
from agent_framework.llm.base import BaseLLMProvider
from agent_framework.llm.factory import create_provider
from agent_framework.sessions import store as session_store
from agent_framework.tools import registry as tool_registry

if TYPE_CHECKING:
    from agent_framework._config import AgentSettings


def _get_provider_type(provider: BaseLLMProvider) -> str:
    """获取 provider 的类型标识"""
    from agent_framework.llm.anthropic_provider import AnthropicProvider
    from agent_framework.llm.openai_provider import OpenAIProvider

    if isinstance(provider, AnthropicProvider):
        return "anthropic"
    if isinstance(provider, OpenAIProvider):
        return "openai"
    return "unknown"


def _resolve_settings_provider_runctx(
    settings: AgentSettings | None,
    provider: BaseLLMProvider | None,
    run_ctx: RunContext | None,
    session_id: str | None,
) -> tuple[AgentSettings, BaseLLMProvider, RunContext]:
    if settings is None:
        from agent_framework._config import AgentSettings as _AS

        settings = _AS.from_env()
    if provider is None:
        provider = create_provider(settings.to_provider_config())
    if run_ctx is None:
        run_ctx = RunContext(session_id=session_id)
    return settings, provider, run_ctx


def _load_history_with_provider_check(
    session_id: str | None,
    provider_type: str,
    run_ctx: RunContext,
    settings: AgentSettings | None = None,
    store=None,
) -> list[dict]:
    if not session_id:
        return []
    active_store = store or _build_session_store(settings)
    history, stored_provider = active_store.load(session_id)
    normalized_provider = (
        stored_provider.strip().lower() if isinstance(stored_provider, str) else ""
    )
    # 兼容旧会话：provider 缺失或空字符串都视为 unknown。
    if normalized_provider in {"", "unknown"}:
        return history
    if history and normalized_provider != provider_type:
        log_event("provider_mismatch", run_ctx, stored=stored_provider, current=provider_type)
        return []
    return history


def _build_session_store(settings: AgentSettings | None):
    sessions_dir = getattr(settings, "sessions_dir", None) if settings is not None else None
    if sessions_dir is None:
        return session_store
    return session_store.create_session_store(sessions_dir)


def run(
    user_input: str,
    settings: AgentSettings | None = None,
    provider: BaseLLMProvider | None = None,
    session_id: str | None = None,
    skills: list[str] | None = None,
    show_turns: bool = False,
    turn_printer=None,
    run_ctx: RunContext | None = None,
    ui_event_printer=None,
) -> str:
    """执行一次完整的 Agent 任务。"""
    return _run_with_runtime(
        user_input,
        settings,
        provider,
        session_id,
        skills,
        show_turns,
        turn_printer,
        run_ctx,
        ui_event_printer,
    )


def _run_with_runtime(
    user_input: str,
    settings: AgentSettings | None,
    provider: BaseLLMProvider | None,
    session_id: str | None,
    skills: list[str] | None,
    show_turns: bool,
    turn_printer,
    run_ctx: RunContext | None,
    ui_event_printer=None,
) -> str:
    settings, provider, run_ctx = _resolve_settings_provider_runctx(
        settings, provider, run_ctx, session_id
    )
    provider_type = _get_provider_type(provider)
    tool_schemas = tool_registry.get_schemas()
    system = build_system_prompt(
        skills,
        enable_subagent_flow=settings.enable_subagent_flow,
        tool_schemas=tool_schemas,
    )
    log_event("run_start", run_ctx, provider=provider_type, max_turns=settings.max_turns)
    active_session_store = _build_session_store(settings)
    history = _load_history_with_provider_check(
        session_id,
        provider_type,
        run_ctx,
        settings=settings,
        store=active_session_store,
    )
    ctx = Context(history)
    ctx.add_user(user_input)
    runtime = AgentRuntime(
        provider=provider,
        settings=settings,
        ctx=ctx,
        tool_registry=tool_registry,
        session_store=active_session_store,
        system=system,
        run_ctx=run_ctx,
        session_id=session_id,
        provider_type=provider_type,
        show_turns=show_turns,
        turn_printer=turn_printer,
        ui_event_printer=ui_event_printer,
    )
    if settings.enable_subagent_flow:
        runtime.enable_subagent_flow(tasks=["task_001"])
    return runtime.run()
