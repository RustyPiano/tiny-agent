# core/agent.py
from __future__ import annotations

from typing import TYPE_CHECKING

from core.context import Context
from core.logging import RunContext, log_event
from core.prompt_builder import build_system_prompt
from core.runtime import AgentRuntime
from llm.base import BaseLLMProvider
from llm.factory import create_provider
from sessions import store as session_store
from tools import registry as tool_registry

if TYPE_CHECKING:
    from config import AgentSettings


def _get_provider_type(provider: BaseLLMProvider) -> str:
    """获取 provider 的类型标识"""
    from llm.anthropic_provider import AnthropicProvider
    from llm.openai_provider import OpenAIProvider

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
        from config import AgentSettings as _AS

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
) -> list[dict]:
    if not session_id:
        return []
    history, stored_provider = session_store.load(session_id)
    if history and stored_provider != "unknown" and stored_provider != provider_type:
        log_event("provider_mismatch", run_ctx, stored=stored_provider, current=provider_type)
        return []
    return history


def run(
    user_input: str,
    settings: AgentSettings | None = None,
    provider: BaseLLMProvider | None = None,
    session_id: str | None = None,
    skills: list[str] | None = None,
    verbose: bool = False,
    run_ctx: RunContext | None = None,
) -> str:
    """执行一次完整的 Agent 任务。"""
    return _run_with_runtime(user_input, settings, provider, session_id, skills, verbose, run_ctx)


def _run_with_runtime(
    user_input: str,
    settings: AgentSettings | None,
    provider: BaseLLMProvider | None,
    session_id: str | None,
    skills: list[str] | None,
    verbose: bool,
    run_ctx: RunContext | None,
) -> str:
    # Backward-compatible no-op: verbose 已迁移到结构化日志体系。
    _ = verbose
    settings, provider, run_ctx = _resolve_settings_provider_runctx(
        settings, provider, run_ctx, session_id
    )
    provider_type = _get_provider_type(provider)
    system = build_system_prompt(skills)
    log_event("run_start", run_ctx, provider=provider_type, max_turns=settings.max_turns)
    history = _load_history_with_provider_check(session_id, provider_type, run_ctx)
    ctx = Context(history)
    ctx.add_user(user_input)
    runtime = AgentRuntime(
        provider=provider,
        settings=settings,
        ctx=ctx,
        tool_registry=tool_registry,
        session_store=session_store,
        system=system,
        run_ctx=run_ctx,
        session_id=session_id,
        provider_type=provider_type,
    )
    return runtime.run()
