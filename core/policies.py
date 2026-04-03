from __future__ import annotations

from enum import Enum, auto

from llm.base import LLMResponse


class Step(Enum):
    CALL_LLM = auto()
    EXECUTE_TOOLS = auto()
    PERSIST = auto()
    DONE = auto()
    MAX_TURNS = auto()


class RuntimePolicy:
    def next_step(self, turn: int, max_turns: int, response: LLMResponse | None) -> Step:
        raise NotImplementedError


class DefaultRuntimePolicy(RuntimePolicy):
    def next_step(self, turn: int, max_turns: int, response: LLMResponse | None) -> Step:
        if response is None:
            return Step.MAX_TURNS if turn >= max_turns else Step.CALL_LLM
        if response.stop_reason == "tool_use":
            return Step.EXECUTE_TOOLS
        if response.stop_reason == "end_turn":
            return Step.DONE
        return Step.PERSIST
