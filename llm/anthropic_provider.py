# llm/anthropic_provider.py
import anthropic as _anthropic

from config import MAX_TOKENS
from llm.base import BaseLLMProvider, LLMResponse, ToolCall


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.client = _anthropic.Anthropic(api_key=api_key)

    def chat(self, messages, system, tools) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools or [],
            messages=messages,
        )

        text = " ".join(b.text for b in resp.content if hasattr(b, "text") and b.text)
        tool_calls = [
            ToolCall(id=b.id, name=b.name, inputs=b.input)
            for b in resp.content
            if b.type == "tool_use"
        ]
        stop = "tool_use" if tool_calls else "end_turn"

        # 序列化 assistant 消息（Anthropic content block → dict）
        assistant_message = {
            "role": "assistant",
            "content": [_block_to_dict(b) for b in resp.content],
        }
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            assistant_message=assistant_message,
        )

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        # Anthropic 把所有 tool_result 打包进同一条 user 消息
        return [{"role": "user", "content": results}]


def _block_to_dict(block) -> dict:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": block.type}
