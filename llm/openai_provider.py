# llm/openai_provider.py
import json

from openai import OpenAI

from config import MAX_TOKENS
from llm.base import BaseLLMProvider, LLMResponse, ToolCall


class OpenAIProvider(BaseLLMProvider):
    """
    兼容所有 OpenAI 格式接口：
      OpenAI 官方        base_url=None
      Ollama             base_url="http://localhost:11434/v1", api_key="ollama"
      LM Studio          base_url="http://localhost:1234/v1",  api_key="lm-studio"
      vLLM / any server  base_url="http://host:port/v1"
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def chat(self, messages, system, tools) -> LLMResponse:
        # OpenAI system 放在 messages 首条
        full_messages = [{"role": "system", "content": system}] + messages

        # 把 Anthropic tool schema 转成 OpenAI function calling 格式
        oai_tools = [_to_openai_tool(t) for t in tools] if tools else []

        kwargs: dict = dict(model=self.model, messages=full_messages, max_tokens=MAX_TOKENS)
        if oai_tools:
            kwargs["tools"] = oai_tools

        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    inputs = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    inputs = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, inputs=inputs))

        text = msg.content or ""
        stop = "tool_use" if tool_calls else "end_turn"

        # 序列化 assistant 消息
        assistant_message: dict = {"role": "assistant", "content": text}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.inputs)},
                }
                for tc in tool_calls
            ]
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            assistant_message=assistant_message,
        )

    def format_tool_result(self, tool_call_id: str, content: str) -> dict:
        # OpenAI tool_result 是独立的 role=tool 消息
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    def tool_results_as_message(self, results: list[dict]) -> list[dict]:
        # OpenAI 每条 tool_result 是独立消息，直接返回列表
        return results


def _to_openai_tool(anthropic_schema: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": anthropic_schema["name"],
            "description": anthropic_schema.get("description", ""),
            "parameters": anthropic_schema.get("input_schema", {}),
        },
    }
