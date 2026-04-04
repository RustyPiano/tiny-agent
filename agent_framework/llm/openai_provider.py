# llm/openai_provider.py
import json

from openai import OpenAI

from agent_framework.llm.base import BaseLLMProvider, LLMResponse, ToolCall


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
        # 官方 OpenAI 端点优先使用 max_completion_tokens；兼容端点沿用 max_tokens
        self._use_max_completion_tokens = base_url is None

        client_kwargs: dict = {}
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        resolved_api_key = _resolve_api_key(api_key=api_key, base_url=base_url)
        if resolved_api_key is not None:
            client_kwargs["api_key"] = resolved_api_key

        self.client = OpenAI(**client_kwargs)

    def chat(self, messages, system, tools, max_tokens=16000) -> LLMResponse:
        # OpenAI system 放在 messages 首条
        full_messages = [{"role": "system", "content": system}] + messages

        # 把 Anthropic tool schema 转成 OpenAI function calling 格式
        oai_tools = [_to_openai_tool(t) for t in tools] if tools else []

        kwargs: dict = dict(model=self.model, messages=full_messages, temperature=0.0)
        if getattr(self, "_use_max_completion_tokens", False):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        if oai_tools:
            kwargs["tools"] = oai_tools

        resp = self.client.chat.completions.create(**kwargs)
        msg = _extract_message(
            resp,
            model=self.model,
            base_url=getattr(self.client, "base_url", None),
        )

        tool_calls = []
        raw_tool_calls = _get_field(msg, "tool_calls") or []
        if isinstance(raw_tool_calls, list):
            for i, tc in enumerate(raw_tool_calls, start=1):
                fn = _get_field(tc, "function")
                name = _get_field(fn, "name")
                if not name:
                    continue
                tool_calls.append(
                    _build_tool_call(
                        call_id=_get_field(tc, "id") or f"tool_call_{i}",
                        name=name,
                        args_payload=_get_field(fn, "arguments", "{}"),
                    )
                )

        if not tool_calls:
            legacy_fc = _get_field(msg, "function_call")
            legacy_name = _get_field(legacy_fc, "name")
            if legacy_name:
                tool_calls.append(
                    _build_tool_call(
                        call_id=_get_field(legacy_fc, "id") or "tool_call_legacy_1",
                        name=legacy_name,
                        args_payload=_get_field(legacy_fc, "arguments", "{}"),
                    )
                )

        text = _message_text(msg)
        stop = "tool_use" if tool_calls else "end_turn"

        # 序列化 assistant 消息
        assistant_message: dict = {"role": "assistant", "content": text}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": (
                            tc.raw_arguments
                            if isinstance(tc.raw_arguments, str)
                            else json.dumps(tc.inputs, ensure_ascii=False)
                        ),
                    },
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


def _resolve_api_key(api_key: str | None, base_url: str | None) -> str | None:
    if api_key:
        return api_key
    if base_url is None:
        # 官方端点下交给 SDK 从 OPENAI_API_KEY 等环境变量读取
        return None
    # 兼容端点通常不校验 key，但 SDK 需要该字段存在
    return "not-needed"


def _get_field(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _message_text(msg) -> str:
    content = _get_field(msg, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
            else:
                text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts)
    return ""


def _extract_message(resp, model: str, base_url) -> object:
    choices = _get_field(resp, "choices")
    if not isinstance(choices, list) or not choices:
        _raise_malformed_response(
            resp,
            model=model,
            base_url=base_url,
            reason="响应缺少 choices，或 choices 为空",
        )

    msg = _get_field(choices[0], "message")
    if msg is None:
        _raise_malformed_response(
            resp,
            model=model,
            base_url=base_url,
            reason="choices[0].message 为空",
        )
    return msg


def _build_tool_call(call_id: str, name: str, args_payload) -> ToolCall:
    parse_error = None
    raw_arguments = args_payload if isinstance(args_payload, str) else None

    if isinstance(args_payload, str):
        try:
            parsed_inputs = json.loads(args_payload) if args_payload.strip() else {}
        except json.JSONDecodeError as e:
            parsed_inputs = {}
            parse_error = f"arguments JSON 解析失败: {e.msg} (pos={e.pos})"
    else:
        parsed_inputs = args_payload
        try:
            raw_arguments = json.dumps(args_payload, ensure_ascii=False)
        except TypeError:
            raw_arguments = str(args_payload)

    if isinstance(parsed_inputs, dict):
        inputs = parsed_inputs
    else:
        inputs = {}
        parse_error = parse_error or (
            f"arguments JSON 必须是 object，实际: {type(parsed_inputs).__name__}"
        )

    return ToolCall(
        id=call_id,
        name=name,
        inputs=inputs,
        parse_error=parse_error,
        raw_arguments=raw_arguments,
    )


def _raise_malformed_response(resp, model: str, base_url, reason: str) -> None:
    error_obj = _get_field(resp, "error")
    error_hint = ""
    if isinstance(error_obj, dict):
        error_hint = error_obj.get("message") or str(error_obj)
    elif error_obj:
        error_hint = str(error_obj)

    provider_hint = (
        "OpenAI 兼容接口返回了非标准 chat.completions 响应。"
        "请检查 base_url/model 是否正确，"
        "并确认该模型支持当前请求格式（尤其是 tools/function calling）。"
    )
    if error_hint:
        provider_hint = f"{provider_hint} 服务端错误: {error_hint}"

    raise RuntimeError(
        f"LLM 响应解析失败: {reason}; model={model!r}; base_url={base_url!r}. {provider_hint}"
    )
